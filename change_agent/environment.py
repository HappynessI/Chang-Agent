"""GT-free environment reset/step logic for iterative change refinement."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Protocol

import numpy as np

from .action_parser import ActionParser, ActionValidationError
from .adapters.omniovcd_adapter import InitializationResult, PairUpdate
from .executor import ActionExecutor
from .state import AgentAction, AgentObservation, ChangeState, VerifierOutput
from .trajectory import Trajectory, TrajectoryEntry
from .verifier import Verifier
from .verifier_regions import attach_verifier_regions


class StateBackend(Protocol):
    def initialize(
        self, t1_image: np.ndarray, t2_image: np.ndarray, query: str
    ) -> InitializationResult: ...

    def rebuild(
        self, t1_mask: np.ndarray, t2_mask: np.ndarray, evidence: Mapping[str, Any]
    ) -> PairUpdate: ...


class ChangeAgentEnvironment:
    """Owns GT-free state/evidence and exposes model masks needed for editing."""

    def __init__(
        self,
        backend: StateBackend,
        executor: ActionExecutor,
        verifier: Verifier,
        *,
        action_parser: ActionParser | None = None,
        max_steps: int = 8,
        inference_only: bool = True,
        run_metadata: dict[str, Any] | None = None,
        selection_policy: str = "conservative_best",
        selection_epsilon: float = 0.0,
        max_selection_area_delta: float = 0.25,
        max_locality_outside_ratio: float = 0.1,
        max_target_mask_change_ratio: float = 0.25,
        max_component_count_delta: int = 4,
        require_tool_before_finish: bool = True,
        verifier_max_regions: int = 6,
        verifier_max_delta_regions: int = 3,
        verifier_min_region_area: int = 4,
        verifier_region_padding_ratio: float = 0.25,
    ):
        if not inference_only:
            raise ValueError("runtime Environment currently supports inference_only=True only")
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        if not 0 <= max_locality_outside_ratio <= 1:
            raise ValueError("max_locality_outside_ratio must be in [0, 1]")
        if not 0 <= max_target_mask_change_ratio <= 1:
            raise ValueError("max_target_mask_change_ratio must be in [0, 1]")
        if max_component_count_delta < 0:
            raise ValueError("max_component_count_delta must be non-negative")
        if (
            verifier_max_regions < 1
            or verifier_max_delta_regions < 1
            or verifier_min_region_area < 1
        ):
            raise ValueError("Verifier region limits must be positive")
        if verifier_region_padding_ratio < 0:
            raise ValueError("verifier_region_padding_ratio must be non-negative")
        self.backend = backend
        self.executor = executor
        self.verifier = verifier
        self.action_parser = action_parser or ActionParser()
        self.max_steps = max_steps
        self.inference_only = True
        self.selection_policy = selection_policy
        self.selection_epsilon = selection_epsilon
        self.max_selection_area_delta = max_selection_area_delta
        self.max_locality_outside_ratio = max_locality_outside_ratio
        self.max_target_mask_change_ratio = max_target_mask_change_ratio
        self.max_component_count_delta = max_component_count_delta
        self.require_tool_before_finish = require_tool_before_finish
        self.verifier_max_regions = verifier_max_regions
        self.verifier_max_delta_regions = verifier_max_delta_regions
        self.verifier_min_region_area = verifier_min_region_area
        self.verifier_region_padding_ratio = verifier_region_padding_ratio
        self.trajectory = Trajectory(
            run_metadata,
            selection_policy=selection_policy,
            selection_epsilon=selection_epsilon,
            max_area_delta=max_selection_area_delta,
        )
        self.state: ChangeState | None = None
        self.feedback: VerifierOutput | None = None
        self.done = False
        self._small_coordinate_streak = 0
        self._point_session_masks: dict[str, np.ndarray] = {}
        self._accepted_point_clicks: dict[
            str, list[tuple[tuple[int, int], bool]]
        ] = {"t1": [], "t2": []}
        self._rejected_action_signatures: set[str] = set()

    def reset(
        self, t1_image: np.ndarray, t2_image: np.ndarray, query: str
    ) -> AgentObservation:
        if not query.strip():
            raise ValueError("query must not be empty")
        t1_image = np.asarray(t1_image)
        t2_image = np.asarray(t2_image)
        reset_verifier = getattr(self.verifier, "reset", None)
        if callable(reset_verifier):
            reset_verifier()
        initialized = self.backend.initialize(t1_image, t2_image, query)
        self.state = self._state_from_update(
            t1_image,
            t2_image,
            query,
            initialized.t1_mask,
            initialized.t2_mask,
            initialized.update,
            step_index=0,
        )
        self._attach_verifier_regions(self.state, None)
        self.feedback = self.verifier.verify(self.state, None, None, None)
        execution = self._with_verifier_evidence(
            {"event": "reset", "candidate_accepted": True}
        )
        self.done = False
        self._small_coordinate_streak = 0
        self._point_session_masks = {
            "t1": np.array(self.state.t1_mask, dtype=bool, copy=True),
            "t2": np.array(self.state.t2_mask, dtype=bool, copy=True),
        }
        self._accepted_point_clicks = {"t1": [], "t2": []}
        self._rejected_action_signatures = set()
        self.trajectory = Trajectory(
            self.trajectory.run_metadata,
            selection_policy=self.selection_policy,
            selection_epsilon=self.selection_epsilon,
            max_area_delta=self.max_selection_area_delta,
        )
        self.trajectory.append(
            TrajectoryEntry(0, None, None, self.feedback, self.state.clone(), execution)
        )
        return self.observation()

    def observation(self) -> AgentObservation:
        if self.state is None:
            raise RuntimeError("environment must be reset before observation")
        return AgentObservation(
            t1_image=np.array(self.state.t1_image, copy=True),
            t2_image=np.array(self.state.t2_image, copy=True),
            query=self.state.query,
            change_mask=np.array(self.state.change_mask, copy=True),
            feedback=self.feedback,
            history_summary=self.trajectory.history_summary(),
            t1_mask=np.array(self.state.t1_mask, copy=True),
            t2_mask=np.array(self.state.t2_mask, copy=True),
        )

    def step(self, action_or_raw: AgentAction | str) -> tuple[AgentObservation, bool]:
        if self.state is None or self.feedback is None:
            raise RuntimeError("environment must be reset before step")
        if self.done:
            raise RuntimeError("episode has already finished")
        if self.state.step_index >= self.max_steps:
            self.done = True
            return self.observation(), True

        raw_action = action_or_raw if isinstance(action_or_raw, str) else None
        raw_payload = None
        coordinate_warning = None
        if raw_action is not None:
            raw_payload = self.action_parser.extract_payload(raw_action)
            coordinate_warning = self._coordinate_warning(raw_payload)
        action = (
            self.action_parser.parse_payload(raw_payload, self.state.image_size)
            if isinstance(action_or_raw, str)
            else action_or_raw
        )
        action = self.action_parser.validate_pixel_action(action, self.state.image_size)
        if (
            action.action == "finish"
            and self.require_tool_before_finish
            and not any(entry.execution.get("tool") for entry in self.trajectory.entries)
            and not self._initial_finish_authorized()
        ):
            raise ActionValidationError("finish is forbidden before a segmentation tool action")
        action_signature = self._action_signature(self.state, action)
        if (
            action.action != "finish"
            and action_signature in self._rejected_action_signatures
        ):
            raise ActionValidationError(
                "action exactly repeats a previously rejected action on the same accepted state"
            )
        previous_state = self.state.clone()
        previous_feedback = self.feedback
        next_index = self.state.step_index + 1
        execution: dict[str, Any] = {}
        execution["action_signature"] = action_signature
        if raw_payload is not None:
            execution["raw_action_payload"] = raw_payload
        if coordinate_warning is not None:
            execution["coordinate_warning"] = coordinate_warning

        if action.action == "finish":
            candidate = self.state.clone()
            candidate.step_index = next_index
            execution["event"] = "finish_requested"
        else:
            target_image = (
                self.state.t1_image if action.target_view == "t1" else self.state.t2_image
            )
            target_mask = (
                self.state.t1_mask if action.target_view == "t1" else self.state.t2_mask
            )
            result = self.executor.execute(
                action,
                target_image,
                target_mask,
                self.state.query,
                point_session_mask=self._point_session_masks[action.target_view],
                point_click_history=tuple(
                    self._accepted_point_clicks[action.target_view]
                ),
            )
            t1_mask = result.mask if action.target_view == "t1" else self.state.t1_mask
            t2_mask = result.mask if action.target_view == "t2" else self.state.t2_mask
            evidence = dict(self.state.evidence)
            evidence.update(result.evidence)
            evidence["last_target_view"] = action.target_view
            update = self.backend.rebuild(t1_mask, t2_mask, evidence)
            candidate = self._state_from_update(
                self.state.t1_image,
                self.state.t2_image,
                self.state.query,
                t1_mask,
                t2_mask,
                update,
                step_index=next_index,
            )
            execution.update(result.evidence)

        previous_area_ratio = float(previous_state.change_mask.mean())
        candidate_area_ratio = float(candidate.change_mask.mean())
        area_delta = abs(candidate_area_ratio - previous_area_ratio)
        hard_rejection_reasons: list[str] = []
        if action.action != "finish":
            if area_delta > self.max_selection_area_delta:
                hard_rejection_reasons.append("mask_area_delta_exceeded")
            locality = execution.get("locality", {})
            if locality.get("outside_roi_ratio", 0.0) > self.max_locality_outside_ratio:
                hard_rejection_reasons.append("locality_outside_roi_exceeded")
            if (
                locality.get("target_mask_change_ratio", 0.0)
                > self.max_target_mask_change_ratio
            ):
                hard_rejection_reasons.append("target_mask_change_exceeded")
            if (
                abs(locality.get("component_count_delta", 0))
                > self.max_component_count_delta
            ):
                hard_rejection_reasons.append("component_count_delta_exceeded")

        verifier_skipped = bool(hard_rejection_reasons)
        if verifier_skipped:
            verifier_output = VerifierOutput(
                quality_score=None,
                progress_score=None,
                comparison="worse",
                error_type=previous_feedback.error_type,
                target_view=previous_feedback.target_view,
                error_region=previous_feedback.error_region,
                suggested_action=previous_feedback.suggested_action,
                feedback=(
                    "Candidate rejected before Qwen verification by hard runtime gates: "
                    + ", ".join(hard_rejection_reasons)
                ),
                accept=False,
                verifier_valid=True,
                localization_valid=previous_feedback.localization_valid,
                stop=False,
            )
            execution["verifier_evidence"] = {
                "type": "runtime_hard_gate_precheck",
                "verifier_skipped": True,
                "rejection_reasons": list(hard_rejection_reasons),
                "gt_available": False,
            }
        else:
            self._attach_verifier_regions(candidate, previous_state)
            verifier_output = self.verifier.verify(
                candidate, self.feedback.quality_score, action, previous_state
            )

        rejection_reasons = list(hard_rejection_reasons)
        if not verifier_skipped and not verifier_output.verifier_valid:
            rejection_reasons.append("verifier_invalid")
        if verifier_output.comparison is not None:
            ranking_progress = {
                "better": 1.0,
                "worse": -1.0,
                "initial": 0.0,
                "unchanged": 0.0,
                "uncertain": 0.0,
            }[verifier_output.comparison]
        else:
            ranking_progress = (
                verifier_output.progress_score
                if verifier_output.progress_score is not None
                else verifier_output.score_delta
            )
        if action.action != "finish" and not verifier_skipped:
            if verifier_output.comparison is not None:
                if verifier_output.comparison != "better":
                    rejection_reasons.append("candidate_effect_not_better")
            elif ranking_progress <= self.selection_epsilon:
                rejection_reasons.append("progress_did_not_improve")
        candidate_accepted = not rejection_reasons
        execution.update(
            {
                "candidate_accepted": candidate_accepted,
                "candidate_rejection_reasons": rejection_reasons,
                "previous_area_ratio": previous_area_ratio,
                "candidate_area_ratio": candidate_area_ratio,
                "candidate_area_delta": area_delta,
                "ranking_progress": ranking_progress,
                "pairwise_comparison": verifier_output.comparison,
                "verifier_skipped_by_hard_gate": verifier_skipped,
            }
        )
        if not verifier_skipped:
            execution = self._with_verifier_evidence(execution)
        if candidate_accepted:
            self.state = candidate
            self.feedback = verifier_output
            if action.action in {"positive_point", "negative_point"}:
                if action.coordinate is None:
                    raise RuntimeError("accepted point action has no coordinate")
                self._accepted_point_clicks[action.target_view].append(
                    (action.coordinate, action.action == "positive_point")
                )
            elif action.action == "box":
                accepted_target_mask = (
                    candidate.t1_mask
                    if action.target_view == "t1"
                    else candidate.t2_mask
                )
                self._point_session_masks[action.target_view] = np.array(
                    accepted_target_mask, dtype=bool, copy=True
                )
                self._accepted_point_clicks[action.target_view] = []
        else:
            # Keep the rejected candidate in the trajectory, but continue the
            # closed loop from the last accepted state and feedback.
            previous_state.step_index = next_index
            self.state = previous_state
            self.feedback = previous_feedback
            reject_callback = getattr(self.verifier, "on_candidate_rejected", None)
            if callable(reject_callback):
                reject_callback(previous_feedback)
            if action.action != "finish":
                self._rejected_action_signatures.add(action_signature)
        self.done = (
            next_index >= self.max_steps
            or (
                candidate_accepted
                and action.action == "finish"
                and verifier_output.stop
            )
        )
        self.trajectory.append(
            TrajectoryEntry(
                next_index,
                raw_action,
                action,
                verifier_output,
                candidate.clone(),
                execution,
            )
        )
        return self.observation(), self.done

    def _coordinate_warning(self, payload: dict[str, Any]) -> str | None:
        values = payload.get("coordinate") or payload.get("box")
        if values is None:
            self._small_coordinate_streak = 0
            return None
        if all(isinstance(value, (int, float)) and 0 <= value <= 255 for value in values):
            self._small_coordinate_streak += 1
        else:
            self._small_coordinate_streak = 0
        if self._small_coordinate_streak >= 2:
            return (
                "consecutive_public_coordinates_all_le_255; interpreted as normalized_1000_xy, "
                "not auto-corrected to pixels"
            )
        return None

    def _with_verifier_evidence(self, execution: dict[str, Any]) -> dict[str, Any]:
        result = dict(execution)
        evidence = getattr(self.verifier, "last_evidence", None)
        if evidence:
            result["verifier_evidence"] = evidence
        return result

    @staticmethod
    def _action_signature(state: ChangeState, action: AgentAction) -> str:
        digest = hashlib.sha256()
        for mask in (state.t1_mask, state.t2_mask, state.change_mask):
            value = np.ascontiguousarray(mask, dtype=np.uint8)
            digest.update(str(value.shape).encode("ascii"))
            digest.update(value.tobytes())
        digest.update(
            json.dumps(
                action.to_dict(), sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        )
        return digest.hexdigest()

    def _initial_finish_authorized(self) -> bool:
        """Allow a verified, error-free initial state to finish without a no-op tool."""

        feedback = self.feedback
        return bool(
            feedback is not None
            and feedback.verifier_valid
            and feedback.comparison == "initial"
            and feedback.error_type == "none"
            and feedback.stop
        )

    def _attach_verifier_regions(
        self, state: ChangeState, previous_state: ChangeState | None
    ) -> None:
        attach_verifier_regions(
            state,
            previous_state,
            max_regions=self.verifier_max_regions,
            max_delta_regions=self.verifier_max_delta_regions,
            min_component_area=self.verifier_min_region_area,
            padding_ratio=self.verifier_region_padding_ratio,
        )

    @property
    def best_state(self) -> ChangeState:
        return self.trajectory.best_entry.state.clone()

    @staticmethod
    def _state_from_update(
        t1_image: np.ndarray,
        t2_image: np.ndarray,
        query: str,
        t1_mask: np.ndarray,
        t2_mask: np.ndarray,
        update: PairUpdate,
        step_index: int,
    ) -> ChangeState:
        return ChangeState(
            t1_image=t1_image,
            t2_image=t2_image,
            query=query,
            t1_mask=t1_mask,
            t2_mask=t2_mask,
            change_mask=update.change_mask,
            t1_instances=update.t1_instances,
            t2_instances=update.t2_instances,
            matching=update.matching,
            evidence=update.evidence,
            step_index=step_index,
        )
