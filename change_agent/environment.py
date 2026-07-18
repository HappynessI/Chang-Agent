"""GT-free environment reset/step logic for iterative change refinement."""

from __future__ import annotations

from typing import Any, Mapping, Protocol

import numpy as np

from .action_parser import ActionParser
from .adapters.omniovcd_adapter import InitializationResult, PairUpdate
from .executor import ActionExecutor
from .state import AgentAction, AgentObservation, ChangeState, VerifierOutput
from .trajectory import Trajectory, TrajectoryEntry
from .verifier import Verifier


class StateBackend(Protocol):
    def initialize(
        self, t1_image: np.ndarray, t2_image: np.ndarray, query: str
    ) -> InitializationResult: ...

    def rebuild(
        self, t1_mask: np.ndarray, t2_mask: np.ndarray, evidence: Mapping[str, Any]
    ) -> PairUpdate: ...


class ChangeAgentEnvironment:
    """Owns hidden masks/evidence and exposes only a restricted Agent observation."""

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
    ):
        if not inference_only:
            raise ValueError("runtime Environment currently supports inference_only=True only")
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.backend = backend
        self.executor = executor
        self.verifier = verifier
        self.action_parser = action_parser or ActionParser()
        self.max_steps = max_steps
        self.inference_only = True
        self.selection_policy = selection_policy
        self.selection_epsilon = selection_epsilon
        self.max_selection_area_delta = max_selection_area_delta
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

    def reset(
        self, t1_image: np.ndarray, t2_image: np.ndarray, query: str
    ) -> AgentObservation:
        if not query.strip():
            raise ValueError("query must not be empty")
        t1_image = np.asarray(t1_image)
        t2_image = np.asarray(t2_image)
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
        self.feedback = self.verifier.verify(self.state, None, None)
        execution = self._with_verifier_evidence({"event": "reset"})
        self.done = False
        self._small_coordinate_streak = 0
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
        next_index = self.state.step_index + 1
        execution: dict[str, Any] = {}
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
            result = self.executor.execute(action, target_image, target_mask, self.state.query)
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

        verifier_output = self.verifier.verify(
            candidate, self.feedback.quality_score, action
        )
        execution = self._with_verifier_evidence(execution)
        self.state = candidate
        self.feedback = verifier_output
        self.done = (
            next_index >= self.max_steps
            or (action.action == "finish" and verifier_output.accept)
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
