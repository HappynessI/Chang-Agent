"""Full-context Qwen verifier without mask-derived Proposal grounding.

Used only for the Direct arm of the Proposal ablation.  It deliberately gives
the model global visual context and lets it author action geometry, while the
Environment still owns coordinate parsing, tool execution, and safety gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from ..action_parser import ActionParser, ActionValidationError
from ..state import AgentAction, ChangeState, VerifierOutput
from ..verifier_protocol import ERROR_TYPES, StageBackend, StageProtocolError


# The direct arm has no region diagnosis schema to keep its vocabulary in line
# with. Qwen nevertheless commonly uses these plain-language equivalents.
# Canonicalize them at the boundary rather than discarding a structurally safe
# full-state verdict and biasing the ablation toward a no-op.
_ERROR_TYPE_ALIASES = {
    "false_negative_change": "false_negative",
    "missing_detection": "false_negative",
    "missing_t1_mask": "false_negative",
    "missing_t2_mask": "false_negative",
    "under_detection": "false_negative",
    "false_positive": "false_positive_change",
    "over_detection": "false_positive_change",
    "spurious_detection": "false_positive_change",
}
_ACTIONS = {"positive_point", "negative_point", "box", "finish"}
_COMPARISONS = {"initial", "better", "worse", "unchanged", "uncertain"}


@dataclass(frozen=True)
class _DirectVerdict:
    comparison: str
    quality_score: float
    progress_score: float
    accept: bool
    error_type: str
    target_view: str | None
    suggested_action: str
    coordinate_normalized_1000: tuple[int, int] | None
    box_normalized_1000: tuple[int, int, int, int] | None
    feedback: str


class DirectQwenVerifier:
    """Global visual diagnosis/action grounding ablation without Proposals."""

    SCHEMA_VERSION = "direct_full_context_v1"

    def __init__(
        self,
        backend: StageBackend,
        *,
        accept_threshold: float = 0.82,
        max_retries: int = 2,
    ):
        if not 0 <= accept_threshold <= 1:
            raise ValueError("accept_threshold must be in [0,1]")
        if max_retries < 1:
            raise ValueError("max_retries must be positive")
        self.backend = backend
        self.accept_threshold = accept_threshold
        self.max_retries = max_retries
        self.last_evidence: dict[str, Any] = {}
        self._last_valid_output: VerifierOutput | None = None

    def reset(self) -> None:
        self.last_evidence = {}
        self._last_valid_output = None
        reset_audit = getattr(self.backend, "reset_audit", None)
        if callable(reset_audit):
            reset_audit()

    def on_candidate_rejected(self, previous_feedback: VerifierOutput) -> None:
        self._last_valid_output = (
            previous_feedback if previous_feedback.verifier_valid else None
        )

    def replan_after_rejection(
        self,
        accepted_state: ChangeState,
        rejected_candidate: ChangeState,
        accepted_feedback: VerifierOutput,
        rejected_feedback: VerifierOutput,
        rejected_action: AgentAction,
        rejection_reasons: Sequence[str],
        rejection_history: Sequence[Mapping[str, Any]],
    ) -> VerifierOutput:
        """Author a new Direct action after Environment rolls a candidate back.

        The accepted masks remain the current state.  The rejected candidate is
        attached only as visual and structured evidence so Qwen can avoid the
        failed edit.  This is deliberately a replan, not a candidate-quality
        decision, therefore its contract requires ``comparison=uncertain`` and
        ``accept=false``.
        """

        audit_start = self._audit_length()
        payload = {
            "mode": "replan",
            "schema": "direct_replan_v1",
            "accepted_score": accepted_feedback.quality_score,
            "accepted_feedback": accepted_feedback.to_dict(),
            "rejected_action": rejected_action.to_dict(),
            "rejection_reasons": list(rejection_reasons),
            "rejection_history": [dict(item) for item in rejection_history],
            "rejected_candidate_verdict": rejected_feedback.to_dict(),
            "rejected_candidate_mask_delta": _mask_delta_summary(
                accepted_state, rejected_candidate
            ),
        }
        try:
            verdict = self._run_stage(
                accepted_state,
                payload,
                mode="replan",
                previous_state=rejected_candidate,
            )
            replanned_action = _verdict_action(verdict, accepted_state.image_size)
            if replanned_action == rejected_action:
                raise StageProtocolError(
                    "direct rollback replan repeated the rejected action"
                )
            output = self._output(
                verdict, accepted_feedback.quality_score, initial=False
            )
            self.last_evidence = {
                "type": "direct_qwen_verifier",
                "schema_version": self.SCHEMA_VERSION,
                "decision_mode": "qwen_full_context_direct_replan",
                "proposal_mode": "direct",
                "verifier_valid": True,
                "localization_valid": output.localization_valid,
                "replan": {
                    "rejected_action": rejected_action.to_dict(),
                    "rejection_reasons": list(rejection_reasons),
                    "rejection_history": payload["rejection_history"],
                    "rejected_candidate_mask_delta": payload[
                        "rejected_candidate_mask_delta"
                    ],
                },
                "direct_verdict": verdict.__dict__,
                "validation_errors": [],
                "backend_calls": self._backend_calls_since(audit_start),
            }
            self._last_valid_output = output
            return output
        except (KeyError, TypeError, ValueError, StageProtocolError) as error:
            self.last_evidence = {
                "type": "direct_qwen_verifier",
                "schema_version": self.SCHEMA_VERSION,
                "decision_mode": "qwen_full_context_direct_replan",
                "proposal_mode": "direct",
                "verifier_valid": False,
                "localization_valid": False,
                "replan": {
                    "rejected_action": rejected_action.to_dict(),
                    "rejection_reasons": list(rejection_reasons),
                    "rejection_history": [dict(item) for item in rejection_history],
                },
                "validation_errors": [str(error)],
                "backend_calls": self._backend_calls_since(audit_start),
            }
            return VerifierOutput(
                quality_score=accepted_feedback.quality_score,
                progress_score=0.0,
                comparison="uncertain",
                error_type=accepted_feedback.error_type,
                target_view=accepted_feedback.target_view,
                error_region=None,
                suggested_action=None,
                feedback="Direct rollback replan was invalid; no action is authorized.",
                accept=False,
                verifier_valid=False,
                localization_valid=False,
                stop=False,
            )

    def verify(
        self,
        state: ChangeState,
        previous_score: float | None,
        previous_action: AgentAction | None,
        previous_state: ChangeState | None = None,
    ) -> VerifierOutput:
        audit_start = self._audit_length()
        try:
            if previous_state is not None and self._states_identical(state, previous_state):
                previous = self._last_valid_output
                output = VerifierOutput(
                    quality_score=previous.quality_score if previous else previous_score,
                    progress_score=0.0,
                    score_delta=0.0,
                    comparison="unchanged",
                    error_type=previous.error_type if previous else "uncertain_region",
                    target_view=previous.target_view if previous else "t2",
                    error_region=previous.error_region if previous else None,
                    suggested_action=previous.suggested_action if previous else None,
                    feedback="Candidate masks are identical to the accepted state.",
                    accept=bool(previous and previous.accept),
                    verifier_valid=True,
                    localization_valid=bool(previous and previous.localization_valid),
                    stop=bool(previous and previous.stop),
                )
                self.last_evidence = {
                    "type": "direct_qwen_verifier",
                    "schema_version": self.SCHEMA_VERSION,
                    "decision_mode": "programmatic_identical_state",
                    "proposal_mode": "direct",
                    "verifier_valid": True,
                    "localization_valid": output.localization_valid,
                    "validation_errors": [],
                    "backend_calls": self._backend_calls_since(audit_start),
                }
                return output
            initial = previous_state is None
            mode = "initial" if initial else "candidate"
            verdict = self._run_stage(
                state,
                {
                    "mode": mode,
                    "schema": "direct_verdict_v1",
                    "previous_score": previous_score,
                    "previous_action": previous_action.to_dict() if previous_action else None,
                },
                mode=mode,
                previous_state=previous_state,
            )
            output = self._output(verdict, previous_score, initial=initial)
            self.last_evidence = {
                "type": "direct_qwen_verifier",
                "schema_version": self.SCHEMA_VERSION,
                "decision_mode": "qwen_full_context_direct_action",
                "proposal_mode": "direct",
                "verifier_valid": True,
                "localization_valid": output.localization_valid,
                "direct_verdict": verdict.__dict__,
                "validation_errors": [],
                "backend_calls": self._backend_calls_since(audit_start),
            }
            self._last_valid_output = output
            return output
        except (KeyError, TypeError, ValueError, StageProtocolError) as error:
            previous = self._last_valid_output
            self.last_evidence = {
                "type": "direct_qwen_verifier",
                "schema_version": self.SCHEMA_VERSION,
                "proposal_mode": "direct",
                "verifier_valid": False,
                "localization_valid": False,
                "validation_errors": [str(error)],
                "backend_calls": self._backend_calls_since(audit_start),
            }
            return VerifierOutput(
                quality_score=previous.quality_score if previous else None,
                progress_score=0.0,
                comparison="uncertain",
                error_type=previous.error_type if previous else "uncertain_region",
                target_view=(
                    previous.target_view
                    if previous
                    else previous_action.target_view if previous_action else "t2"
                ),
                error_region=previous.error_region if previous else None,
                suggested_action=None,
                feedback="Direct verifier output was invalid; no action is authorized.",
                accept=False,
                verifier_valid=False,
                localization_valid=False,
                stop=False,
            )

    def _run_stage(
        self,
        state: ChangeState,
        payload: Mapping[str, Any],
        *,
        mode: str,
        previous_state: ChangeState | None,
    ) -> _DirectVerdict:
        errors: list[str] = []
        repair = getattr(self.backend, "repair_stage", None)
        for attempt in range(self.max_retries):
            try:
                raw = (
                    self.backend.generate_stage("direct", state, payload, previous_state)
                    if attempt == 0
                    else repair("direct", state, payload, errors[-1], previous_state)
                    if callable(repair)
                    else None
                )
                if raw is None:
                    break
                return _parse_direct_verdict(raw, mode=mode)
            except (KeyError, TypeError, ValueError, StageProtocolError) as error:
                errors.append(str(error))
        if not errors:
            raise StageProtocolError("direct backend does not support repair retries")
        raise StageProtocolError(
            f"direct failed after {len(errors)} attempt(s): {errors[-1]}"
        )

    def _output(
        self,
        verdict: _DirectVerdict,
        previous_score: float | None,
        *,
        initial: bool,
    ) -> VerifierOutput:
        region = (
            (verdict.coordinate_normalized_1000[0], verdict.coordinate_normalized_1000[1],
             verdict.coordinate_normalized_1000[0], verdict.coordinate_normalized_1000[1])
            if verdict.coordinate_normalized_1000 is not None
            else verdict.box_normalized_1000
        )
        accept = (
            verdict.error_type == "none" and verdict.quality_score >= self.accept_threshold
            if initial
            else verdict.accept and verdict.comparison == "better"
        )
        stop = bool(
            accept
            and verdict.error_type == "none"
            and verdict.suggested_action == "finish"
        )
        return VerifierOutput(
            quality_score=verdict.quality_score,
            progress_score=verdict.progress_score,
            score_delta=(
                0.0 if previous_score is None else verdict.quality_score - previous_score
            ),
            comparison=verdict.comparison,
            error_type=verdict.error_type,
            target_view=verdict.target_view or "t2",
            error_region=region,
            suggested_action=verdict.suggested_action,
            feedback=verdict.feedback,
            accept=accept,
            verifier_valid=True,
            localization_valid=(verdict.suggested_action == "finish" or region is not None),
            stop=stop,
        )

    def _audit_length(self) -> int:
        history = getattr(self.backend, "call_history", None)
        return len(history) if isinstance(history, list) else 0

    def _backend_calls_since(self, start: int) -> list[dict[str, Any]]:
        history = getattr(self.backend, "call_history", None)
        if not isinstance(history, list):
            return []
        return [dict(item) for item in history[start:] if isinstance(item, Mapping)]

    @staticmethod
    def _states_identical(state: ChangeState, previous_state: ChangeState) -> bool:
        return (
            (state.t1_mask == previous_state.t1_mask).all()
            and (state.t2_mask == previous_state.t2_mask).all()
            and (state.change_mask == previous_state.change_mask).all()
        )


def _parse_direct_verdict(payload: Mapping[str, Any], *, mode: str) -> _DirectVerdict:
    if set(payload) != {"verdict"} or not isinstance(payload["verdict"], Mapping):
        raise StageProtocolError("direct response must contain exactly a verdict object")
    body = payload["verdict"]
    expected = {
        "comparison",
        "quality_score",
        "progress_score",
        "accept",
        "error_type",
        "target_view",
        "suggested_action",
        "coordinate_normalized_1000",
        "box_normalized_1000",
        "feedback",
    }
    if set(body) != expected:
        raise StageProtocolError(
            f"direct verdict must contain exactly {sorted(expected)}; got {sorted(body)}"
        )
    if mode not in {"initial", "candidate", "replan"}:
        raise StageProtocolError(f"unsupported direct verdict mode: {mode!r}")
    initial = mode == "initial"
    comparison = str(body["comparison"])
    if comparison not in _COMPARISONS or (initial and comparison != "initial") or (
        mode == "candidate" and comparison == "initial"
    ) or (mode == "replan" and comparison != "uncertain"):
        raise StageProtocolError("direct verdict comparison is invalid for this state")
    quality = _number(body["quality_score"], "quality_score", 0.0, 1.0)
    progress = _number(body["progress_score"], "progress_score", -1.0, 1.0)
    if not isinstance(body["accept"], bool):
        raise StageProtocolError("direct verdict accept must be a JSON boolean")
    accept = bool(body["accept"])
    error_type = _canonical_error_type(body["error_type"])
    if error_type not in ERROR_TYPES:
        raise StageProtocolError("direct verdict error_type is unsupported")
    if mode == "replan" and accept:
        raise StageProtocolError("direct replan verdict accept must be false")
    if mode == "candidate" and accept and comparison != "better":
        raise StageProtocolError("direct candidate accept=true requires comparison=better")
    action = str(body["suggested_action"])
    if action not in _ACTIONS:
        raise StageProtocolError("direct verdict suggested_action is unsupported")
    target = body["target_view"]
    coordinate = _point(body["coordinate_normalized_1000"])
    box = _box(body["box_normalized_1000"])
    if error_type == "none":
        if target is not None or action != "finish" or coordinate is not None or box is not None:
            raise StageProtocolError("none direct verdict must finish with null target and geometry")
    else:
        if target not in {"t1", "t2"} or action == "finish":
            raise StageProtocolError("direct error verdict needs target_view and tool action")
        if action in {"positive_point", "negative_point"}:
            if coordinate is None or box is not None:
                raise StageProtocolError("direct point action needs only coordinate_normalized_1000")
        elif box is None or coordinate is not None:
            raise StageProtocolError("direct box action needs only box_normalized_1000")
    return _DirectVerdict(
        comparison=comparison,
        quality_score=quality,
        progress_score=progress,
        accept=accept,
        error_type=error_type,
        target_view=target,
        suggested_action=action,
        coordinate_normalized_1000=coordinate,
        box_normalized_1000=box,
        feedback=str(body["feedback"]),
    )


def _canonical_error_type(value: Any) -> str:
    raw = str(value).strip().lower().replace("-", " ").replace(" ", "_")
    return _ERROR_TYPE_ALIASES.get(raw, raw)


def _number(value: Any, name: str, lower: float, upper: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StageProtocolError(f"direct verdict {name} must be numeric")
    result = float(value)
    if not lower <= result <= upper:
        raise StageProtocolError(f"direct verdict {name} is outside [{lower}, {upper}]")
    return result


def _point(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2 or any(
        isinstance(item, bool) or not isinstance(item, int) or not 0 <= item <= 1000
        for item in value
    ):
        raise StageProtocolError("direct coordinate_normalized_1000 must be two [0,1000] integers")
    return int(value[0]), int(value[1])


def _box(value: Any) -> tuple[int, int, int, int] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 4 or any(
        isinstance(item, bool) or not isinstance(item, int) or not 0 <= item <= 1000
        for item in value
    ):
        raise StageProtocolError("direct box_normalized_1000 must contain four [0,1000] integers")
    x1, y1, x2, y2 = (int(item) for item in value)
    if x1 >= x2 or y1 >= y2:
        raise StageProtocolError("direct box_normalized_1000 must be ordered")
    return x1, y1, x2, y2


def _mask_delta_summary(
    accepted_state: ChangeState, rejected_candidate: ChangeState
) -> dict[str, int]:
    """Return GT-free changed-pixel counts for a rejected Direct candidate."""

    return {
        "t1_changed_pixels": int(
            (accepted_state.t1_mask != rejected_candidate.t1_mask).sum()
        ),
        "t2_changed_pixels": int(
            (accepted_state.t2_mask != rejected_candidate.t2_mask).sum()
        ),
        "change_changed_pixels": int(
            (accepted_state.change_mask != rejected_candidate.change_mask).sum()
        ),
    }


def _verdict_action(
    verdict: _DirectVerdict, image_size: tuple[int, int]
) -> AgentAction:
    """Convert Direct public geometry to the same pixel action Environment executes."""

    payload: dict[str, Any] = {
        "target_view": verdict.target_view or "t2",
        "action": verdict.suggested_action,
    }
    if verdict.coordinate_normalized_1000 is not None:
        payload["coordinate"] = list(verdict.coordinate_normalized_1000)
    if verdict.box_normalized_1000 is not None:
        payload["box"] = list(verdict.box_normalized_1000)
    try:
        return ActionParser().parse_payload(payload, image_size)
    except ActionValidationError as error:
        raise StageProtocolError(
            f"direct verdict cannot be converted to an executable action: {error}"
        ) from error
