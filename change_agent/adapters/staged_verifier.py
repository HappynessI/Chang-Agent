"""Stage-oriented MLLM verifier.

This adapter deliberately keeps model responses small and typed.  It can be
used with a local Qwen model or a hosted backend implementing ``StageBackend``.
The Environment remains the authority for proposal geometry and mask
editability; the model cannot invent either.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from ..state import AgentAction, ChangeState, VerifierOutput
from ..verifier_regions import build_verifier_regions
from ..verifier_protocol import (
    ActionPlan,
    Decision,
    Diagnosis,
    EvidenceJudgment,
    EvidenceRecord,
    StageBackend,
    StageProtocolError,
    StageTrace,
)


class StagedQwenVerifier:
    """Run evidence, diagnosis, planning, and decision as separate stages."""

    SCHEMA_VERSION = "staged_verifier_v1"

    def __init__(
        self,
        backend: StageBackend,
        *,
        accept_threshold: float = 0.82,
        max_regions: int = 6,
        max_retries: int = 2,
    ):
        if not 0 <= accept_threshold <= 1:
            raise ValueError("accept_threshold must be in [0,1]")
        if max_regions < 1:
            raise ValueError("max_regions must be positive")
        if max_retries < 1:
            raise ValueError("max_retries must be positive")
        self.backend = backend
        self.accept_threshold = accept_threshold
        self.max_regions = max_regions
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

    def verify(
        self,
        state: ChangeState,
        previous_score: float | None,
        previous_action: AgentAction | None,
        previous_state: ChangeState | None = None,
    ) -> VerifierOutput:
        audit_start = self._audit_length()
        try:
            if previous_state is not None and self._states_identical(
                state, previous_state
            ):
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
                    accept=False,
                    verifier_valid=True,
                    localization_valid=bool(previous and previous.localization_valid),
                    stop=False,
                )
                self.last_evidence = {
                    "type": "staged_qwen_verifier",
                    "schema_version": self.SCHEMA_VERSION,
                    "decision_mode": "programmatic_identical_state",
                    "verifier_valid": True,
                    "localization_valid": output.localization_valid,
                    "validation_errors": [],
                    "backend_calls": self._backend_calls_since(audit_start),
                }
                return output
            proposals = list(state.evidence.get("verifier_region_proposals", []))
            records = tuple(EvidenceRecord.from_proposal(item) for item in proposals)
            if not records:
                raise StageProtocolError("no Environment proposal is available")
            if previous_state is None:
                output, trace = self._verify_initial(state, records, previous_score)
            else:
                output, trace = self._verify_candidate(
                    state, previous_state, records, previous_score
                )
            self.last_evidence = {
                "type": "staged_qwen_verifier",
                "schema_version": self.SCHEMA_VERSION,
                "verifier_valid": output.verifier_valid,
                "localization_valid": output.localization_valid,
                "stage_trace": trace.to_dict(),
                "validation_errors": [],
                "backend_calls": self._backend_calls_since(audit_start),
            }
            if output.verifier_valid:
                self._last_valid_output = output
            return output
        except (KeyError, TypeError, ValueError, StageProtocolError) as error:
            self.last_evidence = {
                "type": "staged_qwen_verifier",
                "schema_version": self.SCHEMA_VERSION,
                "verifier_valid": False,
                "localization_valid": False,
                "validation_errors": [str(error)],
                "backend_calls": self._backend_calls_since(audit_start),
            }
            previous = self._last_valid_output
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
                feedback="Staged verifier output was invalid; no action is authorized.",
                accept=False,
                verifier_valid=False,
                localization_valid=False,
                stop=False,
            )

    def _audit_length(self) -> int:
        history = getattr(self.backend, "call_history", None)
        return len(history) if isinstance(history, list) else 0

    def _backend_calls_since(self, start: int) -> list[dict[str, Any]]:
        history = getattr(self.backend, "call_history", None)
        if not isinstance(history, list):
            return []
        return [dict(item) for item in history[start:] if isinstance(item, Mapping)]

    def _run_stage(
        self,
        stage: str,
        state: ChangeState,
        payload: Mapping[str, Any],
        parser: Callable[[Mapping[str, Any]], Any],
        previous_state: ChangeState | None = None,
    ) -> Any:
        """Generate, validate, and repair one stage without weakening its schema."""

        errors: list[str] = []
        repair = getattr(self.backend, "repair_stage", None)
        for attempt in range(self.max_retries):
            try:
                if attempt == 0:
                    raw = self.backend.generate_stage(
                        stage, state, payload, previous_state
                    )
                elif callable(repair):
                    raw = repair(
                        stage,
                        state,
                        payload,
                        errors[-1],
                        previous_state,
                    )
                else:
                    break
                return parser(raw)
            except (KeyError, TypeError, ValueError, StageProtocolError) as error:
                errors.append(str(error))
        if not errors:
            raise StageProtocolError(f"{stage} backend does not support repair retries")
        raise StageProtocolError(
            f"{stage} failed after {len(errors)} attempt(s): {errors[-1]}"
        )

    @staticmethod
    def _states_identical(state: ChangeState, previous_state: ChangeState) -> bool:
        return (
            (state.t1_mask == previous_state.t1_mask).all()
            and (state.t2_mask == previous_state.t2_mask).all()
            and (state.change_mask == previous_state.change_mask).all()
        )

    def _verify_initial(
        self,
        state: ChangeState,
        records: tuple[EvidenceRecord, ...],
        previous_score: float | None,
    ) -> tuple[VerifierOutput, StageTrace]:
        judgments, diagnoses = self._inspect_and_diagnose(
            state, records, candidate=False, previous_state=None
        )
        selected = self._select_diagnosis(diagnoses)
        plan = self._plan(state, records, selected)
        decision = self._decision(
            state,
            {
                "mode": "initial",
                "evidence": [item.to_dict() for item in records],
                "diagnoses": [item.__dict__ for item in diagnoses],
                "plan": _plan_dict(plan),
                "previous_score": previous_score,
            },
            initial=True,
            previous_state=None,
        )
        if decision.comparison != "initial":
            raise StageProtocolError("initial decision must use comparison=initial")
        output = self._output(decision, plan, selected, previous_score, initial=True)
        return output, StageTrace(
            mode="initial",
            evidence=records,
            judgments=judgments,
            diagnoses=diagnoses,
            plan=plan,
            decision=decision,
        )

    def _verify_candidate(
        self,
        state: ChangeState,
        previous_state: ChangeState,
        records: tuple[EvidenceRecord, ...],
        previous_score: float | None,
    ) -> tuple[VerifierOutput, StageTrace]:
        judgments, diagnoses = self._inspect_and_diagnose(
            state, records, candidate=True, previous_state=previous_state
        )
        decision = self._decision(
            state,
            {
                "mode": "candidate",
                "previous_change_pixels": int(previous_state.change_mask.sum()),
                "candidate_change_pixels": int(state.change_mask.sum()),
                "evidence": [item.to_dict() for item in records],
                "diagnoses": [item.__dict__ for item in diagnoses],
                "previous_score": previous_score,
            },
            initial=False,
            previous_state=previous_state,
        )
        selected = self._select_diagnosis(diagnoses)
        plan: ActionPlan | None = None
        replan_records: tuple[EvidenceRecord, ...] = ()
        replan_judgments: tuple[EvidenceJudgment, ...] = ()
        replan_diagnoses: tuple[Diagnosis, ...] = ()
        output_diagnosis = selected
        if decision.accept and decision.comparison == "better" and decision.stop:
            if selected is not None:
                raise StageProtocolError(
                    "candidate decision cannot stop while a diagnosed error remains"
                )
            plan = ActionPlan(None, "finish", None)
        elif decision.accept and decision.comparison == "better":
            proposal_config = state.evidence.get("verifier_mask_facts", {}).get(
                "proposal_config", {}
            )
            full_proposals = build_verifier_regions(
                state,
                None,
                max_regions=self.max_regions,
                min_component_area=int(proposal_config.get("min_component_area", 1)),
                padding_ratio=float(proposal_config.get("padding_ratio", 0.25)),
            )
            replan_records = tuple(
                EvidenceRecord.from_proposal(item) for item in full_proposals
            )
            if replan_records:
                replan_judgments, replan_diagnoses = self._inspect_and_diagnose(
                    state,
                    replan_records,
                    candidate=False,
                    previous_state=None,
                )
                remaining = self._select_diagnosis(replan_diagnoses)
                plan = self._plan(state, replan_records, remaining)
                output_diagnosis = remaining
            else:
                plan = ActionPlan(None, "finish", None)
        output = self._output(
            decision, plan, output_diagnosis, previous_score, initial=False
        )
        return output, StageTrace(
            mode="candidate",
            evidence=records,
            judgments=judgments,
            diagnoses=diagnoses,
            plan=plan,
            decision=decision,
            replan_evidence=replan_records,
            replan_judgments=replan_judgments,
            replan_diagnoses=replan_diagnoses,
        )

    def _inspect_and_diagnose(
        self,
        state: ChangeState,
        records: tuple[EvidenceRecord, ...],
        *,
        candidate: bool,
        previous_state: ChangeState | None,
    ) -> tuple[tuple[EvidenceJudgment, ...], tuple[Diagnosis, ...]]:
        judgments: list[EvidenceJudgment] = []
        diagnoses: list[Diagnosis] = []
        evidence_stage = "candidate_evidence" if candidate else "evidence"
        diagnosis_stage = "candidate_diagnosis" if candidate else "diagnosis"
        for record in records[: self.max_regions]:
            judgment = self._run_stage(
                evidence_stage,
                state,
                {"region": record.to_dict(), "schema": "evidence_judgment_v1"},
                lambda response, region_id=record.region_id: _parse_judgment(
                    response, region_id
                ),
                previous_state,
            )
            judgments.append(judgment)

            def parse_diagnosis(
                response: Mapping[str, Any],
                current_record: EvidenceRecord = record,
                current_judgment: EvidenceJudgment = judgment,
            ) -> Diagnosis:
                parsed = _parse_diagnosis(response, current_record.region_id)
                self._validate_diagnosis(
                    current_record, current_judgment, parsed, candidate=candidate
                )
                return parsed

            parsed_diagnosis = self._run_stage(
                diagnosis_stage,
                state,
                {
                    "region": record.to_dict(),
                    "visual_judgment": judgment.__dict__,
                    "schema": "diagnosis_v1",
                },
                parse_diagnosis,
                previous_state,
            )
            diagnoses.append(parsed_diagnosis)
        if len(records) > self.max_regions:
            raise StageProtocolError(
                f"proposal count {len(records)} exceeds staged verifier max_regions={self.max_regions}"
            )
        return tuple(judgments), tuple(diagnoses)

    @staticmethod
    def _validate_diagnosis(
        record: EvidenceRecord,
        judgment: EvidenceJudgment,
        diagnosis: Diagnosis,
        *,
        candidate: bool,
    ) -> None:
        if diagnosis.error_type == "false_negative" and record.change_mask_state != "black":
            raise StageProtocolError("false_negative requires a black/missing change region")
        if (
            diagnosis.error_type == "false_positive_change"
            and record.change_mask_state != "white"
        ):
            raise StageProtocolError("false_positive_change requires a white change region")
        if candidate or judgment.evidence_quality == "insufficient":
            return
        states = {judgment.t1_state, judgment.t2_state}
        clear_appearance = states == {"background", "building"}
        if (
            clear_appearance
            and record.change_mask_state == "white"
            and diagnosis.error_type == "false_positive_change"
        ):
            raise StageProtocolError(
                "a clear T1/T2 building appearance difference already supported by a white "
                "change region cannot be labeled false_positive_change"
            )
        if (
            clear_appearance
            and record.change_mask_state == "black"
            and diagnosis.error_type == "none"
        ):
            raise StageProtocolError(
                "a clear T1/T2 appearance difference missing from the change mask cannot be none"
            )

    def _select_diagnosis(
        self, diagnoses: tuple[Diagnosis, ...]
    ) -> Diagnosis | None:
        actionable = [item for item in diagnoses if item.error_type != "none"]
        if not actionable:
            return None
        return max(actionable, key=lambda item: item.confidence)

    def _plan(
        self,
        state: ChangeState,
        records: tuple[EvidenceRecord, ...],
        diagnosis: Diagnosis | None,
    ) -> ActionPlan:
        if diagnosis is None or diagnosis.error_type == "none":
            return ActionPlan(None, "finish", None)
        record = next(item for item in records if item.region_id == diagnosis.region_id)
        def parse_plan(response: Mapping[str, Any]) -> ActionPlan:
            plan = _parse_plan(response, record.region_id)
            if plan.target_view != diagnosis.target_view:
                raise StageProtocolError("plan target_view must match diagnosis target_view")
            if plan.action not in record.allowed_actions and plan.action != "box":
                raise StageProtocolError(
                    f"{record.region_id}: action {plan.action!r} is not editable in target view"
                )
            if plan.action in {"positive_point", "negative_point"}:
                point = plan.coordinate_normalized_1000
                assert point is not None
                x1, y1, x2, y2 = record.box_normalized_1000
                if not (x1 <= point[0] <= x2 and y1 <= point[1] <= y2):
                    raise StageProtocolError("planned point is outside the Environment proposal")
                seed_white = bool(
                    record.editable_seed_white.get(plan.target_view or "", False)
                )
                if plan.action == "negative_point" and not seed_white:
                    raise StageProtocolError(
                        "negative_point requires a white editable seed"
                    )
                if plan.action == "positive_point" and seed_white:
                    raise StageProtocolError(
                        "positive_point requires a black editable seed"
                    )
            return plan

        return self._run_stage(
            "plan",
            state,
            {
                "region": record.to_dict(),
                "diagnosis": diagnosis.__dict__,
                "schema": "action_plan_v1",
            },
            parse_plan,
        )

    def _decision(
        self,
        state: ChangeState,
        payload: Mapping[str, Any],
        *,
        initial: bool,
        previous_state: ChangeState | None,
    ) -> Decision:
        def parse_decision(response: Mapping[str, Any]) -> Decision:
            decision = _parse_decision(response)
            if initial and decision.comparison != "initial":
                raise StageProtocolError("initial decision must use comparison=initial")
            if not initial and decision.comparison == "initial":
                raise StageProtocolError(
                    "candidate decision cannot use comparison=initial"
                )
            if not initial and decision.accept != (decision.comparison == "better"):
                raise StageProtocolError(
                    "candidate accept must be true exactly when comparison=better"
                )
            return decision

        return self._run_stage(
            "decision", state, payload, parse_decision, previous_state
        )

    def _output(
        self,
        decision: Decision,
        plan: ActionPlan | None,
        diagnosis: Diagnosis | None,
        previous_score: float | None,
        *,
        initial: bool,
    ) -> VerifierOutput:
        if plan is None:
            error_type = diagnosis.error_type if diagnosis else "uncertain_region"
            target_view = diagnosis.target_view if diagnosis and diagnosis.target_view else "t2"
            region = None
            suggested_action = None
        elif plan.action == "finish":
            error_type = "none"
            target_view = "t2"
            region = None
            suggested_action = "finish"
        else:
            error_type = diagnosis.error_type if diagnosis else "uncertain_region"
            target_view = plan.target_view or "t2"
            if plan.coordinate_normalized_1000 is not None:
                x, y = plan.coordinate_normalized_1000
                region = (x, y, x, y)
            else:
                assert plan.box_normalized_1000 is not None
                region = plan.box_normalized_1000
            suggested_action = plan.action
        accept = bool(
            decision.accept
            and (
                decision.quality_score >= self.accept_threshold
                and error_type == "none"
                if initial
                else decision.comparison == "better"
            )
        )
        return VerifierOutput(
            quality_score=decision.quality_score,
            progress_score=decision.progress_score,
            score_delta=(
                0.0
                if previous_score is None
                else decision.quality_score - previous_score
            ),
            comparison=decision.comparison,
            error_type=error_type,
            target_view=target_view,
            error_region=region,
            suggested_action=suggested_action,
            feedback=decision.feedback,
            accept=accept,
            verifier_valid=True,
            localization_valid=(
                plan is not None and (plan.action == "finish" or region is not None)
            ),
            stop=bool(
                accept
                and decision.stop
                and plan is not None
                and plan.action == "finish"
            ),
        )


def _parse_judgment(payload: Mapping[str, Any], region_id: str) -> EvidenceJudgment:
    _exact_keys(payload, {"region_id", "visual_judgment"}, "evidence")
    if payload["region_id"] != region_id or not isinstance(payload["visual_judgment"], Mapping):
        raise StageProtocolError("evidence response has the wrong region_id or shape")
    body = payload["visual_judgment"]
    _exact_keys(body, {"t1_state", "t2_state", "visual_confidence", "evidence_quality"}, "visual_judgment")
    t1, t2 = body["t1_state"], body["t2_state"]
    if t1 not in {"building", "background", "mixed", "uncertain"} or t2 not in {"building", "background", "mixed", "uncertain"}:
        raise StageProtocolError("visual judgment contains an invalid RGB state")
    quality = body["evidence_quality"]
    if quality not in {"clear", "ambiguous", "insufficient"}:
        raise StageProtocolError("visual judgment contains an invalid evidence_quality")
    if isinstance(body["visual_confidence"], bool) or not isinstance(
        body["visual_confidence"], (int, float)
    ):
        raise StageProtocolError("visual_confidence must be numeric")
    confidence = float(body["visual_confidence"])
    if not 0 <= confidence <= 1:
        raise StageProtocolError("visual_confidence must be in [0,1]")
    return EvidenceJudgment(region_id, t1, t2, confidence, quality)


def _parse_diagnosis(payload: Mapping[str, Any], region_id: str) -> Diagnosis:
    _exact_keys(payload, {"region_id", "diagnosis"}, "diagnosis")
    if payload["region_id"] != region_id or not isinstance(payload["diagnosis"], Mapping):
        raise StageProtocolError("diagnosis response has the wrong region_id or shape")
    body = payload["diagnosis"]
    _required_keys(
        body,
        required={"error_type", "target_view"},
        optional={"confidence"},
        name="diagnosis body",
    )
    confidence = body.get("confidence", 0.0)
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise StageProtocolError("diagnosis confidence must be numeric")
    return Diagnosis(
        region_id,
        str(body["error_type"]),
        body["target_view"],
        float(confidence),
    )


def _parse_plan(payload: Mapping[str, Any], region_id: str) -> ActionPlan:
    _exact_keys(payload, {"region_id", "plan"}, "plan")
    if payload["region_id"] != region_id or not isinstance(payload["plan"], Mapping):
        raise StageProtocolError("plan response has the wrong region_id or shape")
    body = payload["plan"]
    _exact_keys(body, {"action", "target_view", "coordinate_normalized_1000", "box_normalized_1000"}, "plan body")
    coordinate = body["coordinate_normalized_1000"]
    box = body["box_normalized_1000"]
    return ActionPlan(
        region_id,
        str(body["action"]),
        body["target_view"],
        _integer_tuple(coordinate, 2, "coordinate_normalized_1000")
        if coordinate is not None
        else None,
        _integer_tuple(box, 4, "box_normalized_1000") if box is not None else None,
    )


def _parse_decision(payload: Mapping[str, Any]) -> Decision:
    _exact_keys(
        payload,
        {"decision"},
        "decision envelope",
    )
    body = payload["decision"]
    if not isinstance(body, Mapping):
        raise StageProtocolError("decision must be an object")
    _exact_keys(body, {"comparison", "quality_score", "progress_score", "accept", "stop", "feedback"}, "decision body")
    if any(not isinstance(body[key], bool) for key in ("accept", "stop")):
        raise StageProtocolError("decision accept/stop must be JSON booleans")
    for key in ("quality_score", "progress_score"):
        if isinstance(body[key], bool) or not isinstance(body[key], (int, float)):
            raise StageProtocolError(f"decision {key} must be numeric")
    return Decision(
        str(body["comparison"]),
        float(body["quality_score"]),
        float(body["progress_score"]),
        bool(body["accept"]),
        bool(body["stop"]),
        str(body.get("feedback", "")),
    )


def _exact_keys(payload: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(payload) != expected:
        raise StageProtocolError(
            f"{name} must contain exactly {sorted(expected)}; got {sorted(payload)}"
        )


def _required_keys(
    payload: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str],
    name: str,
) -> None:
    keys = set(payload)
    missing = required - keys
    unexpected = keys - required - optional
    if missing or unexpected:
        raise StageProtocolError(
            f"{name} has missing keys {sorted(missing)} and unexpected keys "
            f"{sorted(unexpected)}"
        )


def _plan_dict(plan: ActionPlan) -> dict[str, Any]:
    return {
        "region_id": plan.region_id,
        "action": plan.action,
        "target_view": plan.target_view,
        "coordinate_normalized_1000": list(plan.coordinate_normalized_1000)
        if plan.coordinate_normalized_1000 is not None
        else None,
        "box_normalized_1000": list(plan.box_normalized_1000)
        if plan.box_normalized_1000 is not None
        else None,
    }


def _integer_tuple(value: Any, length: int, name: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise StageProtocolError(f"{name} must contain {length} integers")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise StageProtocolError(f"{name} must contain only integers")
    return tuple(value)
