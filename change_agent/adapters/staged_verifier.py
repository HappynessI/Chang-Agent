"""Stage-oriented MLLM verifier.

This adapter deliberately keeps model responses small and typed.  It can be
used with a local Qwen model or a hosted backend implementing ``StageBackend``.
The Environment remains the authority for proposal geometry and mask
editability; the model cannot invent either.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from ..coordinates import normalized_point_to_pixel
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
    """Select proposals, inspect local evidence, and resolve geometry in code."""

    SCHEMA_VERSION = "staged_verifier_som_geometry_v2"

    def __init__(
        self,
        backend: StageBackend,
        *,
        accept_threshold: float = 0.82,
        max_regions: int = 6,
        max_selected_regions: int = 3,
        max_retries: int = 2,
        visual_context: str = "hybrid",
    ):
        if not 0 <= accept_threshold <= 1:
            raise ValueError("accept_threshold must be in [0,1]")
        if max_regions < 1:
            raise ValueError("max_regions must be positive")
        if not 1 <= max_selected_regions <= max_regions:
            raise ValueError("max_selected_regions must be within [1,max_regions]")
        if max_retries < 1:
            raise ValueError("max_retries must be positive")
        if visual_context not in {"proposal", "hybrid"}:
            raise ValueError("visual_context must be proposal or hybrid")
        self.backend = backend
        self.accept_threshold = accept_threshold
        self.max_regions = max_regions
        self.max_selected_regions = max_selected_regions
        self.max_retries = max_retries
        self.visual_context = visual_context
        self.last_evidence: dict[str, Any] = {}
        self._last_valid_output: VerifierOutput | None = None
        self._accepted_records: tuple[EvidenceRecord, ...] = ()
        self._accepted_diagnoses: tuple[Diagnosis, ...] = ()

    def reset(self) -> None:
        self.last_evidence = {}
        self._last_valid_output = None
        self._accepted_records = ()
        self._accepted_diagnoses = ()
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
        """Choose a different already-diagnosed region after rollback."""

        rejected = [rejected_action]
        for item in rejection_history:
            action = item.get("action")
            if not isinstance(action, Mapping):
                continue
            try:
                rejected.append(
                    AgentAction(
                        str(action["target_view"]),
                        str(action["action"]),
                        coordinate=(
                            tuple(int(value) for value in action["coordinate"])
                            if action.get("coordinate") is not None
                            else None
                        ),
                        box=(
                            tuple(int(value) for value in action["box"])
                            if action.get("box") is not None
                            else None
                        ),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
        candidates = sorted(
            (
                diagnosis
                for diagnosis in self._accepted_diagnoses
                if diagnosis.error_type
                in {"false_positive_change", "false_negative"}
            ),
            key=lambda item: item.confidence,
            reverse=True,
        )
        for diagnosis in candidates:
            plan = self._plan(accepted_state, self._accepted_records, diagnosis)
            if plan is None or plan.action == "finish":
                continue
            action = _plan_agent_action(plan, accepted_state.image_size)
            if action in rejected:
                continue
            output = VerifierOutput(
                quality_score=accepted_feedback.quality_score,
                progress_score=0.0,
                score_delta=0.0,
                comparison="uncertain",
                error_type=diagnosis.error_type,
                target_view=plan.target_view or "t2",
                error_region=_plan_region(plan),
                suggested_action=plan.action,
                feedback=(
                    f"Rollback excluded failed geometry; try alternate region "
                    f"{diagnosis.region_id}."
                ),
                accept=False,
                verifier_valid=True,
                localization_valid=True,
                stop=False,
            )
            self.last_evidence = {
                "type": "staged_qwen_verifier",
                "schema_version": self.SCHEMA_VERSION,
                "decision_mode": "programmatic_alternate_region_after_rollback",
                "verifier_valid": True,
                "localization_valid": True,
                "replan": {
                    "selected_region_id": diagnosis.region_id,
                    "rejected_action_count": len(rejected),
                    "rejection_reasons": list(rejection_reasons),
                },
                "validation_errors": [],
                "backend_calls": [],
            }
            return output
        output = VerifierOutput(
            quality_score=accepted_feedback.quality_score,
            progress_score=0.0,
            score_delta=0.0,
            comparison="uncertain",
            error_type=accepted_feedback.error_type,
            target_view=accepted_feedback.target_view,
            error_region=None,
            suggested_action=None,
            feedback="No untried safe diagnosed region remains after rollback.",
            accept=False,
            verifier_valid=True,
            localization_valid=False,
            stop=False,
        )
        self.last_evidence = {
            "type": "staged_qwen_verifier",
            "schema_version": self.SCHEMA_VERSION,
            "decision_mode": "programmatic_no_alternate_region_after_rollback",
            "verifier_valid": True,
            "localization_valid": False,
            "replan": {
                "selected_region_id": None,
                "rejected_action_count": len(rejected),
                "rejection_reasons": list(rejection_reasons),
            },
            "validation_errors": [],
            "backend_calls": [],
        }
        return output

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
                finish_authorized = bool(
                    previous
                    and previous.accept
                    and previous.error_type == "none"
                    and previous.suggested_action == "finish"
                )
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
                    stop=finish_authorized,
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
        selected_records = self._select_global_regions(state, records)
        judgments, diagnoses = self._inspect_and_diagnose(
            state,
            selected_records,
            candidate=False,
            previous_state=None,
            proposal_catalog=records,
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
        self._accepted_records = records
        self._accepted_diagnoses = diagnoses
        return output, StageTrace(
            mode="initial",
            evidence=records,
            selected_region_ids=tuple(item.region_id for item in selected_records),
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
            state,
            records,
            candidate=True,
            previous_state=previous_state,
            proposal_catalog=records,
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
        replan_selected_records: tuple[EvidenceRecord, ...] = ()
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
                replan_selected_records = self._select_global_regions(
                    state, replan_records
                )
                replan_judgments, replan_diagnoses = self._inspect_and_diagnose(
                    state,
                    replan_selected_records,
                    candidate=False,
                    previous_state=None,
                    proposal_catalog=replan_records,
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
            selected_region_ids=tuple(item.region_id for item in records),
            judgments=judgments,
            diagnoses=diagnoses,
            plan=plan,
            decision=decision,
            replan_evidence=replan_records,
            replan_selected_region_ids=tuple(
                item.region_id for item in replan_selected_records
            ),
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
        proposal_catalog: tuple[EvidenceRecord, ...],
    ) -> tuple[tuple[EvidenceJudgment, ...], tuple[Diagnosis, ...]]:
        judgments: list[EvidenceJudgment] = []
        diagnoses: list[Diagnosis] = []
        evidence_stage = "candidate_evidence" if candidate else "evidence"
        diagnosis_stage = "candidate_diagnosis" if candidate else "diagnosis"
        for record in records[: self.max_regions]:
            judgment = self._run_stage(
                evidence_stage,
                state,
                {
                    "region": record.to_dict(),
                    "proposal_catalog": [
                        item.to_dict() for item in proposal_catalog
                    ],
                    "schema": "evidence_judgment_v1",
                    "visual_context": self.visual_context,
                },
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
                    "proposal_catalog": [
                        item.to_dict() for item in proposal_catalog
                    ],
                    "visual_judgment": judgment.__dict__,
                    "schema": "diagnosis_v1",
                    "visual_context": self.visual_context,
                },
                parse_diagnosis,
                previous_state,
            )
            diagnoses.append(parsed_diagnosis)
        return tuple(judgments), tuple(diagnoses)

    def _select_global_regions(
        self,
        state: ChangeState,
        records: tuple[EvidenceRecord, ...],
    ) -> tuple[EvidenceRecord, ...]:
        """Ask the model to reference marked regions, never coordinates."""

        if len(records) > self.max_regions:
            raise StageProtocolError(
                f"proposal count {len(records)} exceeds staged verifier max_regions={self.max_regions}"
            )
        catalog = [item.to_dict() for item in records]
        allowed_ids = {item.region_id for item in records}

        def parse_selection(response: Mapping[str, Any]) -> tuple[str, ...]:
            if set(response) != {"selection"} or not isinstance(
                response["selection"], Mapping
            ):
                raise StageProtocolError("selection response must contain only selection")
            body = response["selection"]
            if set(body) != {"region_ids", "reason"}:
                raise StageProtocolError(
                    "selection must contain exactly region_ids and reason"
                )
            region_ids = body["region_ids"]
            if not isinstance(region_ids, list) or not region_ids:
                raise StageProtocolError("selection region_ids must be a non-empty list")
            if len(region_ids) > self.max_selected_regions:
                raise StageProtocolError(
                    f"selection exceeds max_selected_regions={self.max_selected_regions}"
                )
            if any(not isinstance(item, str) for item in region_ids):
                raise StageProtocolError("selection region_ids must contain strings")
            if len(set(region_ids)) != len(region_ids):
                raise StageProtocolError("selection region_ids must be unique")
            unknown = set(region_ids) - allowed_ids
            if unknown:
                raise StageProtocolError(
                    f"selection contains unknown region ids: {sorted(unknown)}"
                )
            return tuple(region_ids)

        selected_ids = self._run_stage(
            "select",
            state,
            {
                "proposal_catalog": catalog,
                "max_selected_regions": self.max_selected_regions,
                "schema": "region_selection_v1",
            },
            parse_selection,
        )
        by_id = {item.region_id: item for item in records}
        return tuple(by_id[region_id] for region_id in selected_ids)

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
        # White/black proposal polarity is a structural invariant.  It is not a
        # semantic correctness proof: one component can contain both a real
        # temporal change and unsupported boundary/interior pixels.  Qwen must
        # therefore retain authority to emit false_positive_change, mixed_error,
        # or none after inspecting RGB and mask coverage.

    def _select_diagnosis(
        self, diagnoses: tuple[Diagnosis, ...]
    ) -> Diagnosis | None:
        actionable = [item for item in diagnoses if item.error_type != "none"]
        if not actionable:
            return None
        safe_priority = {
            "false_positive_change": 2,
            "false_negative": 2,
            "mixed_error": 1,
            "uncertain_region": 0,
        }
        return max(
            actionable,
            key=lambda item: (safe_priority[item.error_type], item.confidence),
        )

    def _plan(
        self,
        state: ChangeState,
        records: tuple[EvidenceRecord, ...],
        diagnosis: Diagnosis | None,
    ) -> ActionPlan | None:
        if diagnosis is None or diagnosis.error_type == "none":
            return ActionPlan(None, "finish", None)
        record = next(item for item in records if item.region_id == diagnosis.region_id)
        if diagnosis.error_type not in {
            "false_positive_change",
            "false_negative",
        }:
            return None
        target_view = diagnosis.target_view
        if target_view not in {"t1", "t2"}:
            return None
        seed_white = bool(record.editable_seed_white.get(target_view, False))
        action = (
            "negative_point"
            if diagnosis.error_type == "false_positive_change"
            else "positive_point"
        )
        if action == "negative_point" and not seed_white:
            return None
        if action == "positive_point" and seed_white:
            return None
        return ActionPlan(
            record.region_id,
            action,
            target_view,
            coordinate_normalized_1000=record.component_seed_normalized_1000,
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
            if not initial and decision.accept and decision.comparison != "better":
                raise StageProtocolError(
                    "candidate accept=true requires comparison=better"
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


def _plan_dict(plan: ActionPlan | None) -> dict[str, Any] | None:
    if plan is None:
        return None
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


def _plan_region(plan: ActionPlan) -> tuple[int, int, int, int]:
    if plan.coordinate_normalized_1000 is not None:
        x, y = plan.coordinate_normalized_1000
        return x, y, x, y
    if plan.box_normalized_1000 is None:
        raise StageProtocolError("executable plan has no geometry")
    return plan.box_normalized_1000


def _plan_agent_action(
    plan: ActionPlan, image_size: tuple[int, int]
) -> AgentAction:
    if plan.target_view not in {"t1", "t2"}:
        raise StageProtocolError("executable plan has no target view")
    if plan.coordinate_normalized_1000 is not None:
        return AgentAction(
            plan.target_view,
            plan.action,
            coordinate=normalized_point_to_pixel(
                plan.coordinate_normalized_1000, image_size
            ),
        )
    raise StageProtocolError("staged rollback currently supports point plans only")
