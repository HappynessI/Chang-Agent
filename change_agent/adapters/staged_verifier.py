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
    AuditChecklist,
    Decision,
    Diagnosis,
    EvidenceJudgment,
    EvidenceRecord,
    StageBackend,
    StageProtocolError,
    StageTrace,
    TransitionAssessment,
)


class StagedQwenVerifier:
    """Select proposals, inspect local evidence, and resolve geometry in code."""

    SCHEMA_VERSION = "staged_verifier_atomic_grounded_audit_v11"

    def __init__(
        self,
        backend: StageBackend,
        *,
        accept_threshold: float = 0.82,
        max_regions: int = 6,
        max_selected_regions: int = 3,
        max_retries: int = 2,
        visual_context: str = "hybrid",
        min_visual_confidence: float = 0.6,
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
        if not 0 <= min_visual_confidence <= 1:
            raise ValueError("min_visual_confidence must be in [0,1]")
        self.backend = backend
        self.accept_threshold = accept_threshold
        self.max_regions = max_regions
        self.max_selected_regions = max_selected_regions
        self.max_retries = max_retries
        self.visual_context = visual_context
        self.min_visual_confidence = min_visual_confidence
        self.last_evidence: dict[str, Any] = {}
        self._last_valid_output: VerifierOutput | None = None
        self._accepted_records: tuple[EvidenceRecord, ...] = ()
        self._accepted_diagnoses: tuple[Diagnosis, ...] = ()
        self._planned_diagnosis: Diagnosis | None = None
        self._regional_validation_errors: list[dict[str, str]] = []

    def reset(self) -> None:
        self.last_evidence = {}
        self._last_valid_output = None
        self._accepted_records = ()
        self._accepted_diagnoses = ()
        self._planned_diagnosis = None
        self._regional_validation_errors = []
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
        candidates = tuple(
            diagnosis
            for diagnosis in self._accepted_diagnoses
            if diagnosis.error_type
            in {"false_positive_change", "false_negative"}
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
        self._regional_validation_errors = []
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
                    state,
                    previous_state,
                    records,
                    previous_score,
                    previous_action,
                )
            self.last_evidence = {
                "type": "staged_qwen_verifier",
                "schema_version": self.SCHEMA_VERSION,
                "verifier_valid": output.verifier_valid,
                "localization_valid": output.localization_valid,
                "stage_trace": trace.to_dict(),
                "validation_errors": [],
                "regional_validation_errors": list(
                    self._regional_validation_errors
                ),
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
        selected_records, judgments, diagnoses = self._audit_regions_in_batches(
            state, records
        )
        selected, plan = self._select_action_plan(state, records, diagnoses)
        completion_passed, completion_reason = self._state_completion_gate(
            state, records, selected_records, judgments, diagnoses
        )
        if plan is not None and plan.action == "finish" and not completion_passed:
            missing = next(
                (
                    item
                    for item in records
                    if item.region_id
                    not in {selected.region_id for selected in selected_records}
                ),
                records[0],
            )
            selected = Diagnosis(
                missing.region_id,
                "uncertain_region",
                None,
                _uncertain_audit_checklist(),
            )
            plan = None
        decision = self._initial_decision(
            state,
            {
                "mode": "initial",
                "evidence": [item.to_dict() for item in records],
                "diagnoses": [item.to_dict() for item in diagnoses],
                "plan": _plan_dict(plan),
                "previous_score": previous_score,
            },
            previous_state=None,
            runtime_quality=_diagnosis_quality(diagnoses),
        )
        if decision.comparison != "initial":
            raise StageProtocolError("initial decision must use comparison=initial")
        if not completion_passed and plan is None:
            decision = Decision(
                decision.comparison,
                decision.quality_score,
                decision.progress_score,
                decision.accept,
                decision.stop,
                f"{decision.feedback}; completion gate: {completion_reason}",
            )
        output = self._output(decision, plan, selected, previous_score, initial=True)
        self._accepted_records = records
        self._accepted_diagnoses = diagnoses
        self._planned_diagnosis = selected
        return output, StageTrace(
            mode="initial",
            evidence=records,
            selected_region_ids=tuple(item.region_id for item in selected_records),
            judgments=judgments,
            diagnoses=diagnoses,
            plan=plan,
            decision=decision,
            transition_assessment=None,
            state_completion_gate_passed=completion_passed,
            state_completion_gate_reason=completion_reason,
        )

    def _verify_candidate(
        self,
        state: ChangeState,
        previous_state: ChangeState,
        records: tuple[EvidenceRecord, ...],
        previous_score: float | None,
        previous_action: AgentAction | None,
    ) -> tuple[VerifierOutput, StageTrace]:
        self._validate_attempted_action(previous_state, previous_action)
        judgments = self._inspect_candidate_evidence(
            state,
            records,
            previous_state=previous_state,
            proposal_catalog=records,
            previous_action=previous_action,
        )
        transition_assessment = self._assess_candidate_transition(
            records, judgments, previous_action
        )
        decision = self._candidate_decision(transition_assessment, previous_score)
        plan: ActionPlan | None = None
        replan_records: tuple[EvidenceRecord, ...] = ()
        replan_selected_records: tuple[EvidenceRecord, ...] = ()
        replan_judgments: tuple[EvidenceJudgment, ...] = ()
        replan_diagnoses: tuple[Diagnosis, ...] = ()
        completion_passed: bool | None = None
        completion_reason: str | None = None
        output_diagnosis = self._select_diagnosis(self._accepted_diagnoses)
        if decision.accept and decision.comparison == "better":
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
                (
                    replan_selected_records,
                    replan_judgments,
                    replan_diagnoses,
                ) = self._audit_regions_in_batches(
                    state, replan_records
                )
                remaining, plan = self._select_action_plan(
                    state, replan_records, replan_diagnoses
                )
                completion_passed, completion_reason = self._state_completion_gate(
                    state,
                    replan_records,
                    replan_selected_records,
                    replan_judgments,
                    replan_diagnoses,
                )
                if (
                    plan is not None
                    and plan.action == "finish"
                    and not completion_passed
                ):
                    missing = next(
                        (
                            item
                            for item in replan_records
                            if item.region_id
                            not in {
                                selected.region_id
                                for selected in replan_selected_records
                            }
                        ),
                        replan_records[0],
                    )
                    remaining = Diagnosis(
                        missing.region_id,
                        "uncertain_region",
                        None,
                        _uncertain_audit_checklist(),
                    )
                    plan = None
                output_diagnosis = remaining
            else:
                plan = ActionPlan(None, "finish", None)
                completion_passed = True
                completion_reason = "no audit region remains"
            self._accepted_records = replan_records
            self._accepted_diagnoses = replan_diagnoses
            self._planned_diagnosis = output_diagnosis
        output = self._output(
            decision, plan, output_diagnosis, previous_score, initial=False
        )
        return output, StageTrace(
            mode="candidate",
            evidence=records,
            selected_region_ids=tuple(item.region_id for item in records),
            judgments=judgments,
            diagnoses=(),
            plan=plan,
            decision=decision,
            transition_assessment=transition_assessment,
            state_completion_gate_passed=completion_passed,
            state_completion_gate_reason=completion_reason,
            replan_evidence=replan_records,
            replan_selected_region_ids=tuple(
                item.region_id for item in replan_selected_records
            ),
            replan_judgments=replan_judgments,
            replan_diagnoses=replan_diagnoses,
        )

    def _audit_regions_in_batches(
        self,
        state: ChangeState,
        records: tuple[EvidenceRecord, ...],
    ) -> tuple[
        tuple[EvidenceRecord, ...],
        tuple[EvidenceJudgment, ...],
        tuple[Diagnosis, ...],
    ]:
        """Audit every region atomically without a lossy selection hand-off.

        v10 split one semantic judgment across selection, RGB evidence, and
        diagnosis calls, then discarded the selection rationale.  v11 uses one
        grounded call per Environment region so mask polarity, RGB states,
        semantic assessment, checklist evidence, and target view are validated
        together.  The Environment ordering is deterministic and complete.
        """

        screened_records, screening_hypotheses = self._screen_region_hypotheses(
            state, records
        )
        judgments: list[EvidenceJudgment] = []
        diagnoses: list[Diagnosis] = []
        for record in screened_records:
            screening_hypothesis = screening_hypotheses[record.region_id]
            try:
                judgment, diagnosis = self._run_stage(
                    "audit",
                    state,
                    {
                        "target_class": state.query,
                        "region": record.to_dict(),
                        "proposal_catalog": [item.to_dict() for item in records],
                        "schema": "atomic_grounded_region_audit_v11",
                        "visual_context": self.visual_context,
                        "screening_hypothesis": screening_hypothesis,
                    },
                    lambda response, current_record=record, current_hypothesis=screening_hypothesis: _parse_atomic_audit(
                        response, current_record, current_hypothesis
                    ),
                    None,
                )
            except StageProtocolError as error:
                error_text = str(error)
                self._regional_validation_errors.append(
                    {"region_id": record.region_id, "error": error_text}
                )
                judgment = EvidenceJudgment(
                    record.region_id,
                    "uncertain",
                    "uncertain",
                    "insufficient",
                    record.change_mask_state,
                    "uncertain",
                    f"Atomic audit failed closed: {error_text}",
                    screening_hypothesis,
                    "uncertain",
                )
                diagnosis = Diagnosis(
                    record.region_id,
                    "uncertain_region",
                    None,
                    _uncertain_audit_checklist(),
                    f"Global hypothesis retained but local audit failed closed: {error_text}",
                )
            self._validate_diagnosis(record, diagnosis)
            judgments.append(judgment)
            diagnoses.append(diagnosis)
        return screened_records, tuple(judgments), tuple(diagnoses)

    def _screen_region_hypotheses(
        self,
        state: ChangeState,
        records: tuple[EvidenceRecord, ...],
    ) -> tuple[tuple[EvidenceRecord, ...], dict[str, str]]:
        """Persist every global screening rationale into its local atomic audit.

        V10 discarded the useful selection reason before diagnosis. This pass still
        covers every Environment region, but each selected batch carries its exact
        global hypothesis into the subsequent local audit and requires an explicit
        confirmation, refutation, or uncertainty result.
        """

        pending = list(records)
        ordered: list[EvidenceRecord] = []
        hypotheses: dict[str, str] = {}
        by_id = {item.region_id: item for item in records}
        while pending:
            allowed_ids = {item.region_id for item in pending}

            def parse_selection(
                response: Mapping[str, Any],
            ) -> tuple[tuple[str, ...], str]:
                if set(response) != {"selection"} or not isinstance(
                    response["selection"], Mapping
                ):
                    raise StageProtocolError(
                        "screening response must contain only selection"
                    )
                body = response["selection"]
                if set(body) != {"region_ids", "reason"}:
                    raise StageProtocolError(
                        "screening selection must contain exactly region_ids and reason"
                    )
                region_ids = body["region_ids"]
                if not isinstance(region_ids, list) or not region_ids:
                    raise StageProtocolError(
                        "screening region_ids must be a non-empty list"
                    )
                if len(region_ids) > self.max_selected_regions:
                    raise StageProtocolError(
                        "screening exceeds max_selected_regions="
                        f"{self.max_selected_regions}"
                    )
                if any(not isinstance(item, str) for item in region_ids):
                    raise StageProtocolError(
                        "screening region_ids must contain strings"
                    )
                if len(set(region_ids)) != len(region_ids):
                    raise StageProtocolError(
                        "screening region_ids must be unique"
                    )
                unknown = set(region_ids) - allowed_ids
                if unknown:
                    raise StageProtocolError(
                        f"screening contains unknown region ids: {sorted(unknown)}"
                    )
                reason = str(body["reason"]).strip()
                if not reason:
                    raise StageProtocolError(
                        "screening reason must be non-empty"
                    )
                return tuple(region_ids), reason

            selected_ids, reason = self._run_stage(
                "select",
                state,
                {
                    "target_class": state.query,
                    "proposal_catalog": [item.to_dict() for item in pending],
                    "max_selected_regions": min(
                        self.max_selected_regions, len(pending)
                    ),
                    "schema": "persistent_region_screening_v11",
                },
                parse_selection,
            )
            selected_set = set(selected_ids)
            for region_id in selected_ids:
                ordered.append(by_id[region_id])
                hypotheses[region_id] = reason
            pending = [
                item for item in pending if item.region_id not in selected_set
            ]
        return tuple(ordered), hypotheses

    def _inspect_and_diagnose(
        self,
        state: ChangeState,
        records: tuple[EvidenceRecord, ...],
        *,
        proposal_catalog: tuple[EvidenceRecord, ...],
    ) -> tuple[tuple[EvidenceJudgment, ...], tuple[Diagnosis, ...]]:
        judgments: list[EvidenceJudgment] = []
        diagnoses: list[Diagnosis] = []
        for record in records[: self.max_regions]:
            judgment = self._run_stage(
                "evidence",
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
                None,
            )
            judgments.append(judgment)
            if not self._judgment_is_sufficient(judgment):
                diagnoses.append(
                    Diagnosis(
                        record.region_id,
                        "uncertain_region",
                        None,
                        _uncertain_audit_checklist(),
                    )
                )
                continue

            def parse_diagnosis(
                response: Mapping[str, Any],
                current_record: EvidenceRecord = record,
                current_judgment: EvidenceJudgment = judgment,
            ) -> Diagnosis:
                parsed = _parse_diagnosis(response, current_record.region_id)
                self._validate_diagnosis(current_record, parsed)
                return parsed

            parsed_diagnosis = self._run_stage(
                "diagnosis",
                state,
                {
                    "region": record.to_dict(),
                    "proposal_catalog": [
                        item.to_dict() for item in proposal_catalog
                    ],
                    "visual_judgment": judgment.to_dict(),
                    "schema": "diagnosis_v1",
                    "visual_context": self.visual_context,
                },
                parse_diagnosis,
                None,
            )
            diagnoses.append(parsed_diagnosis)
        return tuple(judgments), tuple(diagnoses)

    def _inspect_candidate_evidence(
        self,
        state: ChangeState,
        records: tuple[EvidenceRecord, ...],
        *,
        previous_state: ChangeState,
        proposal_catalog: tuple[EvidenceRecord, ...],
        previous_action: AgentAction | None,
    ) -> tuple[EvidenceJudgment, ...]:
        """Inspect candidate deltas without reclassifying black/white state errors."""

        judgments: list[EvidenceJudgment] = []
        for record in records[: self.max_regions]:
            judgment = self._run_stage(
                "candidate_evidence",
                state,
                {
                    "target_class": state.query,
                    "region": record.to_dict(),
                    "proposal_catalog": [
                        item.to_dict() for item in proposal_catalog
                    ],
                    "schema": "candidate_transition_evidence_v1",
                    "visual_context": self.visual_context,
                    "attempted_action": (
                        previous_action.to_dict() if previous_action else None
                    ),
                    "persisted_initial_diagnosis": (
                        self._planned_diagnosis.to_dict()
                        if self._planned_diagnosis is not None
                        else None
                    ),
                },
                lambda response, region_id=record.region_id: _parse_judgment(
                    response, region_id
                ),
                previous_state,
            )
            judgments.append(judgment)
        return tuple(judgments)

    @staticmethod
    def _validate_attempted_action(
        previous_state: ChangeState,
        previous_action: AgentAction | None,
    ) -> None:
        """Recheck point editability against the accepted pre-action state."""

        if previous_action is None:
            raise StageProtocolError("candidate transition requires an attempted action")
        if previous_action.action not in {"positive_point", "negative_point"}:
            return
        if previous_action.coordinate is None:
            raise StageProtocolError("point candidate requires an attempted coordinate")
        x, y = previous_action.coordinate
        width, height = previous_state.image_size
        if not (0 <= x < width and 0 <= y < height):
            raise StageProtocolError("attempted point is outside the accepted state")
        mask = (
            previous_state.t1_mask
            if previous_action.target_view == "t1"
            else previous_state.t2_mask
        )
        seed_white = bool(mask[y, x])
        if previous_action.action == "negative_point" and not seed_white:
            raise StageProtocolError(
                "negative point requires a white seed in the accepted target mask"
            )
        if previous_action.action == "positive_point" and seed_white:
            raise StageProtocolError(
                "positive point requires a black seed in the accepted target mask"
            )

    def _state_completion_gate(
        self,
        state: ChangeState,
        records: tuple[EvidenceRecord, ...],
        selected_records: tuple[EvidenceRecord, ...],
        judgments: tuple[EvidenceJudgment, ...],
        diagnoses: tuple[Diagnosis, ...],
    ) -> tuple[bool, str]:
        """Authorize finish only after every Environment audit region was inspected."""

        facts = state.evidence.get("verifier_mask_facts", {})
        if int(facts.get("initial_audit_uncovered_pixels", 0)) > 0:
            return False, "environment audit mask has uncovered pixels"
        record_ids = {item.region_id for item in records}
        selected_ids = {item.region_id for item in selected_records}
        if selected_ids != record_ids:
            return False, (
                f"audited {len(selected_ids)} of {len(record_ids)} region(s); "
                "unselected regions are not evidence of correctness"
            )
        judgment_ids = {item.region_id for item in judgments}
        diagnosis_ids = {item.region_id for item in diagnoses}
        if judgment_ids != record_ids or diagnosis_ids != record_ids:
            return False, "not every selected region has a judgment and diagnosis"
        if any(not self._judgment_is_sufficient(item) for item in judgments):
            return False, "at least one region lacks sufficient visual evidence"
        if any(item.error_type != "none" for item in diagnoses):
            return False, "at least one diagnosed error remains"
        return True, "all Environment audit regions are covered and diagnosed none"

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
        diagnosis: Diagnosis,
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

    def _judgment_is_sufficient(self, judgment: EvidenceJudgment) -> bool:
        return bool(
            judgment.evidence_quality == "clear"
            and judgment.t1_state in {"building", "background"}
            and judgment.t2_state in {"building", "background"}
        )

    def _assess_candidate_transition(
        self,
        records: tuple[EvidenceRecord, ...],
        judgments: tuple[EvidenceJudgment, ...],
        previous_action: AgentAction | None,
    ) -> TransitionAssessment:
        """Derive benefit/harm from action direction plus local RGB judgments."""

        expected_kind = {
            "negative_point": "delta_removed",
            "positive_point": "delta_added",
        }.get(previous_action.action if previous_action else None)
        by_id = {item.region_id: item for item in judgments}
        sufficient = bool(records) and expected_kind is not None
        intended_improved = False
        introduced_false_positive = False
        introduced_false_negative = False
        boundary_or_artifact_worsened = False
        evidence_parts: list[str] = []
        for record in records:
            judgment = by_id.get(record.region_id)
            if judgment is None or not self._judgment_is_sufficient(judgment):
                sufficient = False
                evidence_parts.append(
                    f"{record.region_id}:{record.audit_kind}:insufficient_visual_evidence"
                )
                continue
            real_target_change = judgment.t1_state != judgment.t2_state
            evidence_parts.append(
                f"{record.region_id}:{record.audit_kind}:"
                f"{judgment.t1_state}->{judgment.t2_state}:"
                f"evidence_quality={judgment.evidence_quality}"
            )
            if record.audit_kind == "delta_removed":
                if real_target_change:
                    introduced_false_negative = True
                elif expected_kind == "delta_removed":
                    intended_improved = True
                else:
                    boundary_or_artifact_worsened = True
            elif record.audit_kind == "delta_added":
                if not real_target_change:
                    introduced_false_positive = True
                elif expected_kind == "delta_added":
                    intended_improved = True
                else:
                    boundary_or_artifact_worsened = True
            else:
                sufficient = False
                evidence_parts.append(f"{record.region_id}:unsupported_delta_polarity")
            if record.audit_kind != expected_kind:
                sufficient = False
                boundary_or_artifact_worsened = True
                evidence_parts.append(
                    f"{record.region_id}:unexpected_for_{previous_action.action if previous_action else 'none'}"
                )
        if expected_kind is None:
            evidence_parts.append("unsupported_or_missing_attempted_action")
        return TransitionAssessment(
            intended_error_improved=intended_improved,
            introduced_false_positive=introduced_false_positive,
            introduced_false_negative=introduced_false_negative,
            boundary_or_artifact_worsened=boundary_or_artifact_worsened,
            evidence_sufficient=sufficient,
            evidence="; ".join(evidence_parts),
        )

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
            key=lambda item: safe_priority[item.error_type],
        )

    def _select_action_plan(
        self,
        state: ChangeState,
        records: tuple[EvidenceRecord, ...],
        diagnoses: tuple[Diagnosis, ...],
    ) -> tuple[Diagnosis | None, ActionPlan | None]:
        """Prefer the highest-priority diagnosis that maps to a safe action.

        A model diagnosis can be semantically plausible yet name a target view
        whose Environment seed is not editable.  Do not let that single invalid
        target suppress another independently diagnosed, executable region.
        """

        selected = self._select_diagnosis(diagnoses)
        if selected is None:
            return None, ActionPlan(None, "finish", None)
        safe_priority = {
            "false_positive_change": 2,
            "false_negative": 2,
            "mixed_error": 1,
            "uncertain_region": 0,
        }
        records_by_id = {item.region_id: item for item in records}
        ranked = sorted(
            (item for item in diagnoses if item.error_type != "none"),
            key=lambda item: (
                -safe_priority[item.error_type],
                records_by_id[item.region_id].component_area,
            ),
        )
        for diagnosis in ranked:
            plan = self._plan(state, records, diagnosis)
            if plan is not None:
                return diagnosis, plan
        return selected, None

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
        action = (
            "negative_point"
            if diagnosis.error_type == "false_positive_change"
            else "positive_point"
        )
        required_seed_white = action == "negative_point"
        target_view = diagnosis.target_view
        target_is_editable = bool(
            target_view in {"t1", "t2"}
            and bool(record.editable_seed_white.get(target_view, False))
            == required_seed_white
        )
        if not target_is_editable:
            editable_views = [
                view
                for view in ("t1", "t2")
                if bool(record.editable_seed_white.get(view, False))
                == required_seed_white
            ]
            if len(editable_views) != 1:
                return None
            target_view = editable_views[0]
        return ActionPlan(
            record.region_id,
            action,
            target_view,
            coordinate_normalized_1000=record.component_seed_normalized_1000,
        )

    def _initial_decision(
        self,
        state: ChangeState,
        payload: Mapping[str, Any],
        *,
        previous_state: ChangeState | None,
        runtime_quality: float,
    ) -> Decision:
        feedback = self._run_stage(
            "decision",
            state,
            payload,
            _parse_initial_assessment,
            previous_state,
        )
        return Decision("initial", runtime_quality, 0.0, False, False, feedback)

    @staticmethod
    def _candidate_decision(
        assessment: TransitionAssessment,
        previous_score: float | None,
    ) -> Decision:
        comparison = assessment.comparison
        progress = {
            "better": 1.0,
            "worse": -1.0,
            "unchanged": 0.0,
            "uncertain": 0.0,
        }[comparison]
        return Decision(
            comparison,
            float(previous_score) if previous_score is not None else 0.0,
            progress,
            comparison == "better",
            False,
            assessment.evidence,
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
        state_ready = bool(plan is not None and plan.action == "finish")
        accept = bool(
            state_ready and error_type == "none"
            if initial
            else decision.comparison == "better" and decision.accept
        )
        state_feedback = (
            "current accepted state has no diagnosed remaining error"
            if state_ready
            else f"next state action: {suggested_action or 'none'} ({error_type})"
        )
        feedback = (
            f"transition: {decision.feedback}; state: {state_feedback}"
            if not initial
            else decision.feedback
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
            feedback=feedback,
            accept=accept,
            verifier_valid=True,
            localization_valid=(
                plan is not None and (plan.action == "finish" or region is not None)
            ),
            stop=bool(accept and state_ready),
        )


def _normalize_evidence_quality(value: Any) -> str:
    quality = str(value).strip().lower()
    if quality == "high":
        return "clear"
    if quality not in {"clear", "ambiguous", "insufficient"}:
        raise StageProtocolError("visual judgment contains an invalid evidence_quality")
    return quality


def _parse_judgment(payload: Mapping[str, Any], region_id: str) -> EvidenceJudgment:
    _exact_keys(payload, {"region_id", "visual_judgment"}, "evidence")
    if payload["region_id"] != region_id or not isinstance(payload["visual_judgment"], Mapping):
        raise StageProtocolError("evidence response has the wrong region_id or shape")
    body = payload["visual_judgment"]
    keys = set(body)
    current_keys = {"t1_state", "t2_state", "evidence_quality"}
    legacy_keys = current_keys | {"visual_confidence"}
    if keys != current_keys and keys != legacy_keys:
        raise StageProtocolError(
            "visual_judgment must contain t1_state, t2_state, and evidence_quality"
        )
    t1, t2 = body["t1_state"], body["t2_state"]
    if t1 not in {"building", "background", "mixed", "uncertain"} or t2 not in {"building", "background", "mixed", "uncertain"}:
        raise StageProtocolError("visual judgment contains an invalid RGB state")
    quality = _normalize_evidence_quality(body["evidence_quality"])
    return EvidenceJudgment(region_id, t1, t2, quality)


def _parse_atomic_audit(
    payload: Mapping[str, Any], record: EvidenceRecord, screening_hypothesis: str
) -> tuple[EvidenceJudgment, Diagnosis]:
    """Parse one v11 audit without allowing semantic state to disappear between calls."""

    _exact_keys(
        payload,
        {"region_id", "visual_judgment", "diagnosis"},
        "atomic audit",
    )
    if payload["region_id"] != record.region_id:
        raise StageProtocolError("atomic audit response has the wrong region_id")
    visual = payload["visual_judgment"]
    if not isinstance(visual, Mapping):
        raise StageProtocolError("atomic audit visual_judgment must be an object")
    _exact_keys(
        visual,
        {
            "change_mask_state",
            "t1_state",
            "t2_state",
            "mask_assessment",
            "evidence_quality",
            "evidence",
            "screening_resolution",
        },
        "atomic audit visual_judgment",
    )
    mask_state = str(visual["change_mask_state"])
    if mask_state != record.change_mask_state:
        raise StageProtocolError(
            "atomic audit must copy authoritative change_mask_state"
        )
    t1_state = str(visual["t1_state"])
    t2_state = str(visual["t2_state"])
    if t1_state not in {"building", "background", "mixed", "uncertain"} or t2_state not in {
        "building",
        "background",
        "mixed",
        "uncertain",
    }:
        raise StageProtocolError("atomic audit contains an invalid RGB state")
    assessment = str(visual["mask_assessment"])
    assessment_to_error = {
        "correct": "none",
        "false_positive": "false_positive_change",
        "false_negative": "false_negative",
        "mixed": "mixed_error",
        "uncertain": "uncertain_region",
    }
    if assessment not in assessment_to_error:
        raise StageProtocolError("atomic audit contains an invalid mask_assessment")
    quality = _normalize_evidence_quality(visual["evidence_quality"])
    evidence = str(visual["evidence"]).strip()
    if not evidence:
        raise StageProtocolError("atomic audit must provide observable visual evidence")
    screening_resolution = str(visual["screening_resolution"])
    if screening_resolution not in {"confirmed", "refuted", "uncertain"}:
        raise StageProtocolError(
            "atomic audit contains an invalid screening_resolution"
        )

    body = payload["diagnosis"]
    if not isinstance(body, Mapping):
        raise StageProtocolError("atomic audit diagnosis must be an object")
    _exact_keys(
        body,
        {"audit_checklist", "target_view", "summary"},
        "atomic audit diagnosis",
    )
    checklist_payload = body["audit_checklist"]
    if not isinstance(checklist_payload, Mapping):
        raise StageProtocolError("atomic audit checklist must be an object")
    checklist = AuditChecklist.from_mapping(checklist_payload)
    if checklist.error_type != assessment_to_error[assessment]:
        raise StageProtocolError(
            "mask_assessment disagrees with runtime-derived audit checklist"
        )
    if quality != "clear" and assessment != "uncertain":
        raise StageProtocolError(
            "ambiguous or insufficient evidence must use uncertain assessment"
        )
    if quality != "clear" and checklist.evidence_sufficient != "uncertain":
        raise StageProtocolError(
            "ambiguous or insufficient evidence must mark evidence_sufficient uncertain"
        )
    expected_resolution = (
        "refuted"
        if assessment == "correct"
        else "uncertain"
        if assessment == "uncertain"
        else "confirmed"
    )
    if screening_resolution != expected_resolution:
        raise StageProtocolError(
            "screening_resolution must explicitly preserve or refute the global "
            "screening hypothesis consistently with mask_assessment"
        )
    if mask_state == "white" and assessment == "false_negative":
        raise StageProtocolError("a white region cannot be a pure false negative")
    if mask_state == "black" and assessment == "false_positive":
        raise StageProtocolError("a black region cannot be a false positive")
    target_view = body["target_view"]
    summary = str(body["summary"]).strip()
    if not summary:
        raise StageProtocolError("atomic audit diagnosis summary must be non-empty")
    if checklist.error_type in {"none", "uncertain_region"} and target_view is not None:
        raise StageProtocolError(
            "correct or uncertain atomic audit must not select a target view"
        )
    judgment = EvidenceJudgment(
        record.region_id,
        t1_state,
        t2_state,
        quality,
        mask_state,
        assessment,
        evidence,
        screening_hypothesis,
        screening_resolution,
    )
    diagnosis = Diagnosis(
        record.region_id,
        checklist.error_type,
        target_view,
        checklist,
        summary,
    )
    return judgment, diagnosis


def _parse_diagnosis(payload: Mapping[str, Any], region_id: str) -> Diagnosis:
    _exact_keys(payload, {"region_id", "diagnosis"}, "diagnosis")
    if payload["region_id"] != region_id or not isinstance(payload["diagnosis"], Mapping):
        raise StageProtocolError("diagnosis response has the wrong region_id or shape")
    body = payload["diagnosis"]
    if set(body) == {"audit_checklist", "target_view"}:
        checklist_payload = body["audit_checklist"]
        if not isinstance(checklist_payload, Mapping):
            raise StageProtocolError("diagnosis audit_checklist must be an object")
        checklist = AuditChecklist.from_mapping(checklist_payload)
        error_type = checklist.error_type
    else:
        # Backward-compatible parser for archived/local scripted backends.  The
        # current prompt never requests model-authored error_type/confidence.
        _required_keys(
            body,
            required={"error_type", "target_view"},
            optional={"confidence"},
            name="diagnosis body",
        )
        error_type = str(body["error_type"])
        checklist = _legacy_audit_checklist(error_type)
    target_view = body["target_view"]
    if error_type in {"none", "uncertain_region"} and target_view is not None:
        raise StageProtocolError(
            "none/uncertain diagnosis must not select a target view"
        )
    return Diagnosis(region_id, error_type, target_view, checklist)


def _uncertain_audit_checklist() -> AuditChecklist:
    return AuditChecklist(
        evidence_sufficient="uncertain",
        target_class_only="uncertain",
        white_pixels_supported="uncertain",
        boundary_alignment="uncertain",
        internal_holes_absent="uncertain",
        changed_object_extent_complete="uncertain",
        fragment_artifacts_absent="uncertain",
    )


def _legacy_audit_checklist(error_type: str) -> AuditChecklist:
    statuses = {
        "evidence_sufficient": "pass",
        "target_class_only": "pass",
        "white_pixels_supported": "pass",
        "boundary_alignment": "pass",
        "internal_holes_absent": "pass",
        "changed_object_extent_complete": "pass",
        "fragment_artifacts_absent": "pass",
    }
    if error_type == "false_positive_change":
        statuses["white_pixels_supported"] = "fail"
    elif error_type == "false_negative":
        statuses["changed_object_extent_complete"] = "fail"
    elif error_type == "mixed_error":
        statuses["white_pixels_supported"] = "fail"
        statuses["changed_object_extent_complete"] = "fail"
    elif error_type == "uncertain_region":
        return _uncertain_audit_checklist()
    elif error_type != "none":
        raise StageProtocolError(f"unsupported legacy error_type: {error_type!r}")
    return AuditChecklist(**statuses)


def _parse_initial_assessment(payload: Mapping[str, Any]) -> str:
    _exact_keys(
        payload,
        {"decision"},
        "decision envelope",
    )
    body = payload["decision"]
    if not isinstance(body, Mapping):
        raise StageProtocolError("decision must be an object")
    if set(body) not in ({"feedback"}, {"quality_score", "feedback"}):
        raise StageProtocolError(
            "initial assessment must contain feedback and no model-authored decisions"
        )
    return str(body["feedback"])


def _diagnosis_quality(diagnoses: tuple[Diagnosis, ...]) -> float:
    scores = [
        item.audit_checklist.quality_score
        for item in diagnoses
        if item.audit_checklist is not None
    ]
    return sum(scores) / len(scores) if scores else 0.0


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
