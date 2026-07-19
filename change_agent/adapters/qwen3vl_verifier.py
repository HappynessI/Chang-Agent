"""Compact diagnosis with RGB temporal-state candidate verification."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from .omniovcd_adapter import connected_components
from ..coordinates import pixel_box_to_normalized, pixel_point_to_normalized
from ..state import AgentAction, ChangeState, VerifierOutput
from ..verifier_regions import attach_verifier_regions


@dataclass(frozen=True)
class _RegionJudgment:
    region_id: str
    verdict: str
    target_view: str | None
    feedback: str
    suggested_action: str | None = None


@dataclass(frozen=True)
class _RegionalAnalysis:
    judgments: tuple[_RegionJudgment, ...]
    error_type: str
    target_view: str
    error_region: tuple[int, int, int, int] | None
    feedback: str
    suggested_action: str | None = None


@dataclass(frozen=True)
class _PairwiseDecision:
    comparison: str
    feedback: str


@dataclass(frozen=True)
class _EffectJudgment:
    region_id: str
    effect: str


@dataclass(frozen=True)
class _TemporalStateJudgment:
    region_id: str
    t1_state: str
    t2_state: str


class Qwen3VLZeroShotVerifier:
    """Classify initial regions and candidate delta effects with compact outputs.

    Qwen never predicts an absolute score or a pairwise comparison. Environment-owned
    boxes make every decision local and auditable; candidate ``better/worse`` is
    derived from elementary T1/T2 RGB states and delta polarity by deterministic
    runtime rules. Mask-context effect labels are retained only as audit evidence.
    """

    SCHEMA_VERSION = "full_batched_guarded_rgb_temporal_effect_v7"
    CANDIDATE_EVIDENCE_MODES = ("mask_context", "rgb_temporal_state")
    ERROR_TYPES = {
        "none",
        "false_positive_change",
        "false_negative",
        "mixed_error",
        "uncertain_region",
    }
    COMPARISONS = {"better", "worse", "unchanged", "uncertain"}
    EFFECT_LABELS = {
        "added_true_change",
        "added_false_change",
        "removed_false_positive",
        "removed_true_change",
        "mixed",
        "uncertain",
    }
    TEMPORAL_STATES = {"building", "background", "mixed", "uncertain"}

    def __init__(
        self,
        *,
        model: Any,
        processor: Any,
        max_new_tokens: int = 1024,
        accept_threshold: float = 0.82,
        max_retries: int = 2,
        max_delta_component_ratio_without_consensus: float = 0.05,
        **legacy_localization_options: Any,
    ):
        if not 0 <= accept_threshold <= 1:
            raise ValueError("accept_threshold must be in [0, 1]")
        if max_retries < 1:
            raise ValueError("max_retries must be positive")
        if not 0 <= max_delta_component_ratio_without_consensus <= 1:
            raise ValueError(
                "max_delta_component_ratio_without_consensus must be in [0, 1]"
            )
        self.model = model
        self.processor = processor
        self.max_new_tokens = max_new_tokens
        # Retained only for CLI/config compatibility. Pairwise mode has no score
        # threshold and records this explicitly in evidence.
        self.accept_threshold = accept_threshold
        self.max_retries = max_retries
        self.max_delta_component_ratio_without_consensus = (
            max_delta_component_ratio_without_consensus
        )
        self.legacy_localization_options = dict(legacy_localization_options)
        self.last_evidence: dict[str, Any] = {}
        self._last_valid_output: VerifierOutput | None = None
        self._candidate_cache: dict[
            str, tuple[VerifierOutput, dict[str, Any]]
        ] = {}

    def reset(self) -> None:
        self._last_valid_output = None
        self.last_evidence = {}
        self._candidate_cache = {}

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
        del previous_score
        if previous_state is not None:
            return self._verify_candidate(state, previous_action, previous_state)

        proposals = state.evidence.get("verifier_region_proposals")
        if not isinstance(proposals, list):
            # Standalone verifier calls (unit tests/smokes) still get the exact same
            # deterministic proposal builder. Normal runtime attaches them in Env.
            proposals = attach_verifier_regions(state, previous_state)
        mask_facts = dict(state.evidence.get("verifier_mask_facts", {}))

        if not proposals:
            return self._invalid_output(
                ["no mask-derived proposal is available for a reliable diagnosis"],
                region_attempts=[],
                pairwise_attempts=[],
                previous_action=previous_action,
            )

        region_errors: list[str] = []
        region_attempts: list[dict[str, Any]] = []
        temporal_judgments: tuple[_TemporalStateJudgment, ...] | None = None
        derivation_detail: list[dict[str, Any]] = []
        regional_analysis: _RegionalAnalysis | None = None
        proposal_config = mask_facts.get("proposal_config", {})
        batch_size = int(
            proposal_config.get("max_regions_per_batch") or len(proposals)
        )
        all_temporal: list[_TemporalStateJudgment] = []
        batches_valid = batch_size > 0
        if not batches_valid:
            region_errors.append("initial audit batch size must be positive")
        for batch_index, start in enumerate(
            range(0, len(proposals), max(batch_size, 1))
        ):
            batch = proposals[start : start + batch_size]
            batch_errors: list[str] = []
            batch_temporal: tuple[_TemporalStateJudgment, ...] | None = None
            for _ in range(self.max_retries):
                raw = self._generate_messages(
                    self.build_messages(
                        state,
                        None,
                        previous_action,
                        previous_state,
                        batch_errors[-1:],
                        proposals_override=batch,
                    )
                )
                try:
                    payload = self._extract_json_object(raw)
                    batch_temporal = self._parse_temporal_state_payload(payload, batch)
                    region_attempts.append(
                        {
                            "batch_index": batch_index,
                            "region_ids": [item["region_id"] for item in batch],
                            "raw": raw,
                            "output": payload,
                        }
                    )
                    break
                except (TypeError, ValueError, KeyError) as error:
                    message = f"initial batch {batch_index}: {error}"
                    batch_errors.append(str(error))
                    region_errors.append(message)
                    region_attempts.append(
                        {
                            "batch_index": batch_index,
                            "region_ids": [item["region_id"] for item in batch],
                            "raw": raw,
                            "error": str(error),
                        }
                    )
            if batch_temporal is None:
                batches_valid = False
                break
            all_temporal.extend(batch_temporal)

        if batches_valid:
            temporal_judgments = tuple(all_temporal)
            judgments, derivation_detail = self._initial_judgments_from_temporal_states(
                temporal_judgments, proposals, state
            )
            regional_analysis = self._derive_regional_analysis(
                judgments, proposals, mask_facts, previous_action
            )

        if regional_analysis is None:
            return self._invalid_output(
                region_errors,
                region_attempts=region_attempts,
                pairwise_attempts=[],
                previous_action=previous_action,
            )

        decision = _PairwiseDecision(
            "initial", "Initial state; no candidate effect ranking was requested."
        )
        output = self._derive_output(regional_analysis, decision)
        self._last_valid_output = output
        self.last_evidence = {
            "type": "qwen3vl_compact_region_zero_shot",
            "decision_mode": "rgb_temporal_state_then_programmatic_initial",
            "gt_available": False,
            "verifier_valid": True,
            "localization_valid": output.localization_valid,
            "mask_facts": mask_facts,
            "region_proposals": proposals,
            "region_judgments": [
                {
                    "region_id": item.region_id,
                    "verdict": item.verdict,
                    "target_view": item.target_view,
                    "feedback": item.feedback,
                }
                for item in regional_analysis.judgments
            ],
            "temporal_state_judgments": [
                {
                    "region_id": item.region_id,
                    "t1_state": item.t1_state,
                    "t2_state": item.t2_state,
                }
                for item in temporal_judgments or ()
            ],
            "initial_state_derivation": derivation_detail,
            "region_attempts": region_attempts,
            "effect_attempts": [],
            "pairwise_attempts": [],
            "comparison": decision.comparison,
            "validation_errors": region_errors,
        }
        return output

    def _verify_candidate(
        self,
        state: ChangeState,
        previous_action: AgentAction | None,
        previous_state: ChangeState,
    ) -> VerifierOutput:
        proposals = list(state.evidence.get("verifier_region_proposals", []))
        facts = dict(state.evidence.get("verifier_mask_facts", {}))
        fingerprint = self._candidate_fingerprint(
            state, previous_state, previous_action, proposals, facts
        )
        cached = self._candidate_cache.get(fingerprint)
        if cached is not None:
            output, evidence = cached
            self.last_evidence = copy.deepcopy(evidence)
            self.last_evidence["cache_hit"] = True
            self.last_evidence["reused_from_step"] = evidence.get("decision_step")
            self._last_valid_output = output if output.verifier_valid else None
            return output

        if self._states_identical(state, previous_state):
            previous = self._last_valid_output
            output = VerifierOutput(
                quality_score=None,
                progress_score=None,
                score_delta=0.0,
                comparison="unchanged",
                error_type=previous.error_type if previous else "uncertain_region",
                target_view=(
                    previous.target_view
                    if previous
                    else previous_action.target_view if previous_action else "t2"
                ),
                error_region=previous.error_region if previous else None,
                suggested_action=previous.suggested_action if previous else None,
                feedback="Candidate masks are identical to the accepted state.",
                accept=bool(previous and previous.accept),
                verifier_valid=True,
                localization_valid=bool(previous and previous.localization_valid),
                stop=bool(previous and previous.stop),
            )
            self.last_evidence = {
                "type": "qwen3vl_compact_delta_effect_zero_shot",
                "decision_mode": "programmatic_identical_state",
                "candidate_fingerprint": fingerprint,
                "decision_key": fingerprint,
                "decision_step": state.step_index,
                "cache_hit": False,
                "comparison": "unchanged",
                "effect_attempts": [],
                "validation_errors": [],
                "gt_available": False,
                "verifier_valid": True,
                "localization_valid": output.localization_valid,
            }
            self._cache_candidate(fingerprint, output)
            return output

        effect_errors: list[str] = []
        effect_attempts: list[dict[str, Any]] = []
        temporal_state_attempts: list[dict[str, Any]] = []
        fusion_detail: list[dict[str, Any]] = []
        temporal_judgments: tuple[_TemporalStateJudgment, ...] | None = None
        judgments: tuple[_EffectJudgment, ...] | None = None
        uncovered = int(facts.get("candidate_delta_uncovered_pixels", 0))
        if uncovered:
            effect_errors.append(
                f"candidate delta has {uncovered} pixels outside the compact proposal set"
            )
        elif not proposals:
            effect_errors.append("candidate delta has no auditable proposal")
        else:
            proposal_config = facts.get("proposal_config", {})
            batch_size = int(
                proposal_config.get("max_regions_per_batch") or len(proposals)
            )
            if batch_size < 1:
                effect_errors.append("candidate delta batch size must be positive")
            else:
                all_effects: list[_EffectJudgment] = []
                all_temporal: list[_TemporalStateJudgment] = []
                decisive_batches_valid = True
                for batch_index, start in enumerate(
                    range(0, len(proposals), batch_size)
                ):
                    batch = proposals[start : start + batch_size]
                    region_ids = [item["region_id"] for item in batch]
                    batch_facts = dict(facts)
                    batch_facts.update(
                        {
                            "batch_index": batch_index,
                            "batch_proposal_count": len(batch),
                            "total_proposal_count": len(proposals),
                            "proposal_count": len(batch),
                        }
                    )
                    mask_judgments: tuple[_EffectJudgment, ...] | None = None
                    mask_errors: list[str] = []
                    for _ in range(self.max_retries):
                        raw = self._generate_messages(
                            self.build_effect_messages(
                                state,
                                previous_state,
                                previous_action,
                                batch,
                                batch_facts,
                                mask_errors[-1:],
                            )
                        )
                        try:
                            payload = self._extract_json_object(raw)
                            mask_judgments = self._parse_effect_payload(payload, batch)
                            effect_attempts.append(
                                {
                                    "evidence_mode": "mask_context",
                                    "batch_index": batch_index,
                                    "region_ids": region_ids,
                                    "raw": raw,
                                    "output": payload,
                                }
                            )
                            break
                        except (TypeError, ValueError, KeyError) as error:
                            mask_errors.append(str(error))
                            effect_errors.append(
                                f"mask_context batch {batch_index}: {error}"
                            )
                            effect_attempts.append(
                                {
                                    "evidence_mode": "mask_context",
                                    "batch_index": batch_index,
                                    "region_ids": region_ids,
                                    "raw": raw,
                                    "error": str(error),
                                }
                            )

                    # Mask-context is advisory. Every batch independently reaches the
                    # decisive RGB pass even if the advisory response is malformed.
                    batch_temporal: tuple[_TemporalStateJudgment, ...] | None = None
                    temporal_errors: list[str] = []
                    for _ in range(self.max_retries):
                        raw = self._generate_messages(
                            self.build_temporal_state_messages(
                                state,
                                previous_state,
                                previous_action,
                                batch,
                                batch_facts,
                                temporal_errors[-1:],
                            )
                        )
                        try:
                            payload = self._extract_json_object(raw)
                            batch_temporal = self._parse_temporal_state_payload(
                                payload, batch
                            )
                            temporal_state_attempts.append(
                                {
                                    "evidence_mode": "rgb_temporal_state",
                                    "batch_index": batch_index,
                                    "region_ids": region_ids,
                                    "raw": raw,
                                    "output": payload,
                                }
                            )
                            break
                        except (TypeError, ValueError, KeyError) as error:
                            temporal_errors.append(str(error))
                            effect_errors.append(
                                f"rgb_temporal_state batch {batch_index}: {error}"
                            )
                            temporal_state_attempts.append(
                                {
                                    "evidence_mode": "rgb_temporal_state",
                                    "batch_index": batch_index,
                                    "region_ids": region_ids,
                                    "raw": raw,
                                    "error": str(error),
                                }
                            )
                    if batch_temporal is None:
                        decisive_batches_valid = False
                        break
                    batch_effects, batch_fusion = self._effects_from_temporal_states(
                        batch_temporal, mask_judgments, batch
                    )
                    previous_change_pixels = max(
                        int(previous_state.change_mask.sum()), 1
                    )
                    total_delta_ratio = (
                        int(facts.get("candidate_delta_pixels", 0))
                        / previous_change_pixels
                    )
                    guarded_effects: list[_EffectJudgment] = []
                    proposal_by_id = {item["region_id"]: item for item in batch}
                    for effect, item in zip(batch_effects, batch_fusion):
                        component_ratio = (
                            int(proposal_by_id[effect.region_id]["component_area"])
                            / previous_change_pixels
                        )
                        consensus_required = (
                            max(component_ratio, total_delta_ratio)
                            > self.max_delta_component_ratio_without_consensus
                        )
                        final_effect = effect.effect
                        if (
                            consensus_required
                            and item["mask_context_agreement"] is not True
                        ):
                            final_effect = "uncertain"
                        item.update(
                            {
                                "component_to_previous_change_ratio": component_ratio,
                                "total_delta_to_previous_change_ratio": total_delta_ratio,
                                "consensus_required": consensus_required,
                                "final_effect": final_effect,
                                "decision_source": (
                                    "rgb_temporal_state_with_large_delta_consensus"
                                    if consensus_required
                                    else "rgb_temporal_state"
                                ),
                            }
                        )
                        guarded_effects.append(
                            _EffectJudgment(effect.region_id, final_effect)
                        )
                    batch_effects = tuple(guarded_effects)
                    all_temporal.extend(batch_temporal)
                    all_effects.extend(batch_effects)
                    fusion_detail.extend(
                        {**item, "batch_index": batch_index}
                        for item in batch_fusion
                    )
                if decisive_batches_valid:
                    temporal_judgments = tuple(all_temporal)
                    judgments = tuple(all_effects)

        if judgments is None:
            output = self._invalid_output(
                effect_errors,
                region_attempts=[],
                pairwise_attempts=[],
                previous_action=previous_action,
            )
            self.last_evidence.update(
                {
                    "type": "qwen3vl_compact_delta_effect_zero_shot",
                    "decision_mode": "batched_rgb_temporal_state_then_programmatic_delta_effect",
                    "candidate_fingerprint": fingerprint,
                    "decision_key": fingerprint,
                    "decision_step": state.step_index,
                    "cache_hit": False,
                    "effect_attempts": effect_attempts,
                    "mask_facts": facts,
                    "region_proposals": proposals,
                    "temporal_state_attempts": temporal_state_attempts,
                    "effect_fusion": fusion_detail,
                }
            )
            self._cache_candidate(fingerprint, output)
            return output

        comparison = self._comparison_from_effects(judgments)
        output = self._derive_effect_output(
            judgments, proposals, previous_action, comparison
        )
        self._last_valid_output = output
        self.last_evidence = {
            "type": "qwen3vl_compact_delta_effect_zero_shot",
            "decision_mode": "batched_rgb_temporal_state_then_programmatic_delta_effect",
            "candidate_fingerprint": fingerprint,
            "decision_key": fingerprint,
            "decision_step": state.step_index,
            "cache_hit": False,
            "gt_available": False,
            "verifier_valid": True,
            "localization_valid": output.localization_valid,
            "mask_facts": facts,
            "region_proposals": proposals,
            "effect_judgments": [
                {"region_id": item.region_id, "effect": item.effect}
                for item in judgments
            ],
            "effect_attempts": effect_attempts,
            "temporal_state_attempts": temporal_state_attempts,
            "temporal_state_judgments": [
                {
                    "region_id": item.region_id,
                    "t1_state": item.t1_state,
                    "t2_state": item.t2_state,
                }
                for item in temporal_judgments or ()
            ],
            "effect_fusion": fusion_detail,
            "pairwise_attempts": [],
            "comparison": comparison,
            "validation_errors": effect_errors,
        }
        self._cache_candidate(fingerprint, output)
        return output

    def _cache_candidate(
        self, fingerprint: str, output: VerifierOutput
    ) -> None:
        self._candidate_cache[fingerprint] = (
            output,
            copy.deepcopy(self.last_evidence),
        )

    def _invalid_output(
        self,
        errors: list[str],
        *,
        region_attempts: list[dict[str, Any]],
        pairwise_attempts: list[dict[str, Any]],
        previous_action: AgentAction | None,
    ) -> VerifierOutput:
        previous = self._last_valid_output
        retained = previous.feedback if previous is not None else None
        messages = ["Verifier invalid; no action is authorized; recheck required."]
        if retained:
            messages.append(f"Previous valid feedback retained: {retained}")
        self.last_evidence = {
            "type": "qwen3vl_compact_region_zero_shot",
            "decision_mode": "rgb_temporal_state_programmatic_verifier",
            "region_attempts": region_attempts,
            "pairwise_attempts": pairwise_attempts,
            "validation_errors": errors,
            "fallback": True,
            "verifier_valid": False,
            "localization_valid": False,
            "gt_available": False,
        }
        return VerifierOutput(
            quality_score=None,
            progress_score=None,
            score_delta=0.0,
            comparison="uncertain",
            error_type=(previous.error_type if previous else "uncertain_region"),
            target_view=(
                previous.target_view
                if previous
                else previous_action.target_view if previous_action else "t2"
            ),
            error_region=previous.error_region if previous else None,
            suggested_action=None,
            feedback=" ".join(messages),
            accept=False,
            verifier_valid=False,
            localization_valid=False,
            stop=False,
        )

    @staticmethod
    def _derive_output(
        analysis: _RegionalAnalysis, decision: _PairwiseDecision
    ) -> VerifierOutput:
        if analysis.suggested_action is not None:
            suggested_action = analysis.suggested_action
            accept = analysis.error_type == "none"
        elif analysis.error_type == "none":
            suggested_action = "finish"
            accept = True
        elif analysis.error_type == "false_positive_change":
            suggested_action = "negative_point"
            accept = False
        elif analysis.error_type == "false_negative":
            suggested_action = "positive_point"
            accept = False
        else:
            suggested_action = "box"
            accept = False
        feedback = analysis.feedback
        if decision.comparison != "initial":
            feedback = f"Pairwise {decision.comparison}: {decision.feedback} {feedback}"
        return VerifierOutput(
            quality_score=None,
            progress_score=None,
            score_delta=0.0,
            comparison=decision.comparison,
            error_type=analysis.error_type,
            target_view=analysis.target_view,
            error_region=analysis.error_region,
            suggested_action=suggested_action,
            feedback=feedback,
            accept=accept,
            verifier_valid=True,
            localization_valid=(analysis.error_type == "none" or analysis.error_region is not None),
            stop=accept,
        )

    def build_messages(
        self,
        state: ChangeState,
        previous_score: float | None,
        previous_action: AgentAction | None,
        previous_state: ChangeState | None = None,
        previous_errors: list[str] | None = None,
        *,
        proposals_override: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        del previous_score, previous_action, previous_state
        proposals = (
            list(proposals_override)
            if proposals_override is not None
            else list(state.evidence.get("verifier_region_proposals", []))
        )
        correction = (
            f"Your previous temporal-state response was invalid: "
            f"{previous_errors[-1]}. Correct it.\n"
            if previous_errors
            else ""
        )
        prompt = (
            "Judge elementary RGB facts at the exact highlighted audit pixels. Predicted masks, "
            "change-mask status, FP/FN labels, and action semantics are intentionally hidden. "
            "Each panel contains T1 RGB and T2 RGB with the exact component surrounded by a "
            "yellow outline, the exact binary audit component, and amplified absolute RGB "
            "difference. Inspect the original RGB inside the yellow outline. For every "
            "region_id, independently classify "
            "the object state at those exact pixels in T1 and T2 as building, background, mixed, "
            "or uncertain. Use mixed when the highlighted component spans both building and "
            "background. Return exactly one compact JSON object mapping every exact region_id to "
            "[t1_state,t2_state], for example "
            "{\"r0\":[\"building\",\"building\"],"
            "\"r1\":[\"background\",\"building\"]}. Do not output FP/FN, true-change, "
            "target views, feedback, scores, actions, coordinates, or extra keys.\n"
            f"{correction}"
        )
        content: list[dict[str, Any]] = [
            {"type": "text", "text": "Fixed clean T1 original image:"},
            {"type": "image", "image": self._as_image(state.t1_image)},
            {"type": "text", "text": "Fixed clean T2 original image:"},
            {"type": "image", "image": self._as_image(state.t2_image)},
        ]
        for proposal in proposals:
            public_proposal = {
                key: proposal[key]
                for key in (
                    "region_id",
                    "component_area",
                    "box_normalized",
                )
                if key in proposal
            }
            content.extend(
                [
                    {
                        "type": "text",
                        "text": (
                            f"Initial RGB audit proposal {proposal['region_id']}: "
                            f"{json.dumps(public_proposal, ensure_ascii=False)}"
                        ),
                    },
                    {"type": "image", "image": self._rgb_initial_panel(state, proposal)},
                ]
            )
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    def build_effect_messages(
        self,
        state: ChangeState,
        previous_state: ChangeState,
        previous_action: AgentAction | None,
        proposals: list[dict[str, Any]],
        facts: dict[str, Any],
        previous_errors: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        correction = (
            f"Your previous effect labels were invalid: {previous_errors[-1]}. Correct them.\n"
            if previous_errors
            else ""
        )
        prompt = (
            "Use the previous/candidate masks and mask-context panels to judge only the pixels "
            "changed by the candidate action. Each supplied delta proposal "
            "has effect_kind added or removed and exact pixel counts. Confirm temporal building "
            "change from the T1/T2 RGB crops. Predicted masks are supporting predictions, not GT. "
            "For an added proposal output added_true_change when the newly white pixels are real "
            "temporal building change, added_false_change when they are false change, mixed when "
            "the component contains both, or uncertain. For a removed proposal output "
            "removed_false_positive when the removed pixels were false change, removed_true_change "
            "when real change was wrongly removed, mixed when both occur, or uncertain. Return "
            "exactly one compact JSON object mapping every exact "
            "region_id to one label string, for example "
            "{\"d0\":\"added_true_change\",\"d1\":\"removed_false_positive\"}. "
            "Do not output feedback sentences, comparison, scores, actions, coordinates, or extra keys.\n"
            f"Exact candidate delta facts: {json.dumps(facts, ensure_ascii=False)}\n"
            f"Action: {json.dumps(self._public_action(previous_action, state.image_size), ensure_ascii=False)}\n"
            f"{correction}"
        )
        content: list[dict[str, Any]] = [
            {"type": "text", "text": "Fixed T1 original image:"},
            {"type": "image", "image": self._as_image(state.t1_image)},
            {"type": "text", "text": "Fixed T2 original image:"},
            {"type": "image", "image": self._as_image(state.t2_image)},
            {"type": "text", "text": "Previous accepted final change mask:"},
            {"type": "image", "image": self._mask_image(previous_state.change_mask)},
            {"type": "text", "text": "Candidate final change mask:"},
            {"type": "image", "image": self._mask_image(state.change_mask)},
        ]
        for proposal in proposals:
            content.extend(
                [
                    {
                        "type": "text",
                        "text": (
                            f"Candidate delta proposal {proposal['region_id']} with exact metadata: "
                            f"{json.dumps(proposal, ensure_ascii=False)}"
                        ),
                    },
                    {
                        "type": "image",
                        "image": self._region_panel(state, proposal, previous_state),
                    },
                ]
            )
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    def build_temporal_state_messages(
        self,
        state: ChangeState,
        previous_state: ChangeState,
        previous_action: AgentAction | None,
        proposals: list[dict[str, Any]],
        facts: dict[str, Any],
        previous_errors: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        del previous_action
        correction = (
            f"Your previous temporal-state output was invalid: {previous_errors[-1]}. Correct it.\n"
            if previous_errors
            else ""
        )
        prompt_facts = {
            key: facts[key]
            for key in (
                "height",
                "width",
                "candidate_delta_pixels",
                "candidate_delta_covered_pixels",
                "candidate_delta_uncovered_pixels",
                "candidate_delta_coverage_ratio",
                "proposal_count",
                "batch_index",
                "batch_proposal_count",
                "total_proposal_count",
                "proposal_config",
            )
            if key in facts
        }
        prompt = (
            "Judge elementary RGB facts at the exact candidate delta pixels. Predicted T1/T2 "
            "object masks and their statistics are intentionally hidden. In each panel: top-left "
            "is the T1 crop with a yellow outline around the exact delta, top-right is the T2 "
            "crop with the same outline, bottom-left is the exact binary delta location, and "
            "bottom-right is amplified absolute RGB difference. Inspect the original RGB inside "
            "the yellow outline. For every "
            "region_id, classify the object state at those pixels independently in T1 and T2 as "
            "building, background, mixed, or uncertain. Use mixed when the component spans both "
            "building and background. Return exactly one compact JSON object mapping every exact "
            "region_id to [t1_state,t2_state], for example "
            "{\"d0\":[\"building\",\"building\"],\"d1\":[\"background\",\"building\"]}. "
            "Do not output candidate quality, added/removed effect labels, feedback, coordinates, "
            "or extra keys.\n"
            f"Delta geometry facts: {json.dumps(prompt_facts, ensure_ascii=False)}\n"
            f"{correction}"
        )
        content: list[dict[str, Any]] = [
            {"type": "text", "text": "Fixed clean T1 original image:"},
            {"type": "image", "image": self._as_image(state.t1_image)},
            {"type": "text", "text": "Fixed clean T2 original image:"},
            {"type": "image", "image": self._as_image(state.t2_image)},
        ]
        for proposal in proposals:
            public_proposal = {
                key: proposal[key]
                for key in (
                    "region_id",
                    "component_area",
                    "box_normalized",
                    "candidate_delta_pixels",
                    "delta_pixels",
                )
                if key in proposal
            }
            content.extend(
                [
                    {
                        "type": "text",
                        "text": (
                            f"RGB temporal-state proposal {proposal['region_id']}: "
                            f"{json.dumps(public_proposal, ensure_ascii=False)}"
                        ),
                    },
                    {
                        "type": "image",
                        "image": self._rgb_delta_panel(state, previous_state, proposal),
                    },
                ]
            )
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    @staticmethod
    def _derive_regional_analysis(
        judgments: tuple[_RegionJudgment, ...],
        proposals: list[dict[str, Any]],
        mask_facts: dict[str, Any],
        previous_action: AgentAction | None,
    ) -> _RegionalAnalysis:
        if not judgments:
            raise ValueError("no mask-derived proposal is available for a reliable diagnosis")
        lookup = {item["region_id"]: item for item in proposals}
        false_positives = [item for item in judgments if item.verdict == "false_positive"]
        false_negatives = [item for item in judgments if item.verdict == "false_negative"]
        uncertain = [item for item in judgments if item.verdict == "uncertain"]
        if false_positives and false_negatives:
            error_type = "mixed_error"
            candidates = false_positives + false_negatives
        elif false_positives:
            error_type = "false_positive_change"
            candidates = false_positives
        elif false_negatives:
            error_type = "false_negative"
            candidates = false_negatives
        elif uncertain:
            error_type = "uncertain_region"
            candidates = uncertain
        else:
            error_type = "none"
            candidates = []
        uncovered = int(mask_facts.get("initial_audit_uncovered_pixels", 0))
        if error_type == "none" and uncovered:
            error_type = "uncertain_region"
        selected = (
            min(candidates, key=lambda item: lookup[item.region_id]["component_area"])
            if candidates
            else None
        )
        target_view = (
            selected.target_view
            if selected is not None and selected.target_view in {"t1", "t2"}
            else previous_action.target_view if previous_action else "t2"
        )
        facts = (
            f"Current change mask contains {int(mask_facts.get('change_pixels', 0))} white pixels "
            f"across {len(proposals)} inspected audit proposals."
        )
        detail = (
            selected.feedback
            if selected is not None
            else (
                f"{uncovered} audit pixels fall outside the supplied panels; "
                "finish is not authorized."
            )
            if uncovered
            else "All inspected audit components are supported by the RGB crops."
        )
        suggested_action = (
            "box"
            if error_type == "mixed_error"
            else selected.suggested_action
            if selected is not None and selected.suggested_action is not None
            else "box" if uncovered and selected is None else None
        )
        if selected is not None and suggested_action in {
            "positive_point",
            "negative_point",
        }:
            seed_x, seed_y = lookup[selected.region_id]["component_seed_normalized"]
            region = (seed_x, seed_y, seed_x, seed_y)
        elif selected is not None:
            region = tuple(lookup[selected.region_id]["box_normalized"])
        elif uncovered and mask_facts.get("initial_audit_uncovered_box_normalized"):
            region = tuple(mask_facts["initial_audit_uncovered_box_normalized"])
        else:
            region = None
        return _RegionalAnalysis(
            judgments,
            error_type,
            target_view,
            region,
            f"{facts} {detail}",
            suggested_action,
        )

    def _parse_effect_payload(
        self, payload: dict[str, Any], proposals: list[dict[str, Any]]
    ) -> tuple[_EffectJudgment, ...]:
        expected = [item["region_id"] for item in proposals]
        if set(payload) != set(expected):
            raise ValueError("effect response must cover every delta region exactly once")
        by_id = {item["region_id"]: item for item in proposals}
        result: list[_EffectJudgment] = []
        for region_id in expected:
            effect = payload[region_id]
            if not isinstance(effect, str) or effect not in self.EFFECT_LABELS:
                raise ValueError("unsupported candidate effect label")
            effect_kind = by_id[region_id].get("effect_kind")
            if effect_kind == "added" and effect.startswith("removed_"):
                raise ValueError("added delta region cannot receive a removed effect label")
            if effect_kind == "removed" and effect.startswith("added_"):
                raise ValueError("removed delta region cannot receive an added effect label")
            result.append(_EffectJudgment(region_id, effect))
        return tuple(result)

    def _parse_temporal_state_payload(
        self, payload: dict[str, Any], proposals: list[dict[str, Any]]
    ) -> tuple[_TemporalStateJudgment, ...]:
        expected = [item["region_id"] for item in proposals]
        if set(payload) != set(expected):
            raise ValueError("temporal-state response must cover every delta region exactly once")
        result: list[_TemporalStateJudgment] = []
        for region_id in expected:
            states = payload[region_id]
            if not isinstance(states, (list, tuple)) or len(states) != 2:
                raise TypeError("each temporal-state value must be [t1_state,t2_state]")
            t1_state, t2_state = states
            if (
                not isinstance(t1_state, str)
                or not isinstance(t2_state, str)
                or t1_state not in self.TEMPORAL_STATES
                or t2_state not in self.TEMPORAL_STATES
            ):
                raise ValueError("unsupported RGB temporal state")
            result.append(_TemporalStateJudgment(region_id, t1_state, t2_state))
        return tuple(result)

    @staticmethod
    def _initial_judgments_from_temporal_states(
        temporal_judgments: tuple[_TemporalStateJudgment, ...],
        proposals: list[dict[str, Any]],
        state: ChangeState,
    ) -> tuple[tuple[_RegionJudgment, ...], list[dict[str, Any]]]:
        """Derive initial FP/FN semantics from RGB facts and mask-owned geometry."""

        by_temporal = {item.region_id: item for item in temporal_judgments}
        judgments: list[_RegionJudgment] = []
        detail: list[dict[str, Any]] = []
        for proposal in proposals:
            region_id = proposal["region_id"]
            temporal = by_temporal[region_id]
            component = Qwen3VLZeroShotVerifier._proposal_initial_component(
                state, proposal
            )
            component_pixels = max(1, int(component.sum()))
            t1_ratio = float(np.logical_and(component, state.t1_mask).sum()) / component_pixels
            t2_ratio = float(np.logical_and(component, state.t2_mask).sum()) / component_pixels
            predicted_views = [
                view
                for view, ratio in (("t1", t1_ratio), ("t2", t2_ratio))
                if ratio >= 0.5
            ]
            audit_kind = proposal.get("audit_kind", "mixed")
            states = {temporal.t1_state, temporal.t2_state}
            temporal_change: bool | None
            target_view: str | None = None
            suggested_action: str | None = None
            if audit_kind not in {"present", "missing"} or states & {
                "mixed",
                "uncertain",
            }:
                verdict = "uncertain"
                temporal_change = None
                suggested_action = "box"
            else:
                temporal_change = temporal.t1_state != temporal.t2_state
                if audit_kind == "present" and temporal_change:
                    verdict = "true_change"
                elif audit_kind == "missing" and not temporal_change:
                    verdict = "no_error"
                elif audit_kind == "missing":
                    verdict = "false_negative"
                    target_view = (
                        "t1" if temporal.t1_state == "building" else "t2"
                    )
                    suggested_action = "positive_point"
                elif len(predicted_views) != 1:
                    # A current change component with both/neither predicted masks is
                    # a matching inconsistency, not a safe one-view segmentation edit.
                    verdict = "uncertain"
                    temporal_change = False
                    suggested_action = "box"
                else:
                    verdict = "false_positive"
                    predicted_view = predicted_views[0]
                    if temporal.t1_state == temporal.t2_state == "building":
                        target_view = "t2" if predicted_view == "t1" else "t1"
                        suggested_action = "positive_point"
                    else:
                        target_view = predicted_view
                        suggested_action = "negative_point"
            feedback = (
                f"{region_id} RGB states are {temporal.t1_state}/{temporal.t2_state}; "
                f"runtime audit kind is {audit_kind}, derived verdict is {verdict}."
            )
            judgments.append(
                _RegionJudgment(
                    region_id,
                    verdict,
                    target_view,
                    feedback,
                    suggested_action,
                )
            )
            detail.append(
                {
                    "region_id": region_id,
                    "audit_kind": audit_kind,
                    "t1_state": temporal.t1_state,
                    "t2_state": temporal.t2_state,
                    "temporal_change": temporal_change,
                    "t1_predicted_component_ratio": t1_ratio,
                    "t2_predicted_component_ratio": t2_ratio,
                    "derived_verdict": verdict,
                    "target_view": target_view,
                    "suggested_action": suggested_action,
                    "decision_source": "rgb_temporal_state_plus_runtime_geometry",
                }
            )
        return tuple(judgments), detail

    @staticmethod
    def _effects_from_temporal_states(
        temporal_judgments: tuple[_TemporalStateJudgment, ...],
        mask_judgments: tuple[_EffectJudgment, ...] | None,
        proposals: list[dict[str, Any]],
    ) -> tuple[tuple[_EffectJudgment, ...], list[dict[str, Any]]]:
        by_temporal = {item.region_id: item for item in temporal_judgments}
        by_mask = {item.region_id: item.effect for item in (mask_judgments or ())}
        result: list[_EffectJudgment] = []
        detail: list[dict[str, Any]] = []
        for proposal in proposals:
            region_id = proposal["region_id"]
            temporal = by_temporal[region_id]
            states = {temporal.t1_state, temporal.t2_state}
            if states & {"mixed", "uncertain"}:
                effect = "uncertain"
                temporal_change = None
            else:
                temporal_change = temporal.t1_state != temporal.t2_state
                if proposal.get("effect_kind") == "added":
                    effect = (
                        "added_true_change" if temporal_change else "added_false_change"
                    )
                elif proposal.get("effect_kind") == "removed":
                    effect = (
                        "removed_true_change"
                        if temporal_change
                        else "removed_false_positive"
                    )
                else:
                    raise ValueError("delta proposal has no valid effect_kind")
            mask_effect = by_mask.get(region_id)
            result.append(_EffectJudgment(region_id, effect))
            detail.append(
                {
                    "region_id": region_id,
                    "t1_state": temporal.t1_state,
                    "t2_state": temporal.t2_state,
                    "temporal_change": temporal_change,
                    "derived_effect": effect,
                    "mask_context_effect": mask_effect,
                    "mask_context_agreement": (
                        mask_effect == effect if mask_effect is not None else None
                    ),
                    "decision_source": "rgb_temporal_state",
                }
            )
        return tuple(result), detail

    @staticmethod
    def _comparison_from_effects(
        judgments: tuple[_EffectJudgment, ...]
    ) -> str:
        beneficial = {"added_true_change", "removed_false_positive"}
        harmful = {"added_false_change", "removed_true_change"}
        labels = {item.effect for item in judgments}
        if labels & {"mixed", "uncertain"} or not labels:
            return "uncertain"
        if labels & beneficial and labels & harmful:
            return "uncertain"
        if labels & harmful:
            return "worse"
        if labels <= beneficial:
            return "better"
        return "uncertain"

    @staticmethod
    def _derive_effect_output(
        judgments: tuple[_EffectJudgment, ...],
        proposals: list[dict[str, Any]],
        previous_action: AgentAction | None,
        comparison: str,
    ) -> VerifierOutput:
        by_id = {item["region_id"]: item for item in proposals}
        target_view = previous_action.target_view if previous_action else "t2"
        if comparison == "better":
            return VerifierOutput(
                quality_score=None,
                progress_score=None,
                score_delta=0.0,
                comparison="better",
                error_type="none",
                target_view=target_view,
                error_region=None,
                suggested_action="finish",
                feedback=(
                    "All inspected candidate delta components have beneficial effect labels; "
                    "the comparison was derived by the runtime."
                ),
                accept=True,
                verifier_valid=True,
                localization_valid=True,
                stop=True,
            )

        selected = next(
            (
                item
                for item in judgments
                if item.effect in {"added_false_change", "removed_true_change"}
            ),
            judgments[0],
        )
        region = tuple(by_id[selected.region_id]["box_normalized"])
        if selected.effect == "added_false_change":
            error_type = "false_positive_change"
            suggested_action = "negative_point"
        elif selected.effect == "removed_true_change":
            error_type = "false_negative"
            suggested_action = "positive_point"
        else:
            error_type = "uncertain_region"
            suggested_action = "box"
        return VerifierOutput(
            quality_score=None,
            progress_score=None,
            score_delta=0.0,
            comparison=comparison,
            error_type=error_type,
            target_view=target_view,
            error_region=region,
            suggested_action=suggested_action,
            feedback=(
                f"Runtime-derived candidate comparison={comparison}; "
                f"{selected.region_id} effect={selected.effect}."
            ),
            accept=False,
            verifier_valid=True,
            localization_valid=True,
            stop=False,
        )

    @staticmethod
    def _states_identical(state: ChangeState, previous_state: ChangeState) -> bool:
        return (
            np.array_equal(state.t1_mask, previous_state.t1_mask)
            and np.array_equal(state.t2_mask, previous_state.t2_mask)
            and np.array_equal(state.change_mask, previous_state.change_mask)
        )

    def _candidate_fingerprint(
        self,
        state: ChangeState,
        previous_state: ChangeState,
        previous_action: AgentAction | None,
        proposals: list[dict[str, Any]],
        facts: dict[str, Any],
    ) -> str:
        digest = hashlib.sha256()
        for array in (
            state.t1_image,
            state.t2_image,
            previous_state.t1_mask,
            previous_state.t2_mask,
            previous_state.change_mask,
            state.t1_mask,
            state.t2_mask,
            state.change_mask,
        ):
            value = np.ascontiguousarray(array)
            digest.update(str(value.shape).encode("ascii"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(value.tobytes())
        identity = {
            "schema_version": self.SCHEMA_VERSION,
            "query": state.query,
            "action": previous_action.to_dict() if previous_action else None,
            "max_new_tokens": self.max_new_tokens,
            "max_retries": self.max_retries,
            "max_delta_component_ratio_without_consensus": (
                self.max_delta_component_ratio_without_consensus
            ),
            "candidate_evidence_modes": self.CANDIDATE_EVIDENCE_MODES,
            "model": getattr(getattr(self.model, "config", None), "_name_or_path", None),
            "proposals": proposals,
            "facts": facts,
        }
        digest.update(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        return digest.hexdigest()

    def _generate_messages(self, messages: list[dict[str, Any]]) -> str:
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        device = getattr(self.model, "device", None)
        if device is not None and hasattr(inputs, "to"):
            inputs = inputs.to(device)
        outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        input_ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids
        generated = outputs[:, input_ids.shape[1] :]
        return self.processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

    @staticmethod
    def _extract_json_object(raw: str) -> dict[str, Any]:
        decoder = json.JSONDecoder()
        start = raw.find("{")
        if start < 0:
            raise ValueError("verifier response contains no JSON object")
        try:
            value, _ = decoder.raw_decode(raw[start:])
        except json.JSONDecodeError as error:
            # Do not scan into nested objects: a truncated top-level response
            # must remain invalid instead of being misread as one region.
            raise ValueError("verifier response contains incomplete JSON object") from error
        if not isinstance(value, dict):
            raise ValueError("verifier response JSON must be an object")
        return value

    @staticmethod
    def _public_action(
        action: AgentAction | None, image_size: tuple[int, int]
    ) -> dict[str, Any] | None:
        if action is None:
            return None
        result: dict[str, Any] = {
            "target_view": action.target_view,
            "action": action.action,
            "coordinate_space": "normalized_0_1000",
        }
        if action.coordinate is not None:
            result["coordinate"] = list(
                pixel_point_to_normalized(action.coordinate, image_size)
            )
        if action.box is not None:
            result["box"] = list(pixel_box_to_normalized(action.box, image_size))
        return result

    @staticmethod
    def _region_panel(
        state: ChangeState,
        proposal: dict[str, Any],
        previous_state: ChangeState | None = None,
        panel_size: int = 192,
    ) -> Image.Image:
        x1, y1, x2, y2 = (int(value) for value in proposal["box_pixels"])
        region = (slice(y1, y2 + 1), slice(x1, x2 + 1))
        change = np.asarray(state.change_mask[region], dtype=bool)
        delta_component = None
        if previous_state is not None and proposal.get("component_seed_pixels"):
            delta_component = Qwen3VLZeroShotVerifier._proposal_delta_component(
                state, previous_state, proposal
            )[region]
        t1 = np.array(
            Qwen3VLZeroShotVerifier._as_image(state.t1_image[region]).convert("RGB"),
            copy=True,
        )
        t2 = np.array(
            Qwen3VLZeroShotVerifier._as_image(state.t2_image[region]).convert("RGB"),
            copy=True,
        )
        overlay = change if delta_component is None else delta_component
        for image in (t1, t2):
            image[overlay] = np.clip(
                image[overlay].astype(np.float32) * 0.45 + np.array([140, 0, 140]),
                0,
                255,
            ).astype(np.uint8)
        if previous_state is None:
            change_rgb = np.repeat((change.astype(np.uint8) * 255)[..., None], 3, axis=2)
        else:
            previous_change = np.asarray(previous_state.change_mask[region], dtype=bool)
            if delta_component is not None:
                stable = np.logical_and(previous_change, change)
                previous_change = np.logical_or(
                    stable, np.logical_and(previous_change, delta_component)
                )
                change = np.logical_or(stable, np.logical_and(change, delta_component))
            change_rgb = np.zeros((*change.shape, 3), dtype=np.uint8)
            change_rgb[..., 0] = previous_change.astype(np.uint8) * 255
            change_rgb[..., 1] = change.astype(np.uint8) * 255
            change_rgb[..., 2] = overlay.astype(np.uint8) * 255
        temporal = np.zeros((*change.shape, 3), dtype=np.uint8)
        temporal[..., 0] = np.asarray(state.t1_mask[region], dtype=np.uint8) * 255
        temporal[..., 1] = np.asarray(state.t2_mask[region], dtype=np.uint8) * 255
        temporal[..., 2] = overlay.astype(np.uint8) * 255
        tiles = [
            Image.fromarray(t1),
            Image.fromarray(t2),
            Image.fromarray(change_rgb),
            Image.fromarray(temporal),
        ]
        canvas = Image.new("RGB", (panel_size * 2, panel_size * 2))
        for index, tile in enumerate(tiles):
            resample = Image.Resampling.BILINEAR if index < 2 else Image.Resampling.NEAREST
            canvas.paste(
                tile.resize((panel_size, panel_size), resample),
                ((index % 2) * panel_size, (index // 2) * panel_size),
            )
        return canvas

    @staticmethod
    def _rgb_delta_panel(
        state: ChangeState,
        previous_state: ChangeState,
        proposal: dict[str, Any],
        panel_size: int = 192,
    ) -> Image.Image:
        """Show outlined RGB evidence, exact delta geometry, and raw difference."""

        x1, y1, x2, y2 = (int(value) for value in proposal["box_pixels"])
        region = (slice(y1, y2 + 1), slice(x1, x2 + 1))
        delta = Qwen3VLZeroShotVerifier._proposal_delta_component(
            state, previous_state, proposal
        )[region]
        raw_t1 = state.t1_image[region]
        raw_t2 = state.t2_image[region]
        t1 = Qwen3VLZeroShotVerifier._outline_component(raw_t1, delta)
        t2 = Qwen3VLZeroShotVerifier._outline_component(raw_t2, delta)
        delta_rgb = np.repeat((delta.astype(np.uint8) * 255)[..., None], 3, axis=2)
        absolute_difference = np.abs(
            raw_t2.astype(np.int16) - raw_t1.astype(np.int16)
        )
        absolute_difference = np.clip(absolute_difference * 2, 0, 255).astype(np.uint8)
        tiles = [
            Image.fromarray(t1),
            Image.fromarray(t2),
            Image.fromarray(delta_rgb),
            Image.fromarray(absolute_difference),
        ]
        canvas = Image.new("RGB", (panel_size * 2, panel_size * 2))
        for index, tile in enumerate(tiles):
            resample = Image.Resampling.NEAREST if index == 2 else Image.Resampling.BILINEAR
            canvas.paste(
                tile.resize((panel_size, panel_size), resample),
                ((index % 2) * panel_size, (index // 2) * panel_size),
            )
        return canvas

    @staticmethod
    def _rgb_initial_panel(
        state: ChangeState,
        proposal: dict[str, Any],
        panel_size: int = 192,
    ) -> Image.Image:
        """Show outlined RGB evidence and exact initial component geometry."""

        x1, y1, x2, y2 = (int(value) for value in proposal["box_pixels"])
        region = (slice(y1, y2 + 1), slice(x1, x2 + 1))
        component = Qwen3VLZeroShotVerifier._proposal_initial_component(
            state, proposal
        )[region]
        raw_t1 = state.t1_image[region]
        raw_t2 = state.t2_image[region]
        t1 = Qwen3VLZeroShotVerifier._outline_component(raw_t1, component)
        t2 = Qwen3VLZeroShotVerifier._outline_component(raw_t2, component)
        component_rgb = np.repeat(
            (component.astype(np.uint8) * 255)[..., None], 3, axis=2
        )
        absolute_difference = np.abs(
            raw_t2.astype(np.int16) - raw_t1.astype(np.int16)
        )
        absolute_difference = np.clip(absolute_difference * 2, 0, 255).astype(np.uint8)
        tiles = [
            Image.fromarray(t1),
            Image.fromarray(t2),
            Image.fromarray(component_rgb),
            Image.fromarray(absolute_difference),
        ]
        canvas = Image.new("RGB", (panel_size * 2, panel_size * 2))
        for index, tile in enumerate(tiles):
            resample = Image.Resampling.NEAREST if index == 2 else Image.Resampling.BILINEAR
            canvas.paste(
                tile.resize((panel_size, panel_size), resample),
                ((index % 2) * panel_size, (index // 2) * panel_size),
            )
        return canvas

    @staticmethod
    def _outline_component(rgb: np.ndarray, component: np.ndarray) -> np.ndarray:
        """Draw an outer yellow ring without covering component RGB pixels."""

        image = np.array(rgb, dtype=np.uint8, copy=True)
        mask = np.asarray(component, dtype=bool)
        padded = np.pad(mask, 1, mode="constant", constant_values=False)
        dilated = np.zeros_like(mask)
        for offset_y in range(3):
            for offset_x in range(3):
                dilated |= padded[
                    offset_y : offset_y + mask.shape[0],
                    offset_x : offset_x + mask.shape[1],
                ]
        outline = np.logical_and(dilated, ~mask)
        image[outline] = np.array([255, 255, 0], dtype=np.uint8)
        return image

    @staticmethod
    def _proposal_initial_component(
        state: ChangeState, proposal: dict[str, Any]
    ) -> np.ndarray:
        audit_kind = proposal.get("audit_kind")
        if audit_kind == "present":
            source = np.asarray(state.change_mask, dtype=bool)
        elif audit_kind == "missing":
            source = np.logical_and(
                np.logical_xor(state.t1_mask, state.t2_mask),
                ~state.change_mask,
            )
        else:
            source = np.logical_or(
                state.change_mask,
                np.logical_and(
                    np.logical_xor(state.t1_mask, state.t2_mask),
                    ~state.change_mask,
                ),
            )
        seed_x, seed_y = proposal.get("component_seed_pixels", (None, None))
        if seed_x is None or seed_y is None:
            raise ValueError("initial proposal has no component seed")
        component = next(
            (
                item
                for item in connected_components(source)
                if item[int(seed_y), int(seed_x)]
            ),
            None,
        )
        if component is None:
            raise ValueError("initial proposal seed is outside its audit component")
        return component

    @staticmethod
    def _proposal_delta_component(
        state: ChangeState,
        previous_state: ChangeState,
        proposal: dict[str, Any],
    ) -> np.ndarray:
        if proposal.get("effect_kind") == "added":
            full_delta = np.logical_and(state.change_mask, ~previous_state.change_mask)
        elif proposal.get("effect_kind") == "removed":
            full_delta = np.logical_and(previous_state.change_mask, ~state.change_mask)
        else:
            raise ValueError("delta proposal has no valid effect_kind")
        seed_x, seed_y = proposal.get("component_seed_pixels", (None, None))
        if seed_x is None or seed_y is None:
            raise ValueError("delta proposal has no component seed")
        component = next(
            (
                item
                for item in connected_components(full_delta)
                if item[int(seed_y), int(seed_x)]
            ),
            None,
        )
        if component is None:
            raise ValueError("delta proposal seed does not identify a candidate component")
        return component

    @staticmethod
    def _as_image(value: Any) -> Image.Image:
        if isinstance(value, Image.Image):
            return value
        array = np.asarray(value)
        if array.dtype != np.uint8:
            if array.max(initial=0) <= 1:
                array = array * 255
            array = np.clip(array, 0, 255).astype(np.uint8)
        return Image.fromarray(array)

    @staticmethod
    def _mask_image(mask: np.ndarray) -> Image.Image:
        return Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255, mode="L")
