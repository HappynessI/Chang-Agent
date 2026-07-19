"""Compact region diagnosis and programmatic candidate-effect verification."""

from __future__ import annotations

import copy
import hashlib
import json
import re
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


@dataclass(frozen=True)
class _RegionalAnalysis:
    judgments: tuple[_RegionJudgment, ...]
    error_type: str
    target_view: str
    error_region: tuple[int, int, int, int] | None
    feedback: str


@dataclass(frozen=True)
class _PairwiseDecision:
    comparison: str
    feedback: str


@dataclass(frozen=True)
class _EffectJudgment:
    region_id: str
    effect: str


class Qwen3VLZeroShotVerifier:
    """Classify initial regions and candidate delta effects with compact outputs.

    Qwen never predicts an absolute score or a pairwise comparison. Environment-owned
    boxes make every decision local and auditable; candidate ``better/worse`` is
    derived from added/removed effect labels by deterministic runtime rules.
    """

    SCHEMA_VERSION = "compact_delta_effect_v2"
    ERROR_TYPES = {
        "none",
        "false_positive_change",
        "false_negative",
        "mixed_error",
        "uncertain_region",
    }
    REGION_VERDICTS = {
        "true_change",
        "false_positive",
        "false_negative",
        "uncertain",
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
    VIEWS = {"t1", "t2"}

    def __init__(
        self,
        *,
        model: Any,
        processor: Any,
        max_new_tokens: int = 1024,
        accept_threshold: float = 0.82,
        max_retries: int = 2,
        **legacy_localization_options: Any,
    ):
        if not 0 <= accept_threshold <= 1:
            raise ValueError("accept_threshold must be in [0, 1]")
        if max_retries < 1:
            raise ValueError("max_retries must be positive")
        self.model = model
        self.processor = processor
        self.max_new_tokens = max_new_tokens
        # Retained only for CLI/config compatibility. Pairwise mode has no score
        # threshold and records this explicitly in evidence.
        self.accept_threshold = accept_threshold
        self.max_retries = max_retries
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

        region_errors: list[str] = []
        region_attempts: list[dict[str, Any]] = []
        regional_analysis: _RegionalAnalysis | None = None
        for _ in range(self.max_retries):
            raw = self._generate_messages(
                self.build_messages(
                    state,
                    None,
                    previous_action,
                    previous_state,
                    region_errors[-1:],
                )
            )
            try:
                payload = self._extract_json_object(raw)
                judgments = self._parse_region_payload(payload, proposals, mask_facts)
                regional_analysis = self._derive_regional_analysis(
                    judgments, proposals, mask_facts, previous_action
                )
                region_attempts.append({"raw": raw, "output": payload})
                break
            except (TypeError, ValueError, KeyError) as error:
                region_errors.append(str(error))
                region_attempts.append({"raw": raw, "error": str(error)})

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
            "decision_mode": "compact_regions_then_programmatic_delta_effect",
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
        judgments: tuple[_EffectJudgment, ...] | None = None
        uncovered = int(facts.get("candidate_delta_uncovered_pixels", 0))
        if uncovered:
            effect_errors.append(
                f"candidate delta has {uncovered} pixels outside the compact proposal set"
            )
        elif not proposals:
            effect_errors.append("candidate delta has no auditable proposal")
        else:
            for _ in range(self.max_retries):
                raw = self._generate_messages(
                    self.build_effect_messages(
                        state,
                        previous_state,
                        previous_action,
                        proposals,
                        facts,
                        effect_errors[-1:],
                    )
                )
                try:
                    payload = self._extract_json_object(raw)
                    judgments = self._parse_effect_payload(payload, proposals)
                    effect_attempts.append({"raw": raw, "output": payload})
                    break
                except (TypeError, ValueError, KeyError) as error:
                    effect_errors.append(str(error))
                    effect_attempts.append({"raw": raw, "error": str(error)})

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
                    "decision_mode": "programmatic_delta_effect",
                    "candidate_fingerprint": fingerprint,
                    "decision_key": fingerprint,
                    "decision_step": state.step_index,
                    "cache_hit": False,
                    "effect_attempts": effect_attempts,
                    "mask_facts": facts,
                    "region_proposals": proposals,
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
            "decision_mode": "programmatic_delta_effect",
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
            "decision_mode": "compact_regions_then_programmatic_delta_effect",
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
        if analysis.error_type == "none":
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
    ) -> list[dict[str, Any]]:
        del previous_score, previous_state
        proposals = list(state.evidence.get("verifier_region_proposals", []))
        facts = dict(state.evidence.get("verifier_mask_facts", {}))
        correction = (
            f"Your previous response was invalid: {previous_errors[-1]}. Correct it.\n"
            if previous_errors
            else ""
        )
        prompt = (
            "Classify each Environment-proposed local region for building change detection. "
            "The full images and masks preserve global context; each local panel makes small "
            "white mask components visible. Panel quadrants are: top-left T1 crop with current "
            "change highlighted magenta, top-right T2 crop with the same highlight, bottom-left "
            "change comparison (white candidate change for the initial state; red=previous, "
            "green=candidate, blue=delta for pairwise state), bottom-right temporal masks "
            "(red=T1 object, green=T2 object, blue=current change). A true temporal building "
            "change is added (background T1/building T2) or disappeared (building T1/background "
            "T2). Unchanged buildings/background are not change. Predicted temporal masks are "
            "supporting predictions, not GT; verify against RGB crops. For each exact region_id, "
            "return true_change when existing white change pixels are supported, false_positive "
            "when existing white pixels are unsupported, false_negative only when a real change "
            "is absent from the current white mask, or uncertain. false_negative is allowed only "
            "when the proposal sources include temporal_difference_missing; it is forbidden for "
            "an ordinary white change component. target_view must be t1 or "
            "t2 for false_positive/false_negative and null otherwise. Return exactly one compact "
            "JSON object mapping every exact region_id to [verdict,target_view], for example "
            "{\"r0\":[\"true_change\",null],\"r1\":[\"false_positive\",\"t2\"]}. "
            "Do not output feedback sentences, scores, comparison, coordinates, actions, GT "
            "claims, or extra keys.\n"
            f"Exact global mask facts (authoritative, not inferred visually): "
            f"{json.dumps(facts, ensure_ascii=False)}\n"
            f"OmniOVCD matching summary (supporting evidence, not GT): "
            f"{json.dumps(state.evidence.get('matching', {}), ensure_ascii=False, default=str)}\n"
            "If change_pixels is greater than zero, the current change mask is NOT empty and "
            "you must not call it empty.\n"
            f"Action that produced the candidate: "
            f"{json.dumps(self._public_action(previous_action, state.image_size), ensure_ascii=False)}\n"
            f"{correction}"
        )
        content = self._global_visual_content(state)
        for proposal in proposals:
            content.extend(
                [
                    {
                        "type": "text",
                        "text": (
                            f"Local proposal {proposal['region_id']} with exact metadata: "
                            f"{json.dumps(proposal, ensure_ascii=False)}"
                        ),
                    },
                    {"type": "image", "image": self._region_panel(state, proposal)},
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
            "Judge only the pixels changed by the candidate action. Each supplied delta proposal "
            "has effect_kind added or removed and exact pixel counts. Confirm temporal building "
            "change from the T1/T2 RGB crops; predicted masks are supporting predictions, not GT. "
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

    @staticmethod
    def _global_visual_content(state: ChangeState) -> list[dict[str, Any]]:
        return [
            {"type": "text", "text": "Full T1 original image:"},
            {"type": "image", "image": Qwen3VLZeroShotVerifier._as_image(state.t1_image)},
            {"type": "text", "text": "Full T2 original image:"},
            {"type": "image", "image": Qwen3VLZeroShotVerifier._as_image(state.t2_image)},
            {"type": "text", "text": "Full predicted T1 object mask:"},
            {"type": "image", "image": Qwen3VLZeroShotVerifier._mask_image(state.t1_mask)},
            {"type": "text", "text": "Full predicted T2 object mask:"},
            {"type": "image", "image": Qwen3VLZeroShotVerifier._mask_image(state.t2_mask)},
            {"type": "text", "text": "Full candidate final change mask:"},
            {"type": "image", "image": Qwen3VLZeroShotVerifier._mask_image(state.change_mask)},
        ]

    def _parse_region_payload(
        self,
        payload: dict[str, Any],
        proposals: list[dict[str, Any]],
        mask_facts: dict[str, Any],
    ) -> tuple[_RegionJudgment, ...]:
        expected_ids = [item["region_id"] for item in proposals]
        values = self._normalize_region_values(payload, expected_ids)
        if len(values) != len(expected_ids):
            raise ValueError("region response must cover every proposal exactly once")
        by_id = {item["region_id"]: item for item in proposals}
        judgments: list[_RegionJudgment] = []
        seen: set[str] = set()
        feedback_text: list[str] = []
        for value in values:
            required = {"region_id", "verdict", "target_view"}
            allowed = required | {"feedback"}
            if (
                not isinstance(value, dict)
                or not required.issubset(value)
                or set(value) - allowed
            ):
                raise ValueError("each region judgment has unexpected or missing fields")
            region_id = value["region_id"]
            if region_id not in by_id or region_id in seen:
                raise ValueError("unknown or duplicate region_id")
            verdict = value["verdict"]
            if verdict not in self.REGION_VERDICTS:
                raise ValueError("unsupported region verdict")
            target_view = value["target_view"]
            if verdict in {"false_positive", "false_negative"}:
                if target_view not in self.VIEWS:
                    raise ValueError("actionable region judgment requires target_view t1/t2")
            elif target_view is not None:
                raise ValueError(
                    "true_change/uncertain region judgment requires target_view null"
                )
            feedback = value.get("feedback")
            if feedback is None:
                feedback = f"{region_id} classified as {verdict}."
            if not isinstance(feedback, str) or not feedback.strip():
                raise TypeError("region feedback must be a non-empty string when supplied")
            if verdict in {"true_change", "false_positive"} and not by_id[region_id]["change_pixels"]:
                raise ValueError(f"{verdict} requires white change pixels in the proposal")
            if (
                verdict == "false_negative"
                and "temporal_difference_missing"
                not in by_id[region_id].get("sources", [])
            ):
                raise ValueError(
                    "false_negative is impossible without a mask-derived missing-change "
                    "proposal"
                )
            seen.add(region_id)
            feedback_text.append(feedback)
            judgments.append(
                _RegionJudgment(region_id, verdict, target_view, feedback.strip())
            )
        if seen != set(expected_ids):
            raise ValueError("region response omitted a proposal")
        if int(mask_facts.get("change_pixels", 0)) > 0 and self._claims_empty(
            " ".join(feedback_text)
        ):
            raise ValueError(
                "diagnosis contradicts authoritative mask facts: current change mask is not empty"
            )
        return tuple(judgments)

    @staticmethod
    def _normalize_region_values(
        payload: dict[str, Any], expected_ids: list[str]
    ) -> list[dict[str, Any]]:
        """Normalize compact output and legacy verbose forms.

        The current prompt asks for ``{"r0": [verdict, view]}``. Older saved
        responses used either ``{"regions": [...]}`` or region-keyed objects;
        retaining those parsers keeps replay artifacts readable. Ordering is always
        restored to Environment proposal order.
        """
        if set(payload) == {"regions"}:
            values = payload["regions"]
            if not isinstance(values, list):
                raise TypeError("regions must be a list")
            return values

        expected = set(expected_ids)
        if set(payload) == expected and all(
            isinstance(payload[region_id], (list, tuple))
            and len(payload[region_id]) == 2
            for region_id in expected_ids
        ):
            return [
                {
                    "region_id": region_id,
                    "verdict": payload[region_id][0],
                    "target_view": payload[region_id][1],
                }
                for region_id in expected_ids
            ]
        if set(payload) == expected and all(
            isinstance(payload[region_id], dict) for region_id in expected_ids
        ):
            values: list[dict[str, Any]] = []
            for region_id in expected_ids:
                value = dict(payload[region_id])
                # The key is authoritative when the model omits the repeated
                # field; retain the strict downstream schema after injection.
                value.setdefault("region_id", region_id)
                values.append(value)
            return values

        raise ValueError(
            "region response must use compact region pairs or a complete legacy form"
        )

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
            max(candidates, key=lambda item: lookup[item.region_id]["component_area"])
            if candidates
            else None
        )
        target_view = (
            selected.target_view
            if selected is not None and selected.target_view in {"t1", "t2"}
            else previous_action.target_view if previous_action else "t2"
        )
        region = (
            tuple(lookup[selected.region_id]["box_normalized"])
            if selected is not None
            else tuple(mask_facts["initial_audit_uncovered_box_normalized"])
            if uncovered and mask_facts.get("initial_audit_uncovered_box_normalized")
            else None
        )
        facts = (
            f"Current change mask contains {int(mask_facts.get('change_pixels', 0))} white pixels "
            f"across {len(proposals)} inspected proposals."
        )
        detail = (
            selected.feedback
            if selected is not None
            else (
                f"{uncovered} audit pixels fall outside the supplied panels; "
                "finish is not authorized."
            )
            if uncovered
            else "All proposed white change regions are supported by the inspected RGB crops."
        )
        return _RegionalAnalysis(
            judgments, error_type, target_view, region, f"{facts} {detail}"
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
            "model": getattr(getattr(self.model, "config", None), "_name_or_path", None),
            "proposals": proposals,
            "facts": facts,
        }
        digest.update(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        return digest.hexdigest()

    @staticmethod
    def _claims_empty(text: str) -> bool:
        return bool(
            re.search(
                r"\b(?:candidate|current|change)\s+(?:change\s+)?mask\s+(?:is|appears|looks)\s+empty\b|\b(?:candidate|current|change)\s+mask\s+contains\s+no\s+white\s+pixels\b",
                text,
                flags=re.IGNORECASE,
            )
        )

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
            if proposal.get("effect_kind") == "added":
                full_delta = np.logical_and(
                    state.change_mask, ~previous_state.change_mask
                )
            elif proposal.get("effect_kind") == "removed":
                full_delta = np.logical_and(
                    previous_state.change_mask, ~state.change_mask
                )
            else:
                raise ValueError("delta proposal has no valid effect_kind")
            seed_x, seed_y = proposal["component_seed_pixels"]
            delta_component = next(
                (
                    component
                    for component in connected_components(full_delta)
                    if component[int(seed_y), int(seed_x)]
                ),
                None,
            )
            if delta_component is None:
                raise ValueError("delta proposal seed does not identify a candidate component")
            delta_component = delta_component[region]
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
