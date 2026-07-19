import json
import unittest

import numpy as np

from change_agent.adapters.qwen3vl_verifier import Qwen3VLZeroShotVerifier
from change_agent.state import AgentAction, ChangeState
from change_agent.verifier_regions import attach_verifier_regions


class FakeInputs(dict):
    def to(self, device):
        return self


class FakeProcessor:
    def __init__(self, payloads):
        self.payloads = payloads if isinstance(payloads, list) else [payloads]
        self.messages_history = []
        self.call_count = 0

    def apply_chat_template(self, messages, **kwargs):
        self.messages_history.append(messages)
        return FakeInputs(input_ids=np.zeros((1, 3), dtype=np.int64))

    def batch_decode(self, generated, **kwargs):
        payload = self.payloads[min(self.call_count, len(self.payloads) - 1)]
        self.call_count += 1
        return [json.dumps(payload)]


class FakeModel:
    device = "cpu"

    def generate(self, **kwargs):
        return np.zeros((1, 5), dtype=np.int64)


def make_state(end=8):
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[4:end, 4:end] = 180
    t1 = np.zeros((16, 16), dtype=bool)
    t2 = np.zeros_like(t1)
    t2[4:end, 4:end] = True
    return ChangeState(image, image, "building", t1, t2, t2)


def region_payload(state, t1_state="background", t2_state="building"):
    return {
        item["region_id"]: [t1_state, t2_state]
        for item in state.evidence["verifier_region_proposals"]
    }


def effect_payload(state, effect, proposals=None):
    proposals = proposals or state.evidence["verifier_region_proposals"]
    return {
        item["region_id"]: effect
        for item in proposals
    }


def temporal_payload(
    state, t1_state="background", t2_state="building", proposals=None
):
    proposals = proposals or state.evidence["verifier_region_proposals"]
    return {
        item["region_id"]: [t1_state, t2_state]
        for item in proposals
    }


class QwenVerifierTest(unittest.TestCase):
    def test_initial_analysis_uses_regions_without_predicting_scores(self):
        state = make_state()
        attach_verifier_regions(state)
        processor = FakeProcessor(region_payload(state))
        verifier = Qwen3VLZeroShotVerifier(model=FakeModel(), processor=processor)

        output = verifier.verify(state, None, None)

        self.assertTrue(output.verifier_valid)
        self.assertEqual(output.comparison, "initial")
        self.assertIsNone(output.quality_score)
        self.assertIsNone(output.progress_score)
        self.assertEqual(output.suggested_action, "finish")
        self.assertTrue(output.stop)
        self.assertIn("white pixels", output.feedback)
        self.assertEqual(
            verifier.last_evidence["decision_mode"],
            "rgb_temporal_state_then_programmatic_initial",
        )

    def test_false_positive_uses_environment_box_instead_of_model_localization(self):
        state = make_state()
        proposals = attach_verifier_regions(state)
        processor = FakeProcessor(region_payload(state, "background", "background"))
        verifier = Qwen3VLZeroShotVerifier(model=FakeModel(), processor=processor)

        output = verifier.verify(state, None, None)

        self.assertEqual(output.error_type, "false_positive_change")
        self.assertEqual(output.suggested_action, "negative_point")
        self.assertEqual(output.target_view, "t2")
        seed = tuple(proposals[0]["component_seed_normalized"])
        self.assertEqual(output.error_region, (*seed, *seed))
        self.assertEqual(processor.call_count, 1)

    def test_initial_action_prioritizes_smallest_actionable_component(self):
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        t1 = np.zeros((32, 32), dtype=bool)
        t2 = np.zeros_like(t1)
        t2[2:12, 2:12] = True
        t2[24:27, 24:27] = True
        state = ChangeState(image, image, "building", t1, t2, t2)
        proposals = attach_verifier_regions(state)
        verifier = Qwen3VLZeroShotVerifier(
            model=FakeModel(),
            processor=FakeProcessor(region_payload(state, "background", "background")),
        )

        output = verifier.verify(state, None, None)

        smallest = min(proposals, key=lambda item: item["component_area"])
        seed = tuple(smallest["component_seed_normalized"])
        self.assertEqual(output.error_region, (*seed, *seed))
        self.assertEqual(smallest["component_area"], 9)

    def test_unchanged_building_component_adds_missing_temporal_mask(self):
        state = make_state()
        proposals = attach_verifier_regions(state)
        verifier = Qwen3VLZeroShotVerifier(
            model=FakeModel(),
            processor=FakeProcessor(region_payload(state, "building", "building")),
        )

        output = verifier.verify(state, None, None)

        self.assertEqual(output.error_type, "false_positive_change")
        self.assertEqual(output.suggested_action, "positive_point")
        self.assertEqual(output.target_view, "t1")
        seed = tuple(proposals[0]["component_seed_normalized"])
        self.assertEqual(output.error_region, (*seed, *seed))

    def test_initial_model_outputs_only_temporal_states_without_target_view(self):
        state = make_state()
        attach_verifier_regions(state)
        processor = FakeProcessor(region_payload(state))
        verifier = Qwen3VLZeroShotVerifier(
            model=FakeModel(), processor=processor, max_retries=2
        )

        output = verifier.verify(state, None, None)

        self.assertTrue(output.verifier_valid)
        self.assertEqual(output.comparison, "initial")
        self.assertEqual(processor.call_count, 1)
        self.assertEqual(verifier.last_evidence["validation_errors"], [])
        self.assertTrue(
            all(
                item["target_view"] is None
                for item in verifier.last_evidence["region_judgments"]
            )
        )
        self.assertEqual(len(verifier.last_evidence["region_judgments"]), len(
                state.evidence["verifier_region_proposals"]
        ))
        prompt = " ".join(
            item["text"]
            for item in processor.messages_history[0][0]["content"]
            if item["type"] == "text"
        )
        self.assertNotIn("target_view", json.dumps(processor.payloads[0]))
        self.assertIn("Predicted masks", prompt)
        self.assertIn("intentionally hidden", prompt)

    def test_initial_components_are_verified_in_separate_full_coverage_batches(self):
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        t1 = np.zeros((32, 32), dtype=bool)
        t2 = np.zeros_like(t1)
        t2[2:5, 2:5] = True
        t2[25:28, 25:28] = True
        state = ChangeState(image, image, "building", t1, t2, t2)
        attach_verifier_regions(state, max_regions=1, padding_ratio=0.0)
        proposals = state.evidence["verifier_region_proposals"]
        verifier = Qwen3VLZeroShotVerifier(
            model=FakeModel(),
            processor=FakeProcessor(
                [
                    {item["region_id"]: ["background", "building"]}
                    for item in proposals
                ]
            ),
        )

        output = verifier.verify(state, None, None)

        self.assertEqual(output.error_type, "none")
        self.assertTrue(output.accept)
        self.assertTrue(output.stop)
        self.assertEqual(verifier.processor.call_count, 2)
        self.assertEqual(
            state.evidence["verifier_mask_facts"]["initial_audit_coverage_ratio"],
            1.0,
        )

    def test_truncated_top_level_json_is_not_misread_as_nested_region(self):
        with self.assertRaisesRegex(ValueError, "incomplete JSON object"):
            Qwen3VLZeroShotVerifier._extract_json_object(
                '```json {"r0": {"verdict": "true_change"}'
            )

    def test_candidate_effect_labels_programmatically_derive_comparison(self):
        previous = make_state(7)
        state = make_state(8)
        attach_verifier_regions(state, previous)
        processor = FakeProcessor(
            [effect_payload(state, "added_true_change"), temporal_payload(state)]
        )
        verifier = Qwen3VLZeroShotVerifier(model=FakeModel(), processor=processor)

        output = verifier.verify(
            state,
            0.9,
            AgentAction("t2", "positive_point", coordinate=(7, 7)),
            previous,
        )

        self.assertEqual(output.comparison, "better")
        self.assertIsNone(output.quality_score)
        self.assertIsNone(output.progress_score)
        effect_text = " ".join(
            item["text"]
            for item in processor.messages_history[0][0]["content"]
            if item["type"] == "text"
        )
        self.assertIn("judge only the pixels changed", effect_text)
        self.assertIn("Do not output feedback sentences", effect_text)
        effect_panels = [
            item["image"]
            for item in processor.messages_history[0][0]["content"]
            if item["type"] == "image" and item["image"].size == (384, 384)
        ]
        self.assertTrue(effect_panels)
        panel = np.asarray(effect_panels[-1])
        self.assertTrue(np.any(panel[..., 0] > 0))
        self.assertTrue(np.any(panel[..., 1] > 0))
        self.assertTrue(np.any(panel[..., 2] > 0))
        self.assertEqual(processor.call_count, 2)
        temporal_state_text = " ".join(
            item["text"]
            for item in processor.messages_history[1][0]["content"]
            if item["type"] == "text"
        )
        self.assertIn("elementary RGB facts", temporal_state_text)
        self.assertNotIn("Previous accepted final change mask", temporal_state_text)
        self.assertNotIn("t1_mask_pixels", temporal_state_text)
        self.assertNotIn("temporal_difference_pixels", temporal_state_text)
        self.assertNotIn("effect_kind", temporal_state_text)
        self.assertNotIn("positive_point", temporal_state_text)
        self.assertNotIn("candidate_added_pixels", temporal_state_text)
        self.assertTrue(
            all(
                item["decision_source"].startswith("rgb_temporal_state")
                for item in verifier.last_evidence["effect_fusion"]
            )
        )

    def test_unsupported_added_delta_is_programmatically_worse(self):
        previous = make_state(7)
        state = make_state(8)
        attach_verifier_regions(state, previous)
        verifier = Qwen3VLZeroShotVerifier(
            model=FakeModel(),
            processor=FakeProcessor(
                [
                    effect_payload(state, "added_true_change"),
                    temporal_payload(state, "background", "background"),
                ]
            ),
        )

        output = verifier.verify(
            state,
            None,
            AgentAction("t2", "positive_point", coordinate=(7, 7)),
            previous,
        )

        self.assertEqual(output.comparison, "uncertain")
        self.assertEqual(output.error_type, "uncertain_region")
        self.assertEqual(output.suggested_action, "box")

    def test_four_delta_components_are_fully_verified_in_two_batches(self):
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        previous_mask = np.zeros((32, 32), dtype=bool)
        previous_mask[28, 28] = True
        current_mask = np.zeros_like(previous_mask)
        for y, x in ((2, 2), (10, 10), (20, 20)):
            current_mask[y, x] = True
        previous = ChangeState(
            image,
            image,
            "building",
            np.zeros_like(previous_mask),
            previous_mask,
            previous_mask,
        )
        state = ChangeState(
            image,
            image,
            "building",
            np.zeros_like(current_mask),
            current_mask,
            current_mask,
        )
        proposals = attach_verifier_regions(
            state, previous, max_delta_regions=3
        )
        batches = [proposals[:3], proposals[3:]]
        payloads = []
        for batch in batches:
            payloads.append(
                {
                    item["region_id"]: (
                        "added_true_change"
                        if item["effect_kind"] == "added"
                        else "removed_false_positive"
                    )
                    for item in batch
                }
            )
            payloads.append(
                {
                    item["region_id"]: (
                        ["background", "building"]
                        if item["effect_kind"] == "added"
                        else ["background", "background"]
                    )
                    for item in batch
                }
            )
        processor = FakeProcessor(payloads)
        verifier = Qwen3VLZeroShotVerifier(model=FakeModel(), processor=processor)

        output = verifier.verify(
            state,
            None,
            AgentAction("t2", "box", box=(0, 0, 31, 31)),
            previous,
        )

        self.assertEqual(len(proposals), 4)
        self.assertEqual(
            state.evidence["verifier_mask_facts"]["candidate_delta_coverage_ratio"],
            1.0,
        )
        self.assertEqual(output.comparison, "better")
        self.assertTrue(output.accept)
        self.assertEqual(processor.call_count, 4)
        self.assertEqual(
            [item["batch_index"] for item in verifier.last_evidence["effect_fusion"]],
            [0, 0, 0, 1],
        )

    def test_identical_candidate_fingerprint_reuses_cached_effect_decision(self):
        previous = make_state(7)
        state = make_state(8)
        attach_verifier_regions(state, previous)
        processor = FakeProcessor(
            [
                effect_payload(state, "added_false_change"),
                temporal_payload(state, "background", "background"),
            ]
        )
        verifier = Qwen3VLZeroShotVerifier(model=FakeModel(), processor=processor)
        action = AgentAction("t2", "positive_point", coordinate=(7, 7))

        first = verifier.verify(state, None, action, previous)
        second = verifier.verify(state, None, action, previous)

        self.assertEqual(first, second)
        self.assertEqual(processor.call_count, 2)
        self.assertTrue(verifier.last_evidence["cache_hit"])
        self.assertEqual(
            verifier.last_evidence["decision_key"],
            verifier.last_evidence["candidate_fingerprint"],
        )
        self.assertEqual(verifier.last_evidence["reused_from_step"], state.step_index)

    def test_rgb_temporal_facts_keep_beneficial_removal_despite_mask_disagreement(self):
        previous = make_state(8)
        state = make_state(7)
        attach_verifier_regions(state, previous)
        processor = FakeProcessor(
            [
                effect_payload(state, "removed_true_change"),
                temporal_payload(state, "building", "building"),
            ]
        )
        verifier = Qwen3VLZeroShotVerifier(
            model=FakeModel(),
            processor=processor,
            max_delta_component_ratio_without_consensus=1.0,
        )

        output = verifier.verify(
            state,
            None,
            AgentAction("t2", "negative_point", coordinate=(7, 7)),
            previous,
        )

        self.assertEqual(output.comparison, "better")
        self.assertTrue(output.accept)
        self.assertEqual(
            {item["effect"] for item in verifier.last_evidence["effect_judgments"]},
            {"removed_false_positive"},
        )
        self.assertTrue(
            all(
                not item["mask_context_agreement"]
                for item in verifier.last_evidence["effect_fusion"]
            )
        )

    def test_invalid_advisory_mask_response_cannot_filter_beneficial_rgb_edit(self):
        previous = make_state(8)
        state = make_state(7)
        attach_verifier_regions(state, previous)
        processor = FakeProcessor(
            [
                {"wrong": "removed_false_positive"},
                {"wrong": "removed_false_positive"},
                temporal_payload(state, "building", "building"),
            ]
        )
        verifier = Qwen3VLZeroShotVerifier(
            model=FakeModel(),
            processor=processor,
            max_retries=2,
            max_delta_component_ratio_without_consensus=1.0,
        )

        output = verifier.verify(
            state,
            None,
            AgentAction("t2", "negative_point", coordinate=(7, 7)),
            previous,
        )

        self.assertEqual(output.comparison, "better")
        self.assertTrue(output.accept)
        self.assertEqual(processor.call_count, 3)
        self.assertTrue(
            all(
                item["mask_context_effect"] is None
                and item["mask_context_agreement"] is None
                for item in verifier.last_evidence["effect_fusion"]
            )
        )
        self.assertTrue(
            any(
                error.startswith("mask_context batch 0:")
                for error in verifier.last_evidence["validation_errors"]
            )
        )

    def test_large_delta_requires_mask_and_rgb_effect_consensus(self):
        previous = make_state(8)
        state = make_state(7)
        attach_verifier_regions(state, previous)
        verifier = Qwen3VLZeroShotVerifier(
            model=FakeModel(),
            processor=FakeProcessor(
                [
                    effect_payload(state, "removed_true_change"),
                    temporal_payload(state, "building", "building"),
                ]
            ),
            max_delta_component_ratio_without_consensus=0.05,
        )

        output = verifier.verify(
            state,
            None,
            AgentAction("t2", "negative_point", coordinate=(7, 7)),
            previous,
        )

        self.assertEqual(output.comparison, "uncertain")
        self.assertFalse(output.accept)
        self.assertEqual(
            verifier.last_evidence["effect_judgments"][0]["effect"], "uncertain"
        )
        fusion = verifier.last_evidence["effect_fusion"][0]
        self.assertTrue(fusion["consensus_required"])
        self.assertEqual(fusion["final_effect"], "uncertain")

    def test_small_beneficial_delta_is_not_filtered_by_advisory_disagreement(self):
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        previous_mask = np.zeros((32, 32), dtype=bool)
        previous_mask[2:22, 2:22] = True
        previous_mask[28:30, 28:30] = True
        current_mask = previous_mask.copy()
        current_mask[28:30, 28:30] = False
        previous = ChangeState(
            image, image, "building", np.zeros_like(previous_mask), previous_mask,
            previous_mask,
        )
        state = ChangeState(
            image, image, "building", np.zeros_like(current_mask), current_mask,
            current_mask,
        )
        attach_verifier_regions(state, previous)
        verifier = Qwen3VLZeroShotVerifier(
            model=FakeModel(),
            processor=FakeProcessor(
                [
                    effect_payload(state, "removed_true_change"),
                    temporal_payload(state, "building", "building"),
                ]
            ),
            max_delta_component_ratio_without_consensus=0.05,
        )

        output = verifier.verify(
            state,
            None,
            AgentAction("t2", "negative_point", coordinate=(28, 28)),
            previous,
        )

        self.assertEqual(output.comparison, "better")
        self.assertTrue(output.accept)
        fusion = verifier.last_evidence["effect_fusion"][0]
        self.assertFalse(fusion["consensus_required"])
        self.assertFalse(fusion["mask_context_agreement"])

    def test_uncertain_rgb_temporal_state_is_rejected(self):
        previous = make_state(7)
        state = make_state(8)
        attach_verifier_regions(state, previous)
        verifier = Qwen3VLZeroShotVerifier(
            model=FakeModel(),
            processor=FakeProcessor(
                [
                    effect_payload(state, "added_true_change"),
                    temporal_payload(state, "uncertain", "building"),
                ]
            ),
        )

        output = verifier.verify(
            state,
            None,
            AgentAction("t2", "positive_point", coordinate=(7, 7)),
            previous,
        )

        self.assertEqual(output.comparison, "uncertain")
        self.assertFalse(output.accept)

    def test_candidate_decision_key_changes_with_context_and_schema(self):
        previous = make_state(7)
        state = make_state(8)
        attach_verifier_regions(state, previous)
        verifier = Qwen3VLZeroShotVerifier(
            model=FakeModel(), processor=FakeProcessor({})
        )
        proposals = state.evidence["verifier_region_proposals"]
        facts = state.evidence["verifier_mask_facts"]
        action = AgentAction("t2", "positive_point", coordinate=(7, 7))
        baseline = verifier._candidate_fingerprint(
            state, previous, action, proposals, facts
        )

        changed_query = state.clone()
        changed_query.query = "warehouse"
        self.assertNotEqual(
            baseline,
            verifier._candidate_fingerprint(
                changed_query, previous, action, proposals, facts
            ),
        )
        self.assertNotEqual(
            baseline,
            verifier._candidate_fingerprint(
                state,
                previous,
                AgentAction("t2", "positive_point", coordinate=(6, 7)),
                proposals,
                facts,
            ),
        )
        verifier.SCHEMA_VERSION = "compact_delta_effect_next"
        self.assertNotEqual(
            baseline,
            verifier._candidate_fingerprint(
                state, previous, action, proposals, facts
            ),
        )

    def test_mixed_or_conflicting_delta_effects_are_conservatively_uncertain(self):
        judgments = Qwen3VLZeroShotVerifier._comparison_from_effects
        from change_agent.adapters.qwen3vl_verifier import _EffectJudgment

        self.assertEqual(judgments((_EffectJudgment("d0", "mixed"),)), "uncertain")
        self.assertEqual(
            judgments(
                (
                    _EffectJudgment("d0", "added_true_change"),
                    _EffectJudgment("d1", "added_false_change"),
                )
            ),
            "uncertain",
        )

    def test_compact_effect_json_fits_a_small_output_budget(self):
        payload = {f"d{index}": "added_true_change" for index in range(3)}
        self.assertLess(len(json.dumps(payload, separators=(",", ":"))), 128)
        temporal = {f"d{index}": ["background", "building"] for index in range(3)}
        self.assertLess(len(json.dumps(temporal, separators=(",", ":"))), 160)

    def test_present_change_region_cannot_derive_false_negative(self):
        state = make_state()
        attach_verifier_regions(state)
        verifier = Qwen3VLZeroShotVerifier(
            model=FakeModel(),
            processor=FakeProcessor(region_payload(state, "background", "building")),
            max_retries=1,
        )

        output = verifier.verify(state, None, None)

        self.assertTrue(output.verifier_valid)
        self.assertEqual(output.error_type, "none")
        self.assertEqual(
            {item["verdict"] for item in verifier.last_evidence["region_judgments"]},
            {"true_change"},
        )

    def test_false_negative_requires_a_mask_derived_missing_proposal(self):
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        t1 = np.zeros((16, 16), dtype=bool)
        t2 = np.zeros_like(t1)
        t2[4:8, 4:8] = True
        state = ChangeState(image, image, "building", t1, t2, np.zeros_like(t1))
        attach_verifier_regions(state)
        verifier = Qwen3VLZeroShotVerifier(
            model=FakeModel(),
            processor=FakeProcessor(region_payload(state, "background", "building")),
        )

        output = verifier.verify(state, None, None)

        self.assertTrue(output.verifier_valid)
        self.assertEqual(output.error_type, "false_negative")

    def test_abstract_initial_labels_are_rejected_then_temporal_states_corrected(self):
        state = make_state()
        attach_verifier_regions(state)
        abstract_labels = {
            item["region_id"]: ["false_negative", "t1"]
            for item in state.evidence["verifier_region_proposals"]
        }
        processor = FakeProcessor(
            [
                abstract_labels,
                region_payload(state),
            ]
        )
        verifier = Qwen3VLZeroShotVerifier(
            model=FakeModel(), processor=processor, max_retries=2
        )

        output = verifier.verify(state, None, None)

        self.assertTrue(output.verifier_valid)
        self.assertEqual(processor.call_count, 2)
        self.assertIn(
            "unsupported RGB temporal state",
            " ".join(verifier.last_evidence["validation_errors"]),
        )

    def test_initial_prompt_hides_predicted_masks_and_uses_exact_rgb_panels(self):
        state = make_state()
        attach_verifier_regions(state)
        verifier = Qwen3VLZeroShotVerifier(
            model=FakeModel(), processor=FakeProcessor(region_payload(state))
        )

        messages = verifier.build_messages(state, None, None)
        content = messages[0]["content"]
        texts = [item["text"] for item in content if item["type"] == "text"]
        images = [item["image"] for item in content if item["type"] == "image"]

        self.assertEqual(
            texts[:2],
            ["Fixed clean T1 original image:", "Fixed clean T2 original image:"],
        )
        self.assertIn("Initial RGB audit proposal r0", texts[2])
        self.assertEqual(images[-1].size, (384, 384))
        self.assertIn("elementary RGB facts", texts[-1])
        self.assertNotIn("predicted T1 object mask", " ".join(texts).lower())
        self.assertNotIn("change_pixels", " ".join(texts))

    def test_component_outline_preserves_audited_rgb_pixels(self):
        rgb = np.zeros((5, 5, 3), dtype=np.uint8)
        rgb[2, 2] = [40, 80, 120]
        component = np.zeros((5, 5), dtype=bool)
        component[2, 2] = True

        outlined = Qwen3VLZeroShotVerifier._outline_component(rgb, component)

        np.testing.assert_array_equal(outlined[2, 2], rgb[2, 2])
        np.testing.assert_array_equal(outlined[1, 2], [255, 255, 0])
        np.testing.assert_array_equal(outlined[0, 0], [0, 0, 0])

    def test_initial_panel_difference_uses_raw_rgb_not_yellow_outline(self):
        state = make_state()
        proposals = attach_verifier_regions(state)

        panel = np.asarray(
            Qwen3VLZeroShotVerifier._rgb_initial_panel(state, proposals[0])
        )

        self.assertTrue(np.any(np.all(panel[:192, :192] == [255, 255, 0], axis=2)))
        self.assertFalse(np.any(panel[192:, 192:]))

    def test_identical_candidate_must_be_unchanged(self):
        previous = make_state()
        state = previous.clone()
        attach_verifier_regions(state, previous)
        processor = FakeProcessor(region_payload(state))
        verifier = Qwen3VLZeroShotVerifier(
            model=FakeModel(), processor=processor, max_retries=2
        )

        output = verifier.verify(state, None, AgentAction("t2", "finish"), previous)

        self.assertTrue(output.verifier_valid)
        self.assertEqual(output.comparison, "unchanged")
        self.assertEqual(verifier.last_evidence["decision_mode"], "programmatic_identical_state")
        self.assertEqual(processor.call_count, 0)

    def test_unknown_region_id_is_invalid_and_cannot_authorize_finish(self):
        state = make_state()
        attach_verifier_regions(state)
        bad = region_payload(state)
        value = bad.pop(next(iter(bad)))
        bad["invented"] = value
        verifier = Qwen3VLZeroShotVerifier(
            model=FakeModel(), processor=FakeProcessor(bad), max_retries=1
        )

        output = verifier.verify(state, None, None)

        self.assertFalse(output.verifier_valid)
        self.assertIsNone(output.suggested_action)
        self.assertFalse(output.stop)

    def test_all_empty_model_outputs_are_not_treated_as_a_complete_diagnosis(self):
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        mask = np.zeros((16, 16), dtype=bool)
        state = ChangeState(image, image, "building", mask, mask, mask)
        attach_verifier_regions(state)
        verifier = Qwen3VLZeroShotVerifier(
            model=FakeModel(),
            processor=FakeProcessor({"regions": []}),
            max_retries=1,
        )

        output = verifier.verify(state, None, None)

        self.assertFalse(output.verifier_valid)
        self.assertIn(
            "no mask-derived proposal",
            " ".join(verifier.last_evidence["validation_errors"]),
        )


if __name__ == "__main__":
    unittest.main()
