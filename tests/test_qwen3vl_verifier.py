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


def region_payload(state, verdict="true_change", feedback="The white region is supported."):
    proposals = state.evidence["verifier_region_proposals"]
    return {
        "regions": [
            {
                "region_id": item["region_id"],
                "verdict": verdict,
                "target_view": "t2" if verdict in {"false_positive", "false_negative"} else None,
                "feedback": feedback,
            }
            for item in proposals
        ]
    }


def keyed_region_payload(
    state, verdict="true_change", feedback="The white region is supported."
):
    """Model-shaped alternative observed in the GPU rollout."""
    return {
        item["region_id"]: {
            "region_id": item["region_id"],
            "verdict": verdict,
            # Qwen often supplies a harmless view even for non-actionable
            # judgments; the verifier should not discard an otherwise valid
            # region analysis for that cosmetic field.
            "target_view": "t1",
            "feedback": feedback,
        }
        for item in state.evidence["verifier_region_proposals"]
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
            "categorical_pairwise_no_absolute_or_progress_score",
        )

    def test_false_positive_uses_environment_box_instead_of_model_localization(self):
        state = make_state()
        proposals = attach_verifier_regions(state)
        processor = FakeProcessor(region_payload(state, "false_positive", "Roof is unchanged."))
        verifier = Qwen3VLZeroShotVerifier(model=FakeModel(), processor=processor)

        output = verifier.verify(state, None, None)

        self.assertEqual(output.error_type, "false_positive_change")
        self.assertEqual(output.suggested_action, "negative_point")
        self.assertEqual(output.target_view, "t2")
        self.assertEqual(output.error_region, tuple(proposals[0]["box_normalized"]))
        self.assertEqual(processor.call_count, 1)

    def test_keyed_region_object_from_qwen_is_normalized(self):
        state = make_state()
        attach_verifier_regions(state)
        processor = FakeProcessor(keyed_region_payload(state))
        verifier = Qwen3VLZeroShotVerifier(model=FakeModel(), processor=processor)

        output = verifier.verify(state, None, None)

        self.assertTrue(output.verifier_valid)
        self.assertEqual(output.comparison, "initial")
        self.assertEqual(len(verifier.last_evidence["region_judgments"]), len(
            state.evidence["verifier_region_proposals"]
        ))

    def test_truncated_top_level_json_is_not_misread_as_nested_region(self):
        with self.assertRaisesRegex(ValueError, "incomplete JSON object"):
            Qwen3VLZeroShotVerifier._extract_json_object(
                '```json {"r0": {"verdict": "true_change"}'
            )

    def test_candidate_pairwise_request_outputs_only_categorical_comparison(self):
        previous = make_state(7)
        state = make_state(8)
        attach_verifier_regions(state, previous)
        processor = FakeProcessor(
            [
                region_payload(state),
                {"comparison": "better", "feedback": "The added edge completes the building."},
            ]
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
        pairwise_text = " ".join(
            item["text"]
            for item in processor.messages_history[1][0]["content"]
            if item["type"] == "text"
        )
        self.assertIn("categorical pairwise gate", pairwise_text)
        self.assertIn("Do not output any quality/progress score", pairwise_text)
        self.assertIn("Previous accepted T1 object mask:", pairwise_text)
        self.assertIn("Candidate T2 object mask:", pairwise_text)
        pairwise_panels = [
            item["image"]
            for item in processor.messages_history[1][0]["content"]
            if item["type"] == "image" and item["image"].size == (384, 384)
        ]
        self.assertTrue(pairwise_panels)
        panel = np.asarray(pairwise_panels[-1])
        self.assertTrue(np.any(panel[..., 0] > 0))
        self.assertTrue(np.any(panel[..., 1] > 0))
        self.assertTrue(np.any(panel[..., 2] > 0))

    def test_nonempty_mask_claimed_empty_is_rejected_and_retried(self):
        state = make_state()
        attach_verifier_regions(state)
        processor = FakeProcessor(
            [
                region_payload(
                    state,
                    "false_negative",
                    "The current change mask is empty and misses a building.",
                ),
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
            "not empty",
            " ".join(verifier.last_evidence["validation_errors"]),
        )

    def test_empty_t1_object_mask_is_not_change_mask_contradiction(self):
        self.assertFalse(
            Qwen3VLZeroShotVerifier._claims_empty(
                "The T1 mask is empty, but the T2 building is visible."
            )
        )
        self.assertTrue(
            Qwen3VLZeroShotVerifier._claims_empty(
                "The current change mask is empty."
            )
        )

    def test_prompt_keeps_full_analysis_and_adds_upscaled_local_panels(self):
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
            texts[:5],
            [
                "Full T1 original image:",
                "Full T2 original image:",
                "Full predicted T1 object mask:",
                "Full predicted T2 object mask:",
                "Full candidate final change mask:",
            ],
        )
        self.assertIn("Local proposal r0", texts[5])
        self.assertEqual(images[-1].size, (384, 384))
        self.assertIn("must not call it empty", texts[-1])

    def test_identical_candidate_must_be_unchanged(self):
        previous = make_state()
        state = previous.clone()
        attach_verifier_regions(state, previous)
        processor = FakeProcessor(
            [
                region_payload(state),
                {"comparison": "better", "feedback": "It is better."},
                {"comparison": "unchanged", "feedback": "The masks are identical."},
            ]
        )
        verifier = Qwen3VLZeroShotVerifier(
            model=FakeModel(), processor=processor, max_retries=2
        )

        output = verifier.verify(state, None, AgentAction("t2", "finish"), previous)

        self.assertTrue(output.verifier_valid)
        self.assertEqual(output.comparison, "unchanged")
        self.assertIn(
            "identical",
            " ".join(verifier.last_evidence["validation_errors"]),
        )

    def test_unknown_region_id_is_invalid_and_cannot_authorize_finish(self):
        state = make_state()
        attach_verifier_regions(state)
        bad = region_payload(state)
        bad["regions"][0]["region_id"] = "invented"
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
