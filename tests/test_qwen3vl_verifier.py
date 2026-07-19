import json
import unittest

import numpy as np

from change_agent.adapters.qwen3vl_verifier import Qwen3VLZeroShotVerifier
from change_agent.state import AgentAction, ChangeState


class FakeInputs(dict):
    def to(self, device):
        return self


class FakeProcessor:
    def __init__(self, payload):
        self.payloads = payload if isinstance(payload, list) else [payload]
        self.messages = None
        self.messages_history = []
        self.call_count = 0

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
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


class QwenVerifierTest(unittest.TestCase):
    def test_zero_shot_verifier_returns_normalized_structured_feedback(self):
        processor = FakeProcessor(
            [
                {
                    "quality_score": 0.4,
                    "progress_score": 0.12,
                    "error_type": "false_positive_change",
                    "feedback": "Remove unsupported changed-building regions.",
                },
                {"target_view": "t2", "error_region": [100, 200, 800, 900]},
            ]
        )
        verifier = Qwen3VLZeroShotVerifier(model=FakeModel(), processor=processor)
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        mask = np.zeros((16, 16), dtype=bool)
        mask[4:8, 4:8] = True
        state = ChangeState(image, image, "building", mask, mask, mask)
        previous_mask = np.zeros_like(mask)
        previous = ChangeState(image, image, "building", previous_mask, mask, mask)
        action = AgentAction("t2", "positive_point", coordinate=(8, 5))
        output = verifier.verify(state, 0.5, action, previous)
        self.assertEqual(output.error_region, (100, 200, 800, 900))
        self.assertAlmostEqual(output.progress_score, 0.12)
        self.assertAlmostEqual(output.score_delta, -0.1)
        self.assertEqual(output.to_dict()["coordinate_space"], "normalized_0_1000")
        self.assertTrue(output.verifier_valid)
        self.assertTrue(output.localization_valid)
        self.assertEqual(output.suggested_action, "negative_point")
        self.assertFalse(output.stop)
        self.assertEqual(verifier.last_evidence["type"], "qwen3vl_zero_shot")
        texts = [
            item["text"]
            for item in processor.messages_history[0][0]["content"]
            if item["type"] == "text"
        ]
        self.assertEqual(
            texts[:8],
            [
                "Fixed T1 original image:",
                "Fixed T2 original image:",
                "Previous valid T1 object mask:",
                "Previous valid T2 object mask:",
                "Previous valid change mask:",
                "Candidate T1 object mask:",
                "Candidate T2 object mask:",
                "Candidate final change mask (primary evaluation target):",
            ],
        )
        self.assertIn("ground-truth-free verifier", texts[-1])
        self.assertIn("added building", texts[-1])
        self.assertIn("unchanged background", texts[-1])
        self.assertIn("empty T1 or T2 mask does not automatically", texts[-1])
        self.assertIn("Judge the final candidate change mask first", texts[-1])
        self.assertIn("progress_score", texts[-1])
        self.assertIn('"coordinate": [533, 333]', texts[-1])

    def test_invalid_outputs_use_auditable_safe_fallback(self):
        processor = FakeProcessor({"quality_score": 0.5})
        verifier = Qwen3VLZeroShotVerifier(model=FakeModel(), processor=processor, max_retries=2)
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        mask = np.zeros((16, 16), dtype=bool)
        state = ChangeState(image, image, "building", mask, mask, mask)
        output = verifier.verify(state, 0.4, None)
        self.assertEqual(output.quality_score, 0.4)
        self.assertEqual(output.error_type, "uncertain_region")
        self.assertFalse(output.accept)
        self.assertFalse(output.stop)
        self.assertIsNone(output.suggested_action)
        self.assertFalse(output.verifier_valid)
        self.assertFalse(output.localization_valid)
        self.assertIn("recheck required", output.feedback)
        self.assertTrue(verifier.last_evidence["fallback"])

    def test_missing_region_runs_a_second_localization_request(self):
        processor = FakeProcessor(
            [
                {
                    "quality_score": 0.95,
                    "progress_score": 0.0,
                    "error_type": "false_negative",
                    "feedback": "A building is missing from the change mask.",
                },
                {"target_view": "t1", "error_region": [100, 200, 800, 900]},
            ]
        )
        verifier = Qwen3VLZeroShotVerifier(
            model=FakeModel(), processor=processor, max_retries=1
        )
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        mask = np.zeros((16, 16), dtype=bool)
        state = ChangeState(image, image, "building", mask, mask, mask)
        output = verifier.verify(state, None, None)

        self.assertEqual(processor.call_count, 2)
        self.assertEqual(output.error_region, (100, 200, 800, 900))
        self.assertEqual(output.error_type, "false_negative")
        self.assertEqual(output.suggested_action, "positive_point")
        self.assertFalse(output.accept)
        self.assertFalse(output.stop)
        self.assertTrue(output.verifier_valid)
        self.assertTrue(output.localization_valid)
        self.assertIn("only target_view", processor.messages[0]["content"][-1]["text"])

    def test_accept_and_action_are_derived_from_none_diagnosis(self):
        payload = {
            "quality_score": 0.95,
            "progress_score": 0.0,
            "error_type": "none",
            "feedback": "The mask is credible.",
        }
        processor = FakeProcessor(payload)
        verifier = Qwen3VLZeroShotVerifier(model=FakeModel(), processor=processor)
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        mask = np.zeros((16, 16), dtype=bool)
        state = ChangeState(image, image, "building", mask, mask, mask)
        output = verifier.verify(state, None, None)

        self.assertEqual(output.error_region, None)
        self.assertEqual(output.suggested_action, "finish")
        self.assertTrue(output.accept)
        self.assertTrue(output.stop)

    def test_failed_localization_preserves_previous_feedback_without_finish(self):
        processor = FakeProcessor(
            [
                {
                    "quality_score": 0.4,
                    "progress_score": 0.0,
                    "error_type": "false_positive_change",
                    "feedback": "Remove the unsupported changed region.",
                },
                {"target_view": "t2", "error_region": [100, 200, 800, 900]},
                {
                    "quality_score": 0.3,
                    "progress_score": -0.2,
                    "error_type": "false_positive_change",
                    "feedback": "There is still an unsupported region.",
                },
                {"not_error_region": [1, 2, 3, 4]},
            ]
        )
        verifier = Qwen3VLZeroShotVerifier(
            model=FakeModel(), processor=processor, max_retries=1
        )
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        mask = np.zeros((16, 16), dtype=bool)
        state = ChangeState(image, image, "building", mask, mask, mask)
        first = verifier.verify(state, None, None)
        second = verifier.verify(state, first.quality_score, None, state)

        self.assertTrue(first.verifier_valid)
        self.assertFalse(second.verifier_valid)
        self.assertFalse(second.localization_valid)
        self.assertIsNone(second.suggested_action)
        self.assertFalse(second.stop)
        self.assertEqual(second.quality_score, first.quality_score)
        self.assertIn(first.feedback, second.feedback)

    def test_initial_progress_must_be_zero(self):
        processor = FakeProcessor(
            {
                "quality_score": 0.7,
                "progress_score": 0.2,
                "error_type": "none",
                "feedback": "The mask appears credible.",
            }
        )
        verifier = Qwen3VLZeroShotVerifier(
            model=FakeModel(), processor=processor, max_retries=1
        )
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        mask = np.zeros((16, 16), dtype=bool)
        state = ChangeState(image, image, "building", mask, mask, mask)
        output = verifier.verify(state, None, None)

        self.assertFalse(output.verifier_valid)
        self.assertEqual(output.progress_score, 0.0)
        self.assertIn(
            "progress_score must be 0.0",
            " ".join(verifier.last_evidence["validation_errors"]),
        )


if __name__ == "__main__":
    unittest.main()
