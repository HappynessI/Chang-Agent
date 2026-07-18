import unittest

import numpy as np

from change_agent.adapters.qwen3vl_adapter import GroundingModelQwen3VL
from change_agent.state import AgentObservation, VerifierOutput


class FakeInputs(dict):
    def to(self, device):
        return self


class FakeProcessor:
    def __init__(self):
        self.messages = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        return FakeInputs(input_ids=np.zeros((1, 3), dtype=np.int64))

    def batch_decode(self, generated, **kwargs):
        return ['{"target_view":"t2","action":"positive_point","coordinate":[500,500]}']


class FakeModel:
    device = "cpu"

    def generate(self, **kwargs):
        return np.zeros((1, 5), dtype=np.int64)


class QwenAdapterTest(unittest.TestCase):
    def test_modern_messages_label_all_images_and_parse_action(self):
        processor = FakeProcessor()
        adapter = GroundingModelQwen3VL(model=FakeModel(), processor=processor)
        image = np.zeros((11, 21, 3), dtype=np.uint8)
        observation = AgentObservation(image, image, "building", np.zeros((11, 21)))
        raw, action = adapter.act(observation)
        texts = [
            item["text"]
            for item in processor.messages[0]["content"]
            if item["type"] == "text"
        ]
        self.assertIn("T1 image", texts[0])
        self.assertIn("T2 image", texts[1])
        self.assertIn("Current binary change mask", texts[2])
        self.assertNotIn("<img>", " ".join(texts))
        self.assertIn("coordinate protocol is system-defined", texts[-1])
        self.assertNotIn("coordinate_frame", texts[-1])
        self.assertEqual(action.coordinate, (10, 5))
        self.assertIn("positive_point", raw)

    def test_validation_error_is_injected_into_retry_prompt(self):
        processor = FakeProcessor()
        adapter = GroundingModelQwen3VL(model=FakeModel(), processor=processor)
        image = np.zeros((11, 21, 3), dtype=np.uint8)
        observation = AgentObservation(image, image, "building", np.zeros((11, 21)))
        adapter.generate_raw(observation, "finish is forbidden before a tool action")
        texts = [
            item["text"]
            for item in processor.messages[0]["content"]
            if item["type"] == "text"
        ]
        self.assertIn("previous action was rejected", texts[-1])
        self.assertIn("finish is forbidden", texts[-1])

    def test_invalid_verifier_feedback_does_not_authorize_finish(self):
        processor = FakeProcessor()
        adapter = GroundingModelQwen3VL(model=FakeModel(), processor=processor)
        image = np.zeros((11, 21, 3), dtype=np.uint8)
        observation = AgentObservation(
            image,
            image,
            "building",
            np.zeros((11, 21)),
            feedback=VerifierOutput(
                quality_score=0.4,
                error_type="false_positive_change",
                suggested_action=None,
                feedback="recheck required",
                verifier_valid=False,
                localization_valid=False,
            ),
            history_summary="step=1, action=box, score=0.400, error=false_positive_change",
        )
        adapter.generate_raw(observation)
        texts = [
            item["text"]
            for item in processor.messages[0]["content"]
            if item["type"] == "text"
        ]
        self.assertIn("Verifier feedback is invalid", texts[-1])
        self.assertIn("cannot authorize finish", texts[-1])


if __name__ == "__main__":
    unittest.main()
