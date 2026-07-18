import json
import unittest

import numpy as np

from change_agent.adapters.qwen3vl_verifier import Qwen3VLZeroShotVerifier
from change_agent.state import ChangeState


class FakeInputs(dict):
    def to(self, device):
        return self


class FakeProcessor:
    def __init__(self, payload):
        self.payload = payload
        self.messages = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        return FakeInputs(input_ids=np.zeros((1, 3), dtype=np.int64))

    def batch_decode(self, generated, **kwargs):
        return [json.dumps(self.payload)]


class FakeModel:
    device = "cpu"

    def generate(self, **kwargs):
        return np.zeros((1, 5), dtype=np.int64)


class QwenVerifierTest(unittest.TestCase):
    def test_zero_shot_verifier_returns_normalized_structured_feedback(self):
        payload = {
            "quality_score": 0.4,
            "error_type": "false_positive_change",
            "target_view": "t2",
            "error_region": [100, 200, 800, 900],
            "suggested_action": "negative_point",
            "feedback": "Remove unsupported changed-building regions.",
            "accept": False,
        }
        processor = FakeProcessor(payload)
        verifier = Qwen3VLZeroShotVerifier(model=FakeModel(), processor=processor)
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        mask = np.zeros((16, 16), dtype=bool)
        mask[4:8, 4:8] = True
        state = ChangeState(image, image, "building", mask, mask, mask)
        output = verifier.verify(state, 0.5, None)
        self.assertEqual(output.error_region, (100, 200, 800, 900))
        self.assertAlmostEqual(output.score_delta, -0.1)
        self.assertEqual(output.to_dict()["coordinate_space"], "normalized_0_1000")
        self.assertEqual(verifier.last_evidence["type"], "qwen3vl_zero_shot")
        texts = [
            item["text"]
            for item in processor.messages[0]["content"]
            if item["type"] == "text"
        ]
        self.assertIn("ground-truth-free verifier", texts[-1])
        self.assertIn("do not alternate views by rule", texts[-1])


if __name__ == "__main__":
    unittest.main()
