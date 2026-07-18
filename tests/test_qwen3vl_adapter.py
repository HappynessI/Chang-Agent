import unittest

import numpy as np

from change_agent.adapters.qwen3vl_adapter import GroundingModelQwen3VL
from change_agent.state import AgentObservation


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
        return ['{"target_view":"t2","action":"positive_point","coordinate":[500,500],"coordinate_frame":"normalized_1000_xy"}']


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
        self.assertEqual(action.coordinate, (10, 5))
        self.assertIn("positive_point", raw)


if __name__ == "__main__":
    unittest.main()
