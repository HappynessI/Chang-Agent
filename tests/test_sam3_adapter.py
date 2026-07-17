import unittest

import numpy as np

from change_agent.adapters.sam3_adapter import SAM3ProcessorAdapter


class FakeProcessor:
    def __init__(self):
        self.box = None

    def set_image(self, image):
        return {"shape": (image.height, image.width)}

    def reset_all_prompts(self, state):
        return None

    def set_text_prompt(self, prompt, state):
        height, width = state["shape"]
        logits = np.full((height, width), -4.0, dtype=np.float32)
        logits[2:6, 3:7] = 4.0
        state.update({"semantic_mask_logits": logits, "presence_score": np.array([0.9])})
        return state

    def add_geometric_prompt(self, box, label, state):
        self.box = box
        return state


class SAM3AdapterTest(unittest.TestCase):
    def test_text_and_box_paths_use_public_processor_api(self):
        processor = FakeProcessor()
        adapter = SAM3ProcessorAdapter(processor)
        image = np.zeros((8, 10, 3), dtype=np.uint8)
        t1, t2, evidence = adapter.initialize_masks(image, image, "building")
        self.assertEqual(int(t1.sum()), 16)
        self.assertTrue(np.array_equal(t1, t2))
        self.assertIn("change_confidence", evidence)
        result = adapter.segment_box(image, (0.5, 0.5, 0.2, 0.3), "building")
        self.assertEqual(int(result.sum()), 16)
        self.assertEqual(processor.box, [0.5, 0.5, 0.2, 0.3])


if __name__ == "__main__":
    unittest.main()

