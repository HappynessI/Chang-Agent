import unittest

import numpy as np

from change_agent.executor import ActionExecutor
from change_agent.state import AgentAction


class PointBackend:
    def __init__(self, output):
        self.output = output

    def refine(self, image, initial_mask, coordinate, is_positive, click_history=()):
        return self.output.copy()


class BoxBackend:
    def __init__(self, output):
        self.output = output

    def segment_box(self, image, box_cxcywh_normalized, query):
        return self.output.copy()


class ExecutorLocalityTest(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((16, 16, 3), dtype=np.uint8)

    def test_positive_point_merges_only_clicked_prediction_component(self):
        initial = np.zeros((16, 16), dtype=bool)
        prediction = np.zeros_like(initial)
        prediction[5:7, 5:7] = True
        prediction[12:14, 12:14] = True
        executor = ActionExecutor(PointBackend(prediction), BoxBackend(prediction))

        result = executor.execute(
            AgentAction("t2", "positive_point", coordinate=(5, 5)),
            self.image,
            initial,
            "building",
        )

        self.assertTrue(result.mask[5:7, 5:7].all())
        self.assertFalse(result.mask[12:14, 12:14].any())
        self.assertEqual(
            result.evidence["locality"]["composition_mode"],
            "merge_clicked_prediction_component",
        )

    def test_negative_point_removes_only_clicked_initial_component(self):
        initial = np.zeros((16, 16), dtype=bool)
        initial[3:5, 3:5] = True
        initial[10:12, 10:12] = True
        executor = ActionExecutor(PointBackend(initial), BoxBackend(initial))

        result = executor.execute(
            AgentAction("t1", "negative_point", coordinate=(3, 3)),
            self.image,
            initial,
            "building",
        )

        self.assertFalse(result.mask[3:5, 3:5].any())
        self.assertTrue(result.mask[10:12, 10:12].all())
        self.assertEqual(result.evidence["locality"]["component_count_delta"], -1)

    def test_box_replaces_only_pixels_inside_box(self):
        initial = np.zeros((16, 16), dtype=bool)
        initial[1:3, 1:3] = True
        initial[5:9, 5:9] = True
        prediction = np.zeros_like(initial)
        executor = ActionExecutor(PointBackend(prediction), BoxBackend(prediction))

        result = executor.execute(
            AgentAction("t2", "box", box=(5, 5, 8, 8)),
            self.image,
            initial,
            "building",
        )

        self.assertTrue(result.mask[1:3, 1:3].all())
        self.assertFalse(result.mask[5:9, 5:9].any())
        self.assertEqual(result.evidence["locality"]["outside_roi_pixels"], 0)
        self.assertEqual(
            result.evidence["locality"]["composition_mode"], "replace_box_roi_only"
        )


if __name__ == "__main__":
    unittest.main()
