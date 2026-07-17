import unittest

import numpy as np

from change_agent.adapters.omniovcd_adapter import MaskPairProcessor, connected_components
from change_agent.perturbations import make_training_target, perturb_mask


class MaskProcessingTest(unittest.TestCase):
    def test_extracts_diagonal_as_one_eight_connected_instance(self):
        mask = np.eye(4, dtype=bool)
        self.assertEqual(len(connected_components(mask)), 1)

    def test_unmatched_instance_becomes_change(self):
        t1 = np.zeros((8, 8), dtype=bool)
        t2 = np.zeros_like(t1)
        t1[1:3, 1:3] = True
        t2[1:3, 1:3] = True
        t2[5:7, 5:7] = True
        update = MaskPairProcessor().rebuild(t1, t2)
        self.assertEqual(update.matching, ((0, 0),))
        self.assertEqual(int(update.change_mask.sum()), 4)

    def test_training_targets_are_offline_and_correct(self):
        gt = np.zeros((10, 10), dtype=bool)
        gt[3:7, 3:7] = True
        candidate = perturb_mask(gt, "dilate", radius=1)
        target = make_training_target(candidate, gt)
        self.assertLess(target.quality, 1)
        self.assertTrue(target.false_positive_map.any())
        self.assertEqual(target.error_type, "false_positive_change")


if __name__ == "__main__":
    unittest.main()

