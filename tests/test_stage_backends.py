import unittest

import numpy as np

from change_agent.adapters.stage_backends import (
    _normalized_crop_box,
    _stage_images,
    _stage_prompt,
)
from change_agent.state import ChangeState


class NormalizedCropBoxTests(unittest.TestCase):
    def test_tiny_region_is_expanded_for_bailian(self):
        crop = _normalized_crop_box((500, 500, 500, 500), (256, 256))
        self.assertEqual(crop, (128, 128, 139, 139))
        self.assertEqual(crop[2] - crop[0], 11)
        self.assertEqual(crop[3] - crop[1], 11)

    def test_expansion_stays_inside_image_at_edge(self):
        crop = _normalized_crop_box((0, 0, 0, 0), (256, 256))
        self.assertEqual(crop, (0, 0, 11, 11))
        self.assertGreaterEqual(crop[0], 0)
        self.assertGreaterEqual(crop[1], 0)
        self.assertLessEqual(crop[2], 256)
        self.assertLessEqual(crop[3], 256)

    def test_existing_large_region_is_unchanged(self):
        crop = _normalized_crop_box((100, 200, 800, 900), (256, 256))
        self.assertEqual(crop, (26, 51, 205, 231))

    def test_proposal_mode_sends_only_local_crops_to_diagnosis(self):
        state = ChangeState(
            np.zeros((32, 32, 3), dtype=np.uint8),
            np.zeros((32, 32, 3), dtype=np.uint8),
            "building",
            np.zeros((32, 32), dtype=bool),
            np.zeros((32, 32), dtype=bool),
            np.zeros((32, 32), dtype=bool),
        )
        payload = {
            "visual_context": "proposal",
            "region": {"box_normalized_1000": [250, 250, 750, 750]},
        }

        images = _stage_images("diagnosis", state, None, payload)

        self.assertEqual(len(images), 5)
        self.assertTrue(all("proposal" in label for label, _ in images))

    def test_hybrid_mode_sends_full_context_and_local_crops(self):
        state = ChangeState(
            np.zeros((32, 32, 3), dtype=np.uint8),
            np.zeros((32, 32, 3), dtype=np.uint8),
            "building",
            np.zeros((32, 32), dtype=bool),
            np.zeros((32, 32), dtype=bool),
            np.zeros((32, 32), dtype=bool),
        )
        payload = {
            "visual_context": "hybrid",
            "region": {"box_normalized_1000": [250, 250, 750, 750]},
        }

        images = _stage_images("diagnosis", state, None, payload)

        self.assertEqual(len(images), 5)
        self.assertEqual(images[0][0], "Exact T1 proposal crop")
        self.assertEqual(images[-1][0], "Exact proposal change-mask crop")

    def test_select_stage_renders_one_numbered_global_overview(self):
        state = ChangeState(
            np.zeros((32, 32, 3), dtype=np.uint8),
            np.zeros((32, 32, 3), dtype=np.uint8),
            "building",
            np.zeros((32, 32), dtype=bool),
            np.pad(np.ones((6, 6), dtype=bool), 13),
            np.pad(np.ones((6, 6), dtype=bool), 13),
        )
        payload = {
            "proposal_catalog": [
                {
                    "region_id": "r0",
                    "audit_kind": "present",
                    "component_seed_normalized_1000": [500, 500],
                    "box_normalized_1000": [400, 400, 600, 600],
                }
            ]
        }

        images = _stage_images("select", state, None, payload)

        self.assertEqual(len(images), 1)
        self.assertIn("Numbered global proposal overview", images[0][0])
        self.assertEqual(images[0][1].size, (96, 32))

    def test_local_stage_keeps_overview_then_zoomed_crops(self):
        state = ChangeState(
            np.zeros((32, 32, 3), dtype=np.uint8),
            np.zeros((32, 32, 3), dtype=np.uint8),
            "building",
            np.zeros((32, 32), dtype=bool),
            np.zeros((32, 32), dtype=bool),
            np.zeros((32, 32), dtype=bool),
        )
        region = {
            "region_id": "r0",
            "audit_kind": "present",
            "component_seed_normalized_1000": [500, 500],
            "box_normalized_1000": [250, 250, 750, 750],
        }
        images = _stage_images(
            "diagnosis",
            state,
            None,
            {
                "visual_context": "proposal",
                "region": region,
                "proposal_catalog": [region],
            },
        )

        self.assertEqual(len(images), 6)
        self.assertIn("Global numbered overview", images[0][0])
        self.assertEqual(images[-1][0], "Exact proposal change-mask crop")

    def test_direct_replan_prompt_forbids_repeating_rejected_action(self):
        prompt = _stage_prompt(
            "direct",
            {
                "mode": "replan",
                "rejected_action": {
                    "target_view": "t2",
                    "action": "positive_point",
                    "coordinate": [217, 166],
                },
                "rejection_reasons": ["locality_outside_roi_exceeded"],
                "rejection_history": [
                    {"action": {"action": "positive_point"}, "step_index": 1}
                ],
            },
        )

        self.assertIn("rollback replan", prompt)
        self.assertIn("Do not repeat the rejected action or geometry", prompt)
        self.assertIn("rejection_history", prompt)
        self.assertIn("comparison must be uncertain", prompt)


if __name__ == "__main__":
    unittest.main()
