import unittest

import numpy as np

from change_agent.state import ChangeState
from change_agent.verifier_regions import (
    attach_verifier_regions,
    build_candidate_delta_regions,
    build_verifier_regions,
)


class VerifierRegionTest(unittest.TestCase):
    def test_proposals_merge_change_and_temporal_difference_sources(self):
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        t1 = np.zeros((32, 32), dtype=bool)
        t2 = np.zeros_like(t1)
        t2[10:15, 12:18] = True
        state = ChangeState(image, image, "building", t1, t2, t2)

        proposals = attach_verifier_regions(state)

        self.assertEqual(len(proposals), 1)
        self.assertIn("change_component", proposals[0]["sources"])
        self.assertIn("temporal_difference", proposals[0]["sources"])
        self.assertEqual(state.evidence["verifier_mask_facts"]["change_pixels"], 30)
        self.assertEqual(proposals[0]["change_pixels"], 30)
        self.assertEqual(
            state.evidence["verifier_mask_facts"]["initial_audit_coverage_ratio"],
            1.0,
        )

    def test_candidate_delta_sources_have_priority_and_region_limit(self):
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        previous_mask = np.zeros((32, 32), dtype=bool)
        previous_mask[2:5, 2:5] = True
        current_mask = previous_mask.copy()
        current_mask[20:25, 20:25] = True
        previous = ChangeState(
            image, image, "building", np.zeros_like(previous_mask), previous_mask, previous_mask
        )
        current = ChangeState(
            image, image, "building", np.zeros_like(current_mask), current_mask, current_mask
        )

        proposals = build_verifier_regions(current, previous, max_regions=1)

        self.assertEqual(len(proposals), 1)
        self.assertIn("candidate_added", proposals[0]["sources"])
        self.assertGreater(proposals[0]["candidate_delta_pixels"], 0)

    def test_single_white_pixel_is_never_lost_by_min_area_filter(self):
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        mask = np.zeros((16, 16), dtype=bool)
        mask[8, 8] = True
        state = ChangeState(image, image, "building", np.zeros_like(mask), mask, mask)

        proposals = build_verifier_regions(state, min_component_area=4)

        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0]["component_area"], 1)
        self.assertEqual(proposals[0]["change_pixels"], 1)

    def test_candidate_delta_regions_keep_added_and_removed_polarity_separate(self):
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        previous_mask = np.zeros((32, 32), dtype=bool)
        previous_mask[2:5, 2:5] = True
        current_mask = np.zeros_like(previous_mask)
        current_mask[20:25, 20:25] = True
        previous = ChangeState(
            image, image, "building", np.zeros_like(previous_mask), previous_mask, previous_mask
        )
        current = ChangeState(
            image, image, "building", np.zeros_like(current_mask), current_mask, current_mask
        )

        proposals = build_candidate_delta_regions(current, previous, max_regions=2)

        self.assertEqual({item["effect_kind"] for item in proposals}, {"added", "removed"})
        self.assertTrue(
            all(len(item["sources"]) == 1 for item in proposals)
        )

    def test_initial_coverage_counts_exact_components_not_padded_boxes(self):
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        t1 = np.zeros((32, 32), dtype=bool)
        t2 = np.zeros_like(t1)
        t2[10:12, 10:12] = True
        t2[10:12, 14:16] = True
        state = ChangeState(image, image, "building", t1, t2, t2)

        proposals = attach_verifier_regions(
            state, max_regions=1, padding_ratio=1.0
        )
        facts = state.evidence["verifier_mask_facts"]

        self.assertEqual(len(proposals), 1)
        self.assertEqual(facts["initial_audit_pixels"], 8)
        self.assertEqual(facts["initial_audit_covered_pixels"], 4)
        self.assertEqual(facts["initial_audit_uncovered_pixels"], 4)
        self.assertEqual(facts["initial_audit_coverage_ratio"], 0.5)

    def test_candidate_attachment_keeps_components_separate_with_full_coverage(self):
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        previous_mask = np.zeros((32, 32), dtype=bool)
        current_mask = np.zeros_like(previous_mask)
        current_mask[2:4, 2:4] = True
        current_mask[10:12, 10:12] = True
        current_mask[20:22, 20:22] = True
        previous = ChangeState(
            image, image, "building", np.zeros_like(previous_mask), previous_mask, previous_mask
        )
        current = ChangeState(
            image, image, "building", np.zeros_like(current_mask), current_mask, current_mask
        )

        proposals = attach_verifier_regions(current, previous, max_regions=6)

        self.assertEqual(len(proposals), 3)
        self.assertTrue(all(item["effect_kind"] == "added" for item in proposals))
        self.assertEqual(sum(item["component_area"] for item in proposals), 12)
        self.assertEqual(
            current.evidence["verifier_mask_facts"]["candidate_delta_uncovered_pixels"],
            0,
        )
        self.assertEqual(
            current.evidence["verifier_mask_facts"]["candidate_delta_coverage_ratio"],
            1.0,
        )

    def test_component_batch_size_preserves_every_delta_component(self):
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        previous_mask = np.zeros((32, 32), dtype=bool)
        current_mask = np.zeros_like(previous_mask)
        for offset in (2, 8, 14, 20):
            current_mask[offset : offset + 2, offset : offset + 2] = True
        previous = ChangeState(
            image, image, "building", np.zeros_like(previous_mask), previous_mask, previous_mask
        )
        current = ChangeState(
            image, image, "building", np.zeros_like(current_mask), current_mask, current_mask
        )

        proposals = attach_verifier_regions(current, previous, max_delta_regions=3)
        facts = current.evidence["verifier_mask_facts"]

        self.assertEqual(len(proposals), 4)
        self.assertEqual([item["batch_index"] for item in proposals], [0, 0, 0, 1])
        self.assertEqual(facts["candidate_delta_covered_pixels"], 16)
        self.assertEqual(facts["candidate_delta_uncovered_pixels"], 0)
        self.assertEqual(facts["candidate_delta_coverage_ratio"], 1.0)
        self.assertEqual(facts["proposal_config"]["max_regions_per_batch"], 3)
        self.assertEqual(facts["proposal_config"]["batch_count"], 2)


if __name__ == "__main__":
    unittest.main()
