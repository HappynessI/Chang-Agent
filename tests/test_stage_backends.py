import unittest

import numpy as np

from change_agent.adapters.stage_backends import (
    _delta_only_contour,
    _normalized_crop_box,
    _stage_images,
    _stage_prompt,
)
from change_agent.state import ChangeState
from change_agent.verifier_protocol import AuditChecklist


class NormalizedCropBoxTests(unittest.TestCase):
    def test_delta_only_contour_preserves_delta_rgb_and_blacks_padding(self):
        image = np.zeros((5, 5, 3), dtype=np.uint8)
        image[2, 2] = (12, 34, 56)
        mask = np.zeros((5, 5), dtype=bool)
        mask[2, 2] = True

        rendered = np.asarray(_delta_only_contour(image, mask))

        self.assertEqual(tuple(rendered[2, 2]), (12, 34, 56))
        self.assertEqual(tuple(rendered[0, 0]), (0, 0, 0))
        self.assertEqual(tuple(rendered[1, 1]), (0, 255, 255))

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

    def test_proposal_mode_sends_marked_overview_and_local_crops(self):
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
            "region": {
                "region_id": "r0",
                "audit_kind": "present",
                "component_seed_normalized_1000": [500, 500],
                "box_normalized_1000": [250, 250, 750, 750],
            },
        }
        payload["proposal_catalog"] = [payload["region"]]

        images = _stage_images("diagnosis", state, None, payload)

        self.assertEqual(len(images), 6)
        self.assertIn("Active-region marked global overview", images[0][0])
        self.assertTrue(all("proposal" in label for label, _ in images[1:]))

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
            "region": {
                "region_id": "r0",
                "audit_kind": "present",
                "component_seed_normalized_1000": [500, 500],
                "box_normalized_1000": [250, 250, 750, 750],
            },
        }
        payload["proposal_catalog"] = [payload["region"]]

        images = _stage_images("diagnosis", state, None, payload)

        self.assertEqual(len(images), 6)
        self.assertIn("Active-region marked global overview", images[0][0])
        self.assertEqual(images[1][0], "Exact T1 proposal crop")
        self.assertEqual(images[-1][0], "Exact proposal change-mask crop")
        self.assertEqual(images[0][1].size, (160, 32))
        self.assertEqual(images[1][1].size, (16, 16))

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

    def test_proposal_local_stage_keeps_active_marked_overview(self):
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
        self.assertIn("yellow is r0", images[0][0])
        self.assertEqual(images[1][0], "Exact T1 proposal crop")
        self.assertEqual(images[-1][0], "Exact proposal change-mask crop")

    def test_hybrid_local_stage_uses_full_state_even_with_catalog(self):
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
                "visual_context": "hybrid",
                "region": region,
                "proposal_catalog": [region],
            },
        )

        self.assertEqual(len(images), 6)
        self.assertIn("Active-region marked global overview", images[0][0])
        self.assertEqual(images[0][1].size, (160, 32))
        self.assertEqual(images[-1][0], "Exact proposal change-mask crop")

    def test_evidence_prompt_describes_the_selected_visual_scope(self):
        proposal_prompt = _stage_prompt(
            "evidence", {"visual_context": "proposal"}
        )
        hybrid_prompt = _stage_prompt("evidence", {"visual_context": "hybrid"})

        self.assertIn("active-region marked RGB/change overview", proposal_prompt)
        self.assertIn("active-region marked full-frame state", hybrid_prompt)

    def test_candidate_region_includes_previous_and_candidate_object_mask_crops(self):
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        previous_mask = np.zeros((32, 32), dtype=bool)
        previous_mask[8:12, 8:12] = True
        candidate_mask = np.zeros_like(previous_mask)
        previous = ChangeState(
            image, image, "building", np.zeros_like(previous_mask), previous_mask, previous_mask
        )
        candidate = ChangeState(
            image, image, "building", np.zeros_like(candidate_mask), candidate_mask, candidate_mask
        )
        region = {
            "region_id": "d0",
            "audit_kind": "delta_removed",
            "component_seed_normalized_1000": [300, 300],
            "box_normalized_1000": [200, 200, 450, 450],
        }

        images = _stage_images(
            "candidate_evidence",
            candidate,
            previous,
            {
                "visual_context": "proposal",
                "region": region,
                "proposal_catalog": [region],
            },
        )

        self.assertEqual(len(images), 11)
        labels = [label for label, _ in images]
        self.assertIn("Exact previous accepted T1 object-mask crop", labels)
        self.assertIn("Exact previous accepted T2 object-mask crop", labels)
        self.assertIn("Exact previous accepted change-mask crop", labels)
        self.assertIn("Exact candidate-added pixel crop", labels)
        self.assertIn("Exact candidate-removed pixel crop", labels)
        self.assertIn(
            "Exact candidate-delta-only T2 RGB crop; cyan contour marks delta",
            labels,
        )
        self.assertNotIn("Exact T1 proposal crop", labels)
        self.assertNotIn("Exact T2 proposal crop", labels)

    def test_direct_candidate_includes_delta_masks_and_exact_delta_crops(self):
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        previous_change = np.zeros((32, 32), dtype=bool)
        candidate_change = previous_change.copy()
        candidate_change[8:10, 8:10] = True
        previous = ChangeState(
            image, image, "building", previous_change, previous_change, previous_change
        )
        candidate = ChangeState(
            image, image, "building", previous_change, candidate_change, candidate_change
        )

        images = _stage_images(
            "direct", candidate, previous, {"mode": "candidate"}
        )

        labels = [label for label, _ in images]
        self.assertIn("Candidate-added change pixels", labels)
        self.assertIn("Candidate-removed change pixels", labels)
        self.assertIn(
            "Exact candidate-delta-only T1 RGB crop; cyan contour marks delta",
            labels,
        )
        self.assertIn("Exact candidate-added pixel crop", labels)

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

    def test_atomic_prompt_keeps_target_semantics_and_grounded_checklist(self):
        audit_prompt = _stage_prompt(
            "audit",
            {
                "target_class": "building",
                "region": {"region_id": "r7"},
            },
        )

        self.assertNotIn("visual_confidence", audit_prompt)
        self.assertNotIn("<CONFIDENCE_0_TO_1>", audit_prompt)
        self.assertIn('"audit_checklist"', audit_prompt)
        self.assertIn('"mask_assessment"', audit_prompt)
        self.assertIn('"evidence":"short observable local evidence"', audit_prompt)
        self.assertIn("trailers, RVs, mobile equipment", audit_prompt)
        self.assertIn("persisted global screening hypothesis", audit_prompt)
        self.assertIn("Never silently replace it", audit_prompt)

    def test_atomic_stage_includes_exact_component_geometry(self):
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        t2 = np.zeros((32, 32), dtype=bool)
        t2[8:12, 8:12] = True
        state = ChangeState(image, image, "building", np.zeros_like(t2), t2, t2)
        region = {
            "region_id": "r0",
            "audit_kind": "present",
            "component_seed_normalized_1000": [300, 300],
            "box_normalized_1000": [200, 200, 450, 450],
        }

        images = _stage_images(
            "audit",
            state,
            None,
            {
                "visual_context": "proposal",
                "region": region,
                "proposal_catalog": [region],
            },
        )

        self.assertEqual(images[-1][0], "Exact audited component geometry crop")
        self.assertTrue(np.asarray(images[-1][1]).any())

    def test_audit_checklist_deterministically_derives_error_and_quality(self):
        correct = AuditChecklist(
            evidence_sufficient="pass",
            target_class_only="pass",
            white_pixels_supported="pass",
            boundary_alignment="pass",
            internal_holes_absent="pass",
            changed_object_extent_complete="pass",
            fragment_artifacts_absent="pass",
        )
        mixed = AuditChecklist(
            evidence_sufficient="pass",
            target_class_only="pass",
            white_pixels_supported="fail",
            boundary_alignment="fail",
            internal_holes_absent="pass",
            changed_object_extent_complete="fail",
            fragment_artifacts_absent="pass",
        )
        missing = AuditChecklist(
            evidence_sufficient="pass",
            target_class_only="not_applicable",
            white_pixels_supported="not_applicable",
            boundary_alignment="not_applicable",
            internal_holes_absent="pass",
            changed_object_extent_complete="fail",
            fragment_artifacts_absent="not_applicable",
        )
        uncertain = AuditChecklist(
            evidence_sufficient="uncertain",
            target_class_only="uncertain",
            white_pixels_supported="uncertain",
            boundary_alignment="uncertain",
            internal_holes_absent="uncertain",
            changed_object_extent_complete="uncertain",
            fragment_artifacts_absent="uncertain",
        )

        self.assertEqual(correct.error_type, "none")
        self.assertEqual(correct.quality_score, 1.0)
        self.assertEqual(mixed.error_type, "mixed_error")
        self.assertEqual(mixed.quality_score, 0.5)
        self.assertEqual(missing.error_type, "false_negative")
        self.assertEqual(missing.quality_score, 0.5)
        self.assertEqual(uncertain.error_type, "uncertain_region")
        self.assertEqual(uncertain.quality_score, 0.0)


if __name__ == "__main__":
    unittest.main()
