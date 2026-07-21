import unittest

import numpy as np

from change_agent.adapters.direct_verifier import DirectQwenVerifier
from change_agent.state import AgentAction, ChangeState, VerifierOutput


def make_state():
    image1 = np.zeros((16, 16, 3), dtype=np.uint8)
    image2 = image1.copy()
    image2[4:9, 4:9] = 180
    t1 = np.zeros((16, 16), dtype=bool)
    t2 = np.zeros_like(t1)
    t2[4:9, 4:9] = True
    return ChangeState(image1, image2, "building", t1, t2, t2)


class DirectBackend:
    def __init__(self, verdict):
        self.verdicts = iter(verdict if isinstance(verdict, list) else [verdict])
        self.calls = []
        self.payloads = []

    def generate_stage(self, stage, state, payload, previous_state=None):
        self.calls.append((stage, payload["mode"], previous_state is not None))
        self.payloads.append(dict(payload))
        return {"verdict": next(self.verdicts)}


class DirectVerifierTest(unittest.TestCase):
    def test_direct_verifier_keeps_false_positive_despite_real_appearance(self):
        backend = DirectBackend(
            {
                "comparison": "initial",
                "quality_score": 0.4,
                "progress_score": 0.0,
                "accept": False,
                "error_type": "false_positive_change",
                "target_view": "t2",
                "suggested_action": "negative_point",
                "coordinate_normalized_1000": [500, 500],
                "box_normalized_1000": None,
                "feedback": "White change extends beyond supported building change.",
            }
        )

        output = DirectQwenVerifier(backend).verify(make_state(), None, None)

        self.assertTrue(output.verifier_valid)
        self.assertEqual(output.error_type, "false_positive_change")
        self.assertEqual(output.suggested_action, "negative_point")
        self.assertEqual(output.error_region, (500, 500, 500, 500))
        self.assertFalse(output.stop)
        self.assertEqual(backend.calls, [("direct", "initial", False)])

    def test_direct_clean_initial_state_can_stop(self):
        backend = DirectBackend(
            {
                "comparison": "initial",
                "quality_score": 0.91,
                "progress_score": 0.0,
                "accept": True,
                "error_type": "none",
                "target_view": None,
                "suggested_action": "finish",
                "coordinate_normalized_1000": None,
                "box_normalized_1000": None,
                "feedback": "No remaining error.",
            }
        )

        output = DirectQwenVerifier(backend).verify(make_state(), None, None)

        self.assertTrue(output.accept)
        self.assertTrue(output.stop)

    def test_direct_verifier_canonicalizes_missing_detection(self):
        backend = DirectBackend(
            {
                "comparison": "initial",
                "quality_score": 0.3,
                "progress_score": -0.7,
                "accept": False,
                "error_type": "missing_detection",
                "target_view": "t2",
                "suggested_action": "positive_point",
                "coordinate_normalized_1000": [650, 340],
                "box_normalized_1000": None,
                "feedback": "T2 misses an object.",
            }
        )

        output = DirectQwenVerifier(backend).verify(make_state(), None, None)

        self.assertTrue(output.verifier_valid)
        self.assertEqual(output.error_type, "false_negative")
        self.assertEqual(output.suggested_action, "positive_point")

    def test_direct_verifier_replans_after_rejected_candidate(self):
        initial_verdict = {
            "comparison": "initial",
            "quality_score": 0.3,
            "progress_score": -0.7,
            "accept": False,
            "error_type": "false_negative",
            "target_view": "t2",
            "suggested_action": "positive_point",
            "coordinate_normalized_1000": [650, 340],
            "box_normalized_1000": None,
            "feedback": "T2 misses an object.",
        }
        replan_verdict = {
            "comparison": "uncertain",
            "quality_score": 0.3,
            "progress_score": 0.0,
            "accept": False,
            "error_type": "false_negative",
            "target_view": "t2",
            "suggested_action": "positive_point",
            "coordinate_normalized_1000": [300, 700],
            "box_normalized_1000": None,
            "feedback": "Use a more local missing-object point.",
        }
        backend = DirectBackend([initial_verdict, replan_verdict])
        verifier = DirectQwenVerifier(backend)
        accepted_state = make_state()
        accepted_feedback = verifier.verify(accepted_state, None, None)
        rejected_candidate = accepted_state.clone()
        rejected_candidate.t2_mask[1, 1] = True
        rejected_candidate.change_mask[1, 1] = True

        output = verifier.replan_after_rejection(
            accepted_state,
            rejected_candidate,
            accepted_feedback,
            VerifierOutput(
                comparison="worse",
                error_type="false_negative",
                target_view="t2",
                suggested_action="positive_point",
                feedback="Candidate exceeded locality.",
            ),
            AgentAction("t2", "positive_point", coordinate=(10, 5)),
            ["locality_outside_roi_exceeded"],
            [
                {
                    "step_index": 1,
                    "action": {"action": "positive_point"},
                    "rejection_reasons": ["locality_outside_roi_exceeded"],
                }
            ],
        )

        self.assertTrue(output.verifier_valid)
        self.assertEqual(output.comparison, "uncertain")
        self.assertFalse(output.accept)
        self.assertEqual(output.error_region, (300, 700, 300, 700))
        self.assertEqual(
            backend.calls,
            [("direct", "initial", False), ("direct", "replan", True)],
        )
        self.assertEqual(
            backend.payloads[-1]["rejection_reasons"],
            ["locality_outside_roi_exceeded"],
        )
        self.assertEqual(
            backend.payloads[-1]["rejected_candidate_mask_delta"]["change_changed_pixels"],
            1,
        )
        self.assertEqual(backend.payloads[-1]["rejection_history"][0]["step_index"], 1)

    def test_direct_replan_rejects_same_pixel_action(self):
        initial_verdict = {
            "comparison": "initial",
            "quality_score": 0.3,
            "progress_score": -0.7,
            "accept": False,
            "error_type": "false_negative",
            "target_view": "t2",
            "suggested_action": "positive_point",
            "coordinate_normalized_1000": [667, 333],
            "box_normalized_1000": None,
            "feedback": "T2 misses an object.",
        }
        repeated_replan_verdict = dict(initial_verdict, comparison="uncertain")
        backend = DirectBackend([initial_verdict, repeated_replan_verdict])
        verifier = DirectQwenVerifier(backend)
        accepted_state = make_state()
        accepted_feedback = verifier.verify(accepted_state, None, None)

        output = verifier.replan_after_rejection(
            accepted_state,
            accepted_state.clone(),
            accepted_feedback,
            VerifierOutput(
                comparison="worse",
                error_type="false_negative",
                target_view="t2",
                suggested_action="positive_point",
                feedback="Candidate was rejected.",
            ),
            AgentAction("t2", "positive_point", coordinate=(10, 5)),
            ["locality_outside_roi_exceeded"],
            [],
        )

        self.assertFalse(output.verifier_valid)
        self.assertIsNone(output.suggested_action)
        self.assertIn(
            "repeated the rejected action",
            verifier.last_evidence["validation_errors"][0],
        )


if __name__ == "__main__":
    unittest.main()
