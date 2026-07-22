import unittest
import json

import numpy as np

from change_agent.adapters.staged_verifier import StagedQwenVerifier
from change_agent.adapters.stage_backends import _extract_stage_json, _stage_prompt
from change_agent.coordinates import normalized_point_to_pixel
from change_agent.state import AgentAction, ChangeState, VerifierOutput
from change_agent.verifier_protocol import StageProtocolError
from change_agent.verifier_regions import attach_verifier_regions


def make_state():
    image1 = np.zeros((16, 16, 3), dtype=np.uint8)
    image2 = image1.copy()
    image2[4:9, 4:9] = 180
    t1 = np.zeros((16, 16), dtype=bool)
    t2 = np.zeros_like(t1)
    t2[4:9, 4:9] = True
    state = ChangeState(image1, image2, "building", t1, t2, t2)
    attach_verifier_regions(state, max_regions=6, min_component_area=1)
    return state


def make_two_region_state():
    image1 = np.zeros((32, 32, 3), dtype=np.uint8)
    image2 = image1.copy()
    t1 = np.zeros((32, 32), dtype=bool)
    t2 = np.zeros_like(t1)
    t2[3:8, 3:8] = True
    t2[20:26, 20:26] = True
    state = ChangeState(image1, image2, "building", t1, t2, t2)
    attach_verifier_regions(state, max_regions=6, min_component_area=1)
    return state


class ScriptedBackend:
    def __init__(self, *, t1="background", t2="building", error="none", target=None, action="negative_point"):
        self.t1 = t1
        self.t2 = t2
        self.error = error
        self.target = target
        self.action = action
        self.calls = []
        self.previous_seen = []

    def generate_stage(self, stage, state, payload, previous_state=None):
        self.calls.append(stage)
        self.previous_seen.append(previous_state is not None)
        if stage == "select":
            return {
                "selection": {
                    "region_ids": [payload["proposal_catalog"][0]["region_id"]],
                    "reason": "Most material marked proposal.",
                }
            }
        region = payload.get("region", {})
        region_id = region.get("region_id")
        if stage in {"evidence", "candidate_evidence"}:
            return {
                "region_id": region_id,
                "visual_judgment": {
                    "t1_state": self.t1,
                    "t2_state": self.t2,
                    "visual_confidence": 0.9,
                    "evidence_quality": "clear",
                },
            }
        if stage in {"diagnosis", "candidate_diagnosis"}:
            return {
                "region_id": region_id,
                "diagnosis": {
                    "error_type": self.error,
                    "target_view": self.target,
                    "confidence": 0.9,
                },
            }
        if stage == "plan":
            return {
                "region_id": region_id,
                "plan": {
                    "action": self.action,
                    "target_view": self.target,
                    "coordinate_normalized_1000": region["component_seed_normalized_1000"],
                    "box_normalized_1000": None,
                },
            }
        mode = payload["mode"]
        return {
            "decision": {
                "comparison": "initial" if mode == "initial" else "better",
                "quality_score": 0.91,
                "progress_score": 0.0 if mode == "initial" else 0.2,
                "accept": self.error == "none" or mode == "candidate",
                "stop": self.error == "none",
                "feedback": "Structured staged decision.",
            }
        }


class RepairingBackend(ScriptedBackend):
    def __init__(self):
        super().__init__(error="none", target=None)
        self.repair_errors = []

    def generate_stage(self, stage, state, payload, previous_state=None):
        if stage == "evidence":
            self.calls.append(stage)
            return {"region": payload["region"], "schema": payload["schema"]}
        return super().generate_stage(stage, state, payload, previous_state)

    def repair_stage(
        self, stage, state, payload, validation_error, previous_state=None
    ):
        self.calls.append(f"repair:{stage}")
        self.repair_errors.append(validation_error)
        region_id = payload["region"]["region_id"]
        return {
            "region_id": region_id,
            "visual_judgment": {
                "t1_state": "background",
                "t2_state": "building",
                "visual_confidence": 0.9,
                "evidence_quality": "clear",
            },
        }


class MissingDiagnosisConfidenceBackend(ScriptedBackend):
    def generate_stage(self, stage, state, payload, previous_state=None):
        if stage == "diagnosis":
            self.calls.append(stage)
            return {
                "region_id": payload["region"]["region_id"],
                "diagnosis": {"error_type": "none", "target_view": None},
            }
        return super().generate_stage(stage, state, payload, previous_state)


class BetterButNotAcceptedBackend(ScriptedBackend):
    def generate_stage(self, stage, state, payload, previous_state=None):
        response = super().generate_stage(stage, state, payload, previous_state)
        if stage == "decision" and payload["mode"] == "candidate":
            response["decision"]["accept"] = False
            response["decision"]["stop"] = False
        return response


class SelectAllBackend(ScriptedBackend):
    def generate_stage(self, stage, state, payload, previous_state=None):
        if stage == "select":
            self.calls.append(stage)
            self.previous_seen.append(previous_state is not None)
            return {
                "selection": {
                    "region_ids": [
                        item["region_id"] for item in payload["proposal_catalog"]
                    ],
                    "reason": "Audit both marked regions.",
                }
            }
        return super().generate_stage(stage, state, payload, previous_state)


class StagedVerifierTest(unittest.TestCase):
    def test_rollback_replan_uses_distinct_cached_region(self):
        state = make_two_region_state()
        backend = SelectAllBackend(
            error="false_positive_change",
            target="t2",
        )
        verifier = StagedQwenVerifier(backend, max_selected_regions=2)
        accepted_feedback = verifier.verify(state, None, None)
        first_normalized = accepted_feedback.error_region[:2]
        first_pixel = normalized_point_to_pixel(first_normalized, state.image_size)
        rejected_action = AgentAction(
            "t2", "negative_point", coordinate=first_pixel
        )

        output = verifier.replan_after_rejection(
            state,
            state.clone(),
            accepted_feedback,
            VerifierOutput(comparison="worse", accept=False),
            rejected_action,
            ["candidate_effect_not_better"],
            [{"action": rejected_action.to_dict()}],
        )

        self.assertTrue(output.verifier_valid)
        self.assertEqual(output.suggested_action, "negative_point")
        self.assertNotEqual(output.error_region[:2], first_normalized)

    def test_identical_finish_authorization_sets_stop(self):
        state = make_state()
        verifier = StagedQwenVerifier(ScriptedBackend())
        verifier._last_valid_output = VerifierOutput(
            quality_score=0.9,
            comparison="better",
            error_type="none",
            suggested_action="finish",
            accept=True,
            stop=False,
        )

        output = verifier.verify(state.clone(), 0.9, None, state)

        self.assertTrue(output.stop)

    def test_stage_parser_selects_schema_match_instead_of_first_json(self):
        correct = {
            "region_id": "r0",
            "visual_judgment": {
                "t1_state": "background",
                "t2_state": "building",
                "visual_confidence": 0.9,
                "evidence_quality": "clear",
            },
        }
        raw = (
            json.dumps({"region": {"region_id": "r0"}, "schema": "input"})
            + "\n"
            + json.dumps(correct)
        )

        self.assertEqual(_extract_stage_json(raw, "evidence"), correct)

    def test_stage_parser_rejects_copied_environment_context(self):
        raw = json.dumps({"region": {"region_id": "r0"}, "schema": "input"})
        with self.assertRaisesRegex(StageProtocolError, "candidate_keys"):
            _extract_stage_json(raw, "evidence")

    def test_prompt_puts_output_contract_before_wrapped_environment(self):
        prompt = _stage_prompt(
            "evidence",
            {"region": {"region_id": "r7"}, "schema": "evidence_judgment_v1"},
        )

        self.assertLess(prompt.index("OUTPUT CONTRACT"), prompt.index("<ENVIRONMENT_FACTS>"))
        self.assertIn('"environment_facts"', prompt)
        self.assertIn('"region_id":"r7"', prompt)
        self.assertIn("Do not copy the Environment envelope", prompt)

    def test_diagnosis_prompt_has_no_default_none_bias(self):
        prompt = _stage_prompt(
            "diagnosis",
            {"region": {"region_id": "r7"}, "schema": "diagnosis_v1"},
        )

        self.assertNotIn("normally correct and error_type is none", prompt)
        self.assertIn("A proposal may contain both correct and incorrect pixels", prompt)
        self.assertIn("Use mixed_error", prompt)
        self.assertIn("Use none only when the whole audited region is supported", prompt)
        self.assertIn("<ERROR_TYPE>", prompt)

    def test_invalid_stage_output_is_repaired_before_verifier_aborts(self):
        backend = RepairingBackend()
        verifier = StagedQwenVerifier(backend, max_retries=2)

        output = verifier.verify(make_state(), None, None)

        self.assertTrue(output.verifier_valid)
        self.assertTrue(output.stop)
        self.assertIn("repair:evidence", backend.calls)
        self.assertIn("must contain exactly", backend.repair_errors[0])

    def test_missing_diagnosis_confidence_uses_conservative_default(self):
        backend = MissingDiagnosisConfidenceBackend(error="none", target=None)
        verifier = StagedQwenVerifier(backend)

        output = verifier.verify(make_state(), None, None)

        self.assertTrue(output.verifier_valid)
        diagnosis = verifier.last_evidence["stage_trace"]["diagnoses"][0]
        self.assertEqual(diagnosis["confidence"], 0.0)

    def test_correct_appearance_change_can_finish_without_action_plan(self):
        backend = ScriptedBackend(error="none", target=None)
        verifier = StagedQwenVerifier(backend)

        output = verifier.verify(make_state(), None, None)

        self.assertTrue(output.verifier_valid)
        self.assertTrue(output.accept)
        self.assertTrue(output.stop)
        self.assertEqual(output.suggested_action, "finish")
        self.assertEqual(
            backend.calls, ["select", "evidence", "diagnosis", "decision"]
        )
        self.assertEqual(
            verifier.last_evidence["stage_trace"]["evidence"][0]["change_mask_state"],
            "white",
        )

    def test_clear_appearance_change_can_still_be_false_positive(self):
        backend = ScriptedBackend(
            error="false_positive_change", target="t2", action="negative_point"
        )
        verifier = StagedQwenVerifier(backend)

        output = verifier.verify(make_state(), None, None)

        self.assertTrue(output.verifier_valid)
        self.assertEqual(output.error_type, "false_positive_change")
        self.assertEqual(output.suggested_action, "negative_point")
        self.assertNotIn("plan", backend.calls)

    def test_false_positive_without_white_target_seed_fails_closed(self):
        backend = ScriptedBackend(
            t1="background",
            t2="background",
            error="false_positive_change",
            target="t1",
            action="negative_point",
        )
        verifier = StagedQwenVerifier(backend)

        output = verifier.verify(make_state(), None, None)

        self.assertTrue(output.verifier_valid)
        self.assertFalse(output.localization_valid)
        self.assertIsNone(output.suggested_action)
        self.assertNotIn("plan", backend.calls)

    def test_valid_false_positive_plan_reuses_environment_seed(self):
        backend = ScriptedBackend(
            t1="background",
            t2="background",
            error="false_positive_change",
            target="t2",
            action="negative_point",
        )
        verifier = StagedQwenVerifier(backend)

        output = verifier.verify(make_state(), None, None)

        self.assertTrue(output.verifier_valid)
        self.assertEqual(output.error_type, "false_positive_change")
        self.assertEqual(output.target_view, "t2")
        self.assertEqual(output.suggested_action, "negative_point")
        self.assertEqual(output.error_region[0], output.error_region[2])
        self.assertEqual(output.error_region[1], output.error_region[3])
        proposal_seed = verifier.last_evidence["stage_trace"]["evidence"][0][
            "component_seed_normalized_1000"
        ]
        self.assertEqual(list(output.error_region[:2]), proposal_seed)
        self.assertNotIn("plan", backend.calls)

    def test_global_selection_rejects_unknown_region_id(self):
        class UnknownRegionBackend(ScriptedBackend):
            def generate_stage(self, stage, state, payload, previous_state=None):
                if stage == "select":
                    self.calls.append(stage)
                    return {
                        "selection": {
                            "region_ids": ["invented"],
                            "reason": "invalid",
                        }
                    }
                return super().generate_stage(stage, state, payload, previous_state)

        verifier = StagedQwenVerifier(UnknownRegionBackend(), max_retries=1)

        output = verifier.verify(make_state(), None, None)

        self.assertFalse(output.verifier_valid)
        self.assertIn("unknown region ids", verifier.last_evidence["validation_errors"][0])

    def test_candidate_comparison_receives_previous_state_and_accepts_better(self):
        candidate = make_state()
        previous_mask = np.zeros_like(candidate.change_mask)
        previous = ChangeState(
            candidate.t1_image,
            candidate.t2_image,
            candidate.query,
            candidate.t1_mask,
            previous_mask,
            previous_mask,
        )
        attach_verifier_regions(candidate, previous, max_regions=6, min_component_area=1)
        backend = ScriptedBackend(error="none", target=None)
        verifier = StagedQwenVerifier(backend)

        output = verifier.verify(candidate, 0.4, None, previous)

        self.assertTrue(output.verifier_valid)
        self.assertEqual(output.comparison, "better")
        self.assertTrue(output.accept)
        self.assertTrue(all(backend.previous_seen))
        self.assertEqual(
            verifier.last_evidence["stage_trace"]["mode"], "candidate"
        )

    def test_candidate_better_but_not_accepted_is_not_repaired_to_accept(self):
        candidate = make_state()
        previous_mask = np.zeros_like(candidate.change_mask)
        previous = ChangeState(
            candidate.t1_image,
            candidate.t2_image,
            candidate.query,
            candidate.t1_mask,
            previous_mask,
            previous_mask,
        )
        attach_verifier_regions(candidate, previous, max_regions=6, min_component_area=1)
        verifier = StagedQwenVerifier(
            BetterButNotAcceptedBackend(error="none", target=None)
        )

        output = verifier.verify(candidate, 0.4, None, previous)

        self.assertTrue(output.verifier_valid)
        self.assertEqual(output.comparison, "better")
        self.assertFalse(output.accept)


if __name__ == "__main__":
    unittest.main()
