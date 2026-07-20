import unittest
import json

import numpy as np

from change_agent.adapters.staged_verifier import StagedQwenVerifier
from change_agent.adapters.stage_backends import _extract_stage_json, _stage_prompt
from change_agent.state import ChangeState
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


class StagedVerifierTest(unittest.TestCase):
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
        self.assertEqual(backend.calls, ["evidence", "diagnosis", "decision"])
        self.assertEqual(
            verifier.last_evidence["stage_trace"]["evidence"][0]["change_mask_state"],
            "white",
        )

    def test_clear_appearance_change_cannot_be_false_positive(self):
        backend = ScriptedBackend(
            error="false_positive_change", target="t2", action="negative_point"
        )
        verifier = StagedQwenVerifier(backend)

        output = verifier.verify(make_state(), None, None)

        self.assertFalse(output.verifier_valid)
        self.assertIn(
            "cannot be labeled false_positive_change",
            verifier.last_evidence["validation_errors"][0],
        )
        self.assertNotIn("plan", backend.calls)

    def test_false_positive_plan_must_target_a_white_editable_seed(self):
        backend = ScriptedBackend(
            t1="background",
            t2="background",
            error="false_positive_change",
            target="t1",
            action="negative_point",
        )
        verifier = StagedQwenVerifier(backend)

        output = verifier.verify(make_state(), None, None)

        self.assertFalse(output.verifier_valid)
        self.assertIn(
            "negative_point requires a white editable seed",
            verifier.last_evidence["validation_errors"][0],
        )

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


if __name__ == "__main__":
    unittest.main()
