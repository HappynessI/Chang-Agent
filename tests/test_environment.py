import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from change_agent.adapters.omniovcd_adapter import InitializationResult, MaskPairProcessor
from change_agent.environment import ChangeAgentEnvironment
from change_agent.executor import ActionExecutor
from change_agent.state import AgentAction, VerifierOutput


class Backend:
    def __init__(self):
        self.processor = MaskPairProcessor(overlap_threshold=0.5)

    def initialize(self, t1_image, t2_image, query):
        t1 = np.zeros(t1_image.shape[:2], dtype=bool)
        t2 = np.zeros_like(t1)
        t1[2:5, 2:5] = True
        return InitializationResult(t1, t2, self.processor.rebuild(t1, t2))

    def rebuild(self, t1_mask, t2_mask, evidence):
        return self.processor.rebuild(t1_mask, t2_mask, evidence)


class Point:
    def refine(self, image, initial_mask, coordinate, is_positive):
        output = initial_mask.copy()
        x, y = coordinate
        output[y, x] = is_positive
        return output


class Box:
    def __init__(self):
        self.last_box = None

    def segment_box(self, image, box_cxcywh_normalized, query):
        self.last_box = box_cxcywh_normalized
        output = np.zeros(image.shape[:2], dtype=bool)
        output[2:5, 2:5] = True
        return output


class EvidenceVerifier:
    def verify(self, state, previous_score, previous_action):
        # A matched T1/T2 instance produces an empty change mask and the best score.
        score = 0.9 if not state.change_mask.any() else 0.3
        delta = 0 if previous_score is None else score - previous_score
        return VerifierOutput(
            quality_score=score,
            score_delta=delta,
            error_type="none" if score > 0.8 else "uncertain_region",
            target_view="t2",
            suggested_action="finish" if score > 0.8 else "box",
            feedback="test feedback",
            accept=score > 0.8,
        )


class EnvironmentTest(unittest.TestCase):
    def setUp(self):
        self.box = Box()
        self.environment = ChangeAgentEnvironment(
            Backend(), ActionExecutor(Point(), self.box), EvidenceVerifier(), max_steps=4
        )
        self.image1 = np.zeros((8, 8, 3), dtype=np.uint8)
        self.image2 = np.ones_like(self.image1)

    def test_box_step_rebuilds_state_and_finish_can_be_accepted(self):
        observation = self.environment.reset(self.image1, self.image2, "building")
        self.assertTrue(observation.change_mask.any())
        observation, done = self.environment.step(
            AgentAction("t2", "box", box=(2, 2, 5, 5))
        )
        self.assertFalse(done)
        self.assertFalse(observation.change_mask.any())
        self.assertIsNotNone(self.box.last_box)
        _, done = self.environment.step(AgentAction("t2", "finish"))
        self.assertTrue(done)
        self.assertEqual(self.environment.best_state.step_index, 1)

    def test_observation_does_not_expose_hidden_masks_or_gt(self):
        observation = self.environment.reset(self.image1, self.image2, "building")
        public = observation.to_mapping()
        self.assertNotIn("t1_mask", public)
        self.assertNotIn("t2_mask", public)
        self.assertFalse(any("gt" in key.lower() for key in public))

    def test_verifier_feedback_declares_normalized_public_coordinates(self):
        observation = self.environment.reset(self.image1, self.image2, "building")
        feedback = observation.feedback.to_dict()
        self.assertEqual(feedback["coordinate_space"], "normalized_0_1000")

    def test_raw_action_and_trajectory_artifacts(self):
        self.environment.reset(self.image1, self.image2, "building")
        raw = '{"target_view":"t1","action":"positive_point","coordinate":[0,0]}'
        self.environment.step(raw)
        with tempfile.TemporaryDirectory() as directory:
            path = self.environment.trajectory.save(directory)
            payload = json.loads(path.read_text())
            self.assertEqual(payload["steps"][1]["raw_action"], raw)
            self.assertTrue((Path(directory) / "masks" / "step_001.npy").exists())

    def test_trajectory_can_save_masks_in_a_separate_directory(self):
        self.environment.reset(self.image1, self.image2, "building")
        self.environment.step(AgentAction("t1", "positive_point", coordinate=(0, 0)))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trajectory_path = self.environment.trajectory.save(
                root / "trajectories" / "sample", root / "masks" / "sample"
            )
            payload = json.loads(trajectory_path.read_text())
            mask_path = (trajectory_path.parent / payload["steps"][1]["change_mask_file"]).resolve()
            self.assertTrue(mask_path.exists())
            self.assertEqual(mask_path, (root / "masks" / "sample" / "step_001.npy").resolve())

    def test_runtime_rejects_non_inference_mode(self):
        with self.assertRaises(ValueError):
            ChangeAgentEnvironment(
                Backend(),
                ActionExecutor(Point(), self.box),
                EvidenceVerifier(),
                inference_only=False,
            )


if __name__ == "__main__":
    unittest.main()
