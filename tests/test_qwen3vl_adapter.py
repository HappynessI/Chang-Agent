import unittest

import numpy as np

from change_agent.adapters.qwen3vl_adapter import GroundingModelQwen3VL
from change_agent.state import AgentObservation, VerifierOutput


class FakeInputs(dict):
    def to(self, device):
        return self


class FakeProcessor:
    def __init__(self):
        self.messages = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        return FakeInputs(input_ids=np.zeros((1, 3), dtype=np.int64))

    def batch_decode(self, generated, **kwargs):
        return ['{"target_view":"t2","action":"positive_point","coordinate":[500,500]}']


class FakeModel:
    device = "cpu"

    def generate(self, **kwargs):
        return np.zeros((1, 5), dtype=np.int64)


class QwenAdapterTest(unittest.TestCase):
    def test_modern_messages_label_all_images_and_parse_action(self):
        processor = FakeProcessor()
        adapter = GroundingModelQwen3VL(model=FakeModel(), processor=processor)
        image = np.zeros((11, 21, 3), dtype=np.uint8)
        observation = AgentObservation(image, image, "building", np.zeros((11, 21)))
        raw, action = adapter.act(observation)
        texts = [
            item["text"]
            for item in processor.messages[0]["content"]
            if item["type"] == "text"
        ]
        self.assertIn("T1 image", texts[0])
        self.assertIn("T2 image", texts[1])
        self.assertIn("Current binary change mask", texts[2])
        self.assertNotIn("<img>", " ".join(texts))
        self.assertIn("coordinate protocol is system-defined", texts[-1])
        self.assertIn("Never output coordinate_frame", texts[-1])
        self.assertEqual(action.coordinate, (10, 5))
        self.assertIn("positive_point", raw)

    def test_validation_error_is_injected_into_retry_prompt(self):
        processor = FakeProcessor()
        adapter = GroundingModelQwen3VL(model=FakeModel(), processor=processor)
        image = np.zeros((11, 21, 3), dtype=np.uint8)
        observation = AgentObservation(image, image, "building", np.zeros((11, 21)))
        adapter.generate_raw(observation, "finish is forbidden before a tool action")
        texts = [
            item["text"]
            for item in processor.messages[0]["content"]
            if item["type"] == "text"
        ]
        self.assertIn("previous action was rejected", texts[-1])
        self.assertIn("finish is forbidden", texts[-1])

    def test_verifier_point_recommendation_injects_only_point_example(self):
        processor = FakeProcessor()
        adapter = GroundingModelQwen3VL(model=FakeModel(), processor=processor)
        image = np.zeros((11, 21, 3), dtype=np.uint8)
        observation = AgentObservation(
            image,
            image,
            "building",
            np.zeros((11, 21)),
            feedback=VerifierOutput(
                quality_score=0.45,
                progress_score=0.12,
                error_type="false_negative",
                target_view="t2",
                error_region=(400, 200, 700, 600),
                suggested_action="positive_point",
                feedback="Several new buildings are missing.",
            ),
        )
        adapter.generate_raw(observation)
        prompt = processor.messages[0]["content"][-1]["text"]
        self.assertIn("Verifier recommends a point action", prompt)
        self.assertIn(
            '{"target_view":"t2","action":"positive_point","coordinate":[620,410]}',
            prompt,
        )
        self.assertIn("Never omit coordinate", prompt)
        self.assertNotIn('"box":[120,180,760,820]', prompt)

    def test_verifier_box_recommendation_injects_only_box_example(self):
        processor = FakeProcessor()
        adapter = GroundingModelQwen3VL(model=FakeModel(), processor=processor)
        image = np.zeros((11, 21, 3), dtype=np.uint8)
        observation = AgentObservation(
            image,
            image,
            "building",
            np.zeros((11, 21)),
            feedback=VerifierOutput(
                quality_score=0.4,
                progress_score=-0.1,
                error_type="mixed_error",
                target_view="t1",
                error_region=(100, 100, 800, 900),
                suggested_action="box",
                feedback="The mixed error needs a regional edit.",
            ),
        )
        adapter.generate_raw(observation)
        prompt = processor.messages[0]["content"][-1]["text"]
        self.assertIn("Verifier recommends a box action", prompt)
        self.assertIn(
            '{"target_view":"t1","action":"box","box":[120,180,760,820]}',
            prompt,
        )
        self.assertIn("Never omit box", prompt)
        self.assertNotIn('"coordinate":[620,410]', prompt)

    def test_retry_for_missing_point_coordinate_repeats_exact_structure(self):
        processor = FakeProcessor()
        adapter = GroundingModelQwen3VL(model=FakeModel(), processor=processor)
        image = np.zeros((11, 21, 3), dtype=np.uint8)
        observation = AgentObservation(image, image, "building", np.zeros((11, 21)))
        adapter.generate_raw(
            observation,
            "coordinate must contain exactly 2 numbers",
            '{"target_view":"t1","action":"positive_point"}',
        )
        prompt = processor.messages[0]["content"][-1]["text"]
        self.assertIn("previous point action omitted coordinate", prompt)
        self.assertIn(
            '{"target_view":"t1","action":"positive_point","coordinate":[x,y]}',
            prompt,
        )
        self.assertIn("numeric values in [0,1000]", prompt)

    def test_retry_for_missing_box_repeats_exact_structure(self):
        processor = FakeProcessor()
        adapter = GroundingModelQwen3VL(model=FakeModel(), processor=processor)
        image = np.zeros((11, 21, 3), dtype=np.uint8)
        observation = AgentObservation(image, image, "building", np.zeros((11, 21)))
        adapter.generate_raw(
            observation,
            "box must contain exactly 4 numbers",
            '{"target_view":"t2","action":"box"}',
        )
        prompt = processor.messages[0]["content"][-1]["text"]
        self.assertIn("previous box action omitted box", prompt)
        self.assertIn(
            '{"target_view":"t2","action":"box","box":[x1,y1,x2,y2]}',
            prompt,
        )
        self.assertIn("Never omit box", prompt)

    def test_invalid_verifier_feedback_does_not_authorize_finish(self):
        processor = FakeProcessor()
        adapter = GroundingModelQwen3VL(model=FakeModel(), processor=processor)
        image = np.zeros((11, 21, 3), dtype=np.uint8)
        observation = AgentObservation(
            image,
            image,
            "building",
            np.zeros((11, 21)),
            feedback=VerifierOutput(
                quality_score=0.4,
                error_type="false_positive_change",
                suggested_action=None,
                feedback="recheck required",
                verifier_valid=False,
                localization_valid=False,
            ),
            history_summary="step=1, action=box, score=0.400, error=false_positive_change",
        )
        adapter.generate_raw(observation)
        texts = [
            item["text"]
            for item in processor.messages[0]["content"]
            if item["type"] == "text"
        ]
        self.assertIn("Verifier feedback is invalid", texts[-1])
        self.assertIn("cannot authorize finish", texts[-1])


if __name__ == "__main__":
    unittest.main()
