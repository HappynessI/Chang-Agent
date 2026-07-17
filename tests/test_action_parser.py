import unittest

from change_agent.action_parser import ActionParser, ActionValidationError
from change_agent.executor import xyxy_to_normalized_cxcywh


class ActionParserTest(unittest.TestCase):
    def setUp(self):
        self.parser = ActionParser()

    def test_parses_json_fence_and_converts_xy(self):
        action = self.parser.parse(
            '```json\n{"target_view":"t2","action":"positive_point",'
            '"coordinate":[1000,500]}\n```',
            (101, 51),
        )
        self.assertEqual(action.coordinate, (100, 25))
        self.assertEqual(action.target_view, "t2")

    def test_parses_box(self):
        action = self.parser.parse(
            '{"target_view":"t1","action":"box","box":[100,200,900,800]}',
            (101, 101),
        )
        self.assertEqual(action.box, (10, 20, 90, 80))

    def test_rejects_invalid_payloads(self):
        bad = [
            '{"target_view":"t3","action":"finish"}',
            '{"target_view":"t1","action":"positive_point","coordinate":[-1,4]}',
            '{"target_view":"t1","action":"box","box":[500,0,400,1000]}',
            '{"target_view":"t1","action":"finish","box":[0,0,1,1]}',
            '{"target_view":"t1","action":"finish","surprise":1}',
            "not json",
        ]
        for raw in bad:
            with self.subTest(raw=raw), self.assertRaises(ActionValidationError):
                self.parser.parse(raw, (100, 100))

    def test_box_conversion_for_sam3(self):
        result = xyxy_to_normalized_cxcywh((10, 20, 30, 60), (100, 100))
        self.assertEqual(result, (0.2, 0.4, 0.2, 0.4))


if __name__ == "__main__":
    unittest.main()

