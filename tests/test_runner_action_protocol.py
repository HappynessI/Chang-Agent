import unittest

from change_agent.action_parser import ActionValidationError
from tools.run_levir_change_agent import _execute_action_with_retries


class InvalidAgent:
    def __init__(self):
        self.validation_errors = []

    def generate_raw(self, observation, validation_error=None):
        self.validation_errors.append(validation_error)
        return '{"target_view":"t2","action":"finish"}'


class RejectingEnvironment:
    def __init__(self):
        self.calls = []

    def step(self, raw):
        self.calls.append(raw)
        raise ActionValidationError("finish is not authorized")


class RunnerActionProtocolTest(unittest.TestCase):
    def test_retry_exhaustion_does_not_execute_a_fallback_action(self):
        agent = InvalidAgent()
        environment = RejectingEnvironment()
        observation = object()

        returned, errors, executed = _execute_action_with_retries(
            agent, environment, observation, retries=3
        )

        self.assertIs(returned, observation)
        self.assertFalse(executed)
        self.assertEqual(len(errors), 3)
        self.assertEqual(len(environment.calls), 3)
        self.assertEqual(agent.validation_errors[1:], [
            "finish is not authorized",
            "finish is not authorized",
        ])
        self.assertTrue(all('"action":"finish"' in raw for raw in environment.calls))


if __name__ == "__main__":
    unittest.main()
