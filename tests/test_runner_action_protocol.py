import unittest
from types import SimpleNamespace

from change_agent.action_parser import ActionValidationError
from tools.run_levir_change_agent import (
    _execute_action_with_retries,
    _execute_direct_verifier_action,
    _initial_verifier_stop_reason,
)


class InvalidAgent:
    def __init__(self):
        self.validation_errors = []
        self.previous_raws = []

    def generate_raw(self, observation, validation_error=None, previous_raw=None):
        self.validation_errors.append(validation_error)
        self.previous_raws.append(previous_raw)
        return '{"target_view":"t2","action":"finish"}'


class RejectingEnvironment:
    def __init__(self):
        self.calls = []

    def step(self, raw):
        self.calls.append(raw)
        raise ActionValidationError("finish is not authorized")


class DirectEnvironment:
    def __init__(self):
        self.calls = []
        self.trajectory = SimpleNamespace(
            entries=[SimpleNamespace(execution={})]
        )

    def step(self, raw):
        self.calls.append(raw)
        return "next_observation", False


class RunnerActionProtocolTest(unittest.TestCase):
    def test_initial_invalid_verifier_stops_before_agent_actions(self):
        observation = SimpleNamespace(
            feedback=SimpleNamespace(verifier_valid=False)
        )

        self.assertEqual(
            _initial_verifier_stop_reason(observation), "initial_verifier_invalid"
        )
        self.assertIsNone(
            _initial_verifier_stop_reason(
                SimpleNamespace(feedback=SimpleNamespace(verifier_valid=True))
            )
        )

    def test_initial_authorized_finish_stops_before_agent_actions(self):
        observation = SimpleNamespace(
            feedback=SimpleNamespace(
                verifier_valid=True,
                accept=True,
                stop=True,
                comparison="initial",
                error_type="none",
            )
        )

        self.assertEqual(
            _initial_verifier_stop_reason(observation),
            "initial_verifier_authorized_finish",
        )

    def test_retry_exhaustion_does_not_execute_a_fallback_action(self):
        agent = InvalidAgent()
        environment = RejectingEnvironment()
        observation = object()

        returned, errors, executed = _execute_action_with_retries(
            agent, environment, observation, retries=3, loop_index=4
        )

        self.assertIs(returned, observation)
        self.assertFalse(executed)
        self.assertEqual(len(errors), 3)
        self.assertEqual(len(environment.calls), 3)
        self.assertEqual(agent.validation_errors[1:], [
            "finish is not authorized",
            "finish is not authorized",
        ])
        self.assertEqual(agent.previous_raws[1:], [
            '{"target_view":"t2","action":"finish"}',
            '{"target_view":"t2","action":"finish"}',
        ])
        self.assertTrue(all('"action":"finish"' in raw for raw in environment.calls))
        self.assertEqual([item["loop_index"] for item in errors], [4, 4, 4])
        self.assertEqual([item["attempt_index"] for item in errors], [1, 2, 3])
        self.assertTrue(all("prompt_hash" in item for item in errors))

    def test_direct_verifier_executes_its_normalized_point_plan(self):
        environment = DirectEnvironment()
        observation = SimpleNamespace(
            feedback=SimpleNamespace(
                verifier_valid=True,
                target_view="t2",
                suggested_action="negative_point",
                error_region=(250, 300, 250, 300),
            )
        )

        returned, errors, executed = _execute_direct_verifier_action(
            environment, observation, loop_index=2
        )

        self.assertEqual(returned, "next_observation")
        self.assertEqual(errors, [])
        self.assertTrue(executed)
        self.assertEqual(
            environment.calls,
            ['{"target_view":"t2","action":"negative_point","coordinate":[250,300]}'],
        )
        self.assertEqual(
            environment.trajectory.entries[-1].execution["action_generation"]["source"],
            "direct_verifier_full_context",
        )


if __name__ == "__main__":
    unittest.main()
