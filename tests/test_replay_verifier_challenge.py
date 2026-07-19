import tempfile
import unittest
from pathlib import Path

from tools.replay_verifier_challenge import atomic_output, comparison_label


class ReplayVerifierChallengeTest(unittest.TestCase):
    def test_comparison_labels_use_declared_epsilon(self):
        self.assertEqual(comparison_label(0.01, epsilon=0.001), "better")
        self.assertEqual(comparison_label(-0.01, epsilon=0.001), "worse")
        self.assertEqual(comparison_label(0.0005, epsilon=0.001), "unchanged")

    def test_failed_replay_leaves_no_output_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "replay"

            def fail(_temporary):
                raise RuntimeError("model load failed")

            with self.assertRaisesRegex(RuntimeError, "model load failed"):
                atomic_output(output, fail)

            self.assertFalse(output.exists())
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_successful_replay_commits_only_after_writer_returns(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "replay"

            def write_success(temporary):
                (temporary / "report.json").write_text("{}")

            atomic_output(output, write_success)

            self.assertTrue((output / "report.json").exists())


if __name__ == "__main__":
    unittest.main()
