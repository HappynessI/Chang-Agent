import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from change_agent.adapters.subprocess_adapters import SubprocessSAM3Initializer


class Completed:
    returncode = 0
    stdout = "ok"
    stderr = ""


class SubprocessInitializationTest(unittest.TestCase):
    def test_fresh_initializer_loads_masks_and_persisted_confidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initializer = SubprocessSAM3Initializer(
                "/usr/bin/python3",
                root / "worker.py",
                root / "artifacts",
                checkpoint=root / "sam3.pt",
                bpe=root / "bpe.gz",
            )

            def fake_run(command, **kwargs):
                def value(flag):
                    return Path(command[command.index(flag) + 1])

                t1_mask = np.zeros((8, 8), dtype=np.uint8)
                t2_mask = np.ones((8, 8), dtype=np.uint8)
                np.save(value("--output-mask"), t1_mask)
                np.save(value("--output-mask-t2"), t2_mask)
                evidence_dir = value("--evidence-dir")
                evidence_dir.mkdir(parents=True)
                t1_conf = evidence_dir / "t1_confidence_map.npy"
                t2_conf = evidence_dir / "t2_confidence_map.npy"
                np.save(t1_conf, np.full((8, 8), 0.2, dtype=np.float32))
                np.save(t2_conf, np.full((8, 8), 0.7, dtype=np.float32))
                report = {
                    "status": "success",
                    "intermediate_artifacts": {
                        "t1": {"confidence_map": {"file": str(t1_conf)}},
                        "t2": {"confidence_map": {"file": str(t2_conf)}},
                    },
                }
                value("--report").write_text(json.dumps(report), encoding="utf-8")
                return Completed()

            image = np.zeros((8, 8, 3), dtype=np.uint8)
            with patch("change_agent.adapters.subprocess_adapters.subprocess.run", fake_run):
                t1, t2, evidence = initializer.initialize_masks(image, image, "building")
            self.assertFalse(t1.any())
            self.assertTrue(t2.all())
            self.assertTrue(np.allclose(evidence["change_confidence"], 0.7))
            self.assertEqual(evidence["initializer"], "live_sam3_dual_view_text_prompt")


if __name__ == "__main__":
    unittest.main()
