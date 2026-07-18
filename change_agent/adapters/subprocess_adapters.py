"""Real-model tool adapters executed in isolated Python environments."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


class _SubprocessTool:
    def __init__(
        self,
        python: str | Path,
        worker: str | Path,
        artifact_dir: str | Path,
        *,
        pythonpath: Iterable[str | Path] = (),
        device: str = "cuda",
        timeout_seconds: int = 300,
    ):
        self.python = str(Path(python).resolve())
        self.worker = str(Path(worker).resolve())
        self.artifact_dir = Path(artifact_dir)
        self.pythonpath = tuple(str(Path(item).resolve()) for item in pythonpath)
        self.device = device
        self.timeout_seconds = timeout_seconds
        self.call_index = 0
        self.last_evidence: dict[str, Any] = {}

    def _call(
        self,
        mode: str,
        image: np.ndarray,
        initial_mask: np.ndarray | None,
        extra_args: list[str],
    ) -> np.ndarray:
        call_dir = self.artifact_dir / f"{mode}_{self.call_index:03d}"
        self.call_index += 1
        call_dir.mkdir(parents=True, exist_ok=False)
        image_path = call_dir / "image.png"
        mask_path = call_dir / "initial_mask.npy"
        output_path = call_dir / "output_mask.npy"
        report_path = call_dir / "report.json"
        stdout_path = call_dir / "stdout.log"
        stderr_path = call_dir / "stderr.log"
        Image.fromarray(_uint8_image(image)).save(image_path)
        if initial_mask is not None:
            np.save(mask_path, np.asarray(initial_mask, dtype=np.uint8))

        command = [
            self.python,
            self.worker,
            mode,
            "--image",
            str(image_path),
            "--output-mask",
            str(output_path),
            "--report",
            str(report_path),
            "--device",
            self.device,
        ]
        if initial_mask is not None:
            command.extend(["--initial-mask", str(mask_path)])
        command.extend(extra_args)
        env = os.environ.copy()
        if self.pythonpath:
            existing = env.get("PYTHONPATH")
            env["PYTHONPATH"] = os.pathsep.join(
                self.pythonpath + ((existing,) if existing else ())
            )
        result = subprocess.run(
            command,
            cwd=str(Path(self.worker).parent),
            env=env,
            text=True,
            capture_output=True,
            timeout=self.timeout_seconds,
        )
        stdout_path.write_text(result.stdout, encoding="utf-8")
        stderr_path.write_text(result.stderr, encoding="utf-8")
        if result.returncode != 0:
            raise RuntimeError(
                f"{mode} worker failed with exit={result.returncode}; see {stderr_path}"
            )
        if not output_path.is_file() or not report_path.is_file():
            raise RuntimeError(f"{mode} worker did not produce its declared artifacts")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.last_evidence = {
            "worker_mode": mode,
            "worker_command": command,
            "worker_report": report,
            "worker_artifact_dir": str(call_dir),
        }
        return np.asarray(np.load(output_path), dtype=bool)


class SubprocessPointBackend(_SubprocessTool):
    def __init__(self, *args: Any, checkpoint: str | Path, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.checkpoint = str(Path(checkpoint).resolve())

    def refine(
        self,
        image: np.ndarray,
        initial_mask: np.ndarray,
        coordinate: tuple[int, int],
        is_positive: bool,
    ) -> np.ndarray:
        x, y = coordinate
        return self._call(
            "point",
            image,
            initial_mask,
            [
                "--checkpoint",
                self.checkpoint,
                "--coordinate",
                str(x),
                str(y),
                "--is-positive",
                "1" if is_positive else "0",
            ],
        )


class SubprocessBoxBackend(_SubprocessTool):
    def __init__(
        self,
        *args: Any,
        checkpoint: str | Path,
        bpe: str | Path,
        resolution: int = 1008,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.checkpoint = str(Path(checkpoint).resolve())
        self.bpe = str(Path(bpe).resolve())
        self.resolution = resolution

    def segment_box(
        self,
        image: np.ndarray,
        box_cxcywh_normalized: tuple[float, float, float, float],
        query: str,
    ) -> np.ndarray:
        return self._call(
            "box",
            image,
            None,
            [
                "--checkpoint",
                self.checkpoint,
                "--bpe",
                self.bpe,
                "--resolution",
                str(self.resolution),
                "--query",
                query,
                "--box",
                *(str(value) for value in box_cxcywh_normalized),
            ],
        )


def _uint8_image(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype != np.uint8:
        if array.max(initial=0) <= 1:
            array = array * 255
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array
