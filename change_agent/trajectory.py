"""Trajectory recording, best-state tracking, and reproducibility artifacts."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .state import AgentAction, ChangeState, VerifierOutput


@dataclass
class TrajectoryEntry:
    step_index: int
    raw_action: str | None
    parsed_action: AgentAction | None
    verifier: VerifierOutput
    state: ChangeState
    execution: dict[str, Any]

    def to_dict(self, mask_file: str | None = None) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "raw_action": self.raw_action,
            "parsed_action": self.parsed_action.to_dict() if self.parsed_action else None,
            "verifier": self.verifier.to_dict(),
            "execution": _json_safe(self.execution),
            "change_mask_file": mask_file,
        }


class Trajectory:
    def __init__(self, run_metadata: dict[str, Any] | None = None):
        self.entries: list[TrajectoryEntry] = []
        self.run_metadata = default_run_metadata()
        self.run_metadata.update(run_metadata or {})

    def append(self, entry: TrajectoryEntry) -> None:
        if self.entries and entry.step_index <= self.entries[-1].step_index:
            raise ValueError("trajectory step indices must be strictly increasing")
        self.entries.append(entry)

    @property
    def best_entry(self) -> TrajectoryEntry:
        if not self.entries:
            raise RuntimeError("trajectory is empty")
        return max(self.entries, key=lambda item: item.verifier.quality_score)

    def history_summary(self, limit: int = 4) -> str:
        recent = self.entries[-limit:]
        parts = []
        for item in recent:
            action = item.parsed_action.action if item.parsed_action else "reset"
            parts.append(
                f"step={item.step_index}, action={action}, "
                f"score={item.verifier.quality_score:.3f}, error={item.verifier.error_type}"
            )
        return "; ".join(parts)

    def save(self, output_dir: str | Path) -> Path:
        output_dir = Path(output_dir)
        masks_dir = output_dir / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)
        serialized = []
        for entry in self.entries:
            relative = Path("masks") / f"step_{entry.step_index:03d}.npy"
            np.save(output_dir / relative, entry.state.change_mask.astype(np.uint8))
            serialized.append(entry.to_dict(str(relative)))
        payload = {
            "metadata": _json_safe(self.run_metadata),
            "best_step": self.best_entry.step_index if self.entries else None,
            "steps": serialized,
        }
        path = output_dir / "trajectory.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path


def default_run_metadata() -> dict[str, Any]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)

