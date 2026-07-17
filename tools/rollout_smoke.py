#!/usr/bin/env python3
"""Run the adapter → Environment → Verifier feedback loop with memory telemetry."""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np

from change_agent.adapters.omniovcd_adapter import OmniOVCDAdapter
from change_agent.agent import ScriptedAgent
from change_agent.environment import ChangeAgentEnvironment
from change_agent.executor import ActionExecutor
from change_agent.runner import ChangeAgentRunner
from change_agent.state import AgentAction
from change_agent.verifier import RuleBasedVerifier


def rss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / 1024.0 if value > 10_000 else value / (1024.0 * 1024.0)


class MockPointBackend:
    def refine(self, image, initial_mask, coordinate, is_positive):
        result = initial_mask.copy()
        x, y = coordinate
        result[y, x] = is_positive
        return result


def initialize_masks(t1, t2, query):
    t1_mask = np.zeros(t1.shape[:2], dtype=bool)
    t2_mask = np.zeros_like(t1_mask)
    t1_mask[5:13, 5:13] = True
    return t1_mask, t2_mask, {"change_confidence": np.full(t1_mask.shape, 0.8)}


def segment_box(image, box_cxcywh, query):
    height, width = image.shape[:2]
    cx, cy, box_width, box_height = box_cxcywh
    x1, x2 = round((cx - box_width / 2) * width), round((cx + box_width / 2) * width)
    y1, y2 = round((cy - box_height / 2) * height), round((cy + box_height / 2) * height)
    result = np.zeros((height, width), dtype=bool)
    result[max(0, y1) : min(height, y2), max(0, x1) : min(width, x2)] = True
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("/tmp/change_agent_rollout_smoke"))
    args = parser.parse_args()
    start = time.monotonic()
    backend = OmniOVCDAdapter(initialize_masks, segment_box)
    environment = ChangeAgentEnvironment(
        backend,
        ActionExecutor(MockPointBackend(), backend),
        RuleBasedVerifier(accept_threshold=0.6, min_change_ratio=0.0),
        max_steps=3,
        run_metadata={"dataset_split": "synthetic-adapter-smoke", "seed": 42},
    )
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    observation = environment.reset(image, image.copy(), "building change")
    agent = ScriptedAgent(
        [AgentAction("t2", "box", box=(5, 5, 13, 13)), AgentAction("t2", "finish")]
    )
    best = ChangeAgentRunner(environment, agent).run(observation)
    trajectory_path = environment.trajectory.save(args.output)
    result = {
        "steps": len(environment.trajectory.entries) - 1,
        "best_step": environment.trajectory.best_entry.step_index,
        "best_change_pixels": int(best.change_mask.sum()),
        "elapsed_seconds": round(time.monotonic() - start, 3),
        "rss_peak_mb": round(rss_mb(), 2),
        "cuda_available": False,
        "trajectory": str(trajectory_path),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

