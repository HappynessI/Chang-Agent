#!/usr/bin/env python3
"""Replay saved candidates through the GT-free Verifier, then score them offline.

GT is opened only after each Verifier response has been generated. The output
directory is committed atomically, so a model/configuration failure leaves no
partial result under ``outputs/``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from change_agent.adapters.omniovcd_adapter import MaskPairProcessor
from change_agent.adapters.qwen3vl_adapter import GroundingModelQwen3VL
from change_agent.adapters.qwen3vl_verifier import Qwen3VLZeroShotVerifier
from change_agent.state import AgentAction, ChangeState
from change_agent.verifier_regions import attach_verifier_regions


SAMPLES = ("test_20_15.png", "test_78_13.png", "test_85_16.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query", default="building")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--verifier-retries", type=int, default=2)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--comparison-epsilon", type=float, default=1e-6)
    parser.add_argument("--max-regions", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
    if not args.run_dir.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {args.run_dir}")

    def build(temp_output: Path) -> None:
        qwen = GroundingModelQwen3VL(
            args.model_path,
            device_map=args.device_map,
            max_new_tokens=args.max_new_tokens,
        )
        verifier = Qwen3VLZeroShotVerifier(
            model=qwen.model,
            processor=qwen.processor,
            max_new_tokens=args.max_new_tokens,
            max_retries=args.verifier_retries,
        )
        result = replay_run(
            args.run_dir,
            args.input_root,
            verifier,
            query=args.query,
            comparison_epsilon=args.comparison_epsilon,
            max_regions=args.max_regions,
        )
        temp_output.mkdir(parents=True, exist_ok=True)
        (temp_output / "replay_report.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    atomic_output(args.output, build)
    print(json.dumps({"status": "success", "output": str(args.output)}, indent=2))


def replay_run(
    run_dir: Path,
    input_root: Path,
    verifier: Any,
    *,
    query: str = "building",
    comparison_epsilon: float = 1e-6,
    max_regions: int = 6,
) -> dict[str, Any]:
    """Replay all saved tool candidates without exposing GT to ``verifier``."""

    if comparison_epsilon < 0:
        raise ValueError("comparison_epsilon must be non-negative")
    metrics = json.loads((run_dir / "per_sample_metrics.json").read_text())
    samples: dict[str, Any] = {}
    for sample_file in SAMPLES:
        sample = Path(sample_file).stem
        trajectory_path = run_dir / "trajectories" / sample / "trajectory.json"
        trajectory = json.loads(trajectory_path.read_text())
        image1 = np.asarray(Image.open(input_root / "A" / sample_file).convert("RGB"))
        image2 = np.asarray(Image.open(input_root / "B" / sample_file).convert("RGB"))
        gt = np.asarray(Image.open(input_root / "label_cvt" / sample_file)) > 0
        initial_t1, initial_t2 = _initial_masks(run_dir, sample)
        processor = MaskPairProcessor()
        initial = _make_state(
            image1,
            image2,
            query,
            initial_t1,
            initial_t2,
            processor,
        )
        initial_iou = _iou(initial.change_mask, gt)
        entries: list[dict[str, Any]] = []
        for step in trajectory["steps"][1:]:
            if not step.get("tool") or not step.get("parsed_action"):
                continue
            candidate_t1, candidate_t2 = _candidate_masks(
                run_dir, sample, step, initial_t1, initial_t2
            )
            candidate = _make_state(
                image1,
                image2,
                query,
                candidate_t1,
                candidate_t2,
                processor,
            )
            saved_change_path = (trajectory_path.parent / step["change_mask_file"]).resolve()
            saved_change = np.asarray(np.load(saved_change_path), dtype=bool)
            if saved_change.shape != candidate.change_mask.shape:
                raise ValueError(
                    f"saved candidate mask shape {saved_change.shape} does not match "
                    f"reconstructed state {candidate.change_mask.shape}"
                )
            # Preserve the exact candidate presented in the original trajectory;
            # T1/T2 masks are reconstructed from tool artifacts for the local
            # temporal views, while this field is the persisted final mask.
            candidate.change_mask = saved_change
            # The previous state is the accepted initial state for this historical
            # run: every saved tool candidate was rejected and rolled back.
            verifier.reset()
            attach_verifier_regions(candidate, initial, max_regions=max_regions)
            action = _action_from_dict(step["parsed_action"])
            output = verifier.verify(candidate, None, action, initial)

            # GT is consulted only here, after the verifier has returned.
            candidate_iou = _iou(candidate.change_mask, gt)
            expected = comparison_label(
                candidate_iou - initial_iou, epsilon=comparison_epsilon
            )
            entries.append(
                {
                    "step_index": step["step_index"],
                    "candidate_accepted_in_original_run": step.get("candidate_accepted"),
                    "verifier_output": output.to_dict(),
                    "verifier_evidence": getattr(verifier, "last_evidence", {}),
                    "offline_after_verifier": {
                        "initial_iou": initial_iou,
                        "candidate_iou": candidate_iou,
                        "expected_comparison": expected,
                        "comparison_match": output.comparison == expected,
                    },
                }
            )
        samples[sample] = {
            "initial_iou": initial_iou,
            "original_verifier_best_step": trajectory.get("verifier_best_step"),
            "entries": entries,
        }
    summary = _summary(samples)
    return {
        "decision_mode": "region_classification_then_categorical_pairwise",
        "gt_policy": "GT opened only after verifier output for each candidate",
        "run_dir": str(run_dir),
        "samples": samples,
        "summary": summary,
        "source_metrics": metrics,
    }


def atomic_output(output: Path, writer: Callable[[Path], None]) -> None:
    """Commit a generated output directory only when ``writer`` succeeds."""

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        writer(temporary)
        if output.exists():
            raise FileExistsError(f"output appeared during run: {output}")
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def comparison_label(delta: float, *, epsilon: float = 1e-6) -> str:
    if delta > epsilon:
        return "better"
    if delta < -epsilon:
        return "worse"
    return "unchanged"


def _initial_masks(run_dir: Path, sample: str) -> tuple[np.ndarray, np.ndarray]:
    directory = run_dir / "tool_runs" / sample / "sam3_initialization" / "initialize_000"
    return (
        np.asarray(np.load(directory / "t1_mask.npy"), dtype=bool),
        np.asarray(np.load(directory / "t2_mask.npy"), dtype=bool),
    )


def _candidate_masks(
    run_dir: Path,
    sample: str,
    step: dict[str, Any],
    initial_t1: np.ndarray,
    initial_t2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    t1 = np.array(initial_t1, dtype=bool, copy=True)
    t2 = np.array(initial_t2, dtype=bool, copy=True)
    parsed = step["parsed_action"]
    target = parsed["target_view"]
    worker_command = step.get("execution", {}).get("tool_result", {}).get("worker_command", [])
    output_mask = _argument_value(worker_command, "--output-mask")
    if output_mask is None:
        # Older artifacts retain the same deterministic point directory layout.
        output_mask = str(
            run_dir / "tool_runs" / sample / "point_{:03d}".format(step["step_index"] - 1) / "output_mask.npy"
        )
    result = np.asarray(np.load(output_mask), dtype=bool)
    if target == "t1":
        t1 = result
    else:
        t2 = result
    return t1, t2


def _make_state(
    image1: np.ndarray,
    image2: np.ndarray,
    query: str,
    t1_mask: np.ndarray,
    t2_mask: np.ndarray,
    processor: MaskPairProcessor,
) -> ChangeState:
    update = processor.rebuild(t1_mask, t2_mask)
    return ChangeState(
        image1,
        image2,
        query,
        t1_mask,
        t2_mask,
        update.change_mask,
        t1_instances=update.t1_instances,
        t2_instances=update.t2_instances,
        matching=update.matching,
        evidence=update.evidence,
    )


def _action_from_dict(value: dict[str, Any]) -> AgentAction:
    return AgentAction(
        target_view=value["target_view"],
        action=value["action"],
        coordinate=tuple(value["coordinate"]) if value.get("coordinate") else None,
        box=tuple(value["box"]) if value.get("box") else None,
    )


def _argument_value(command: list[str], name: str) -> str | None:
    try:
        return command[command.index(name) + 1]
    except (ValueError, IndexError):
        return None


def _iou(prediction: np.ndarray, target: np.ndarray) -> float:
    intersection = int(np.logical_and(prediction, target).sum())
    union = int(np.logical_or(prediction, target).sum())
    return intersection / union if union else 1.0


def _summary(samples: dict[str, Any]) -> dict[str, Any]:
    rows = [
        entry["offline_after_verifier"]
        for sample in samples.values()
        for entry in sample["entries"]
    ]
    return {
        "candidate_count": len(rows),
        "comparison_match_count": sum(bool(row["comparison_match"]) for row in rows),
        "comparison_accuracy": (
            sum(bool(row["comparison_match"]) for row in rows) / len(rows)
            if rows
            else None
        ),
    }


if __name__ == "__main__":
    main()
