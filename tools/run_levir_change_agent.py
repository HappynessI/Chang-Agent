#!/usr/bin/env python3
"""Run the real Qwen → Environment → segmentation-tool loop on three LEVIR samples."""

from __future__ import annotations

import argparse
import json
import random
import resource
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from change_agent.action_parser import ActionValidationError
from change_agent.adapters.omniovcd_adapter import MaskPairProcessor, OmniOVCDAdapter
from change_agent.adapters.qwen3vl_adapter import GroundingModelQwen3VL
from change_agent.adapters.subprocess_adapters import (
    SubprocessBoxBackend,
    SubprocessPointBackend,
)
from change_agent.environment import ChangeAgentEnvironment
from change_agent.executor import ActionExecutor
from change_agent.verifier import RuleBasedVerifier


SAMPLES = ("test_20_15.png", "test_78_13.png", "test_85_16.png")


def parse_args() -> argparse.Namespace:
    root = Path("/Data/wyh/CD-SegAgent")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        default=root / "outputs/omniovcd_levir_3sample_vis_20260717_145327/input_subset",
    )
    parser.add_argument(
        "--initial-mask-root",
        type=Path,
        default=root / "outputs/omniovcd_levir_3sample_vis_20260717_145327/model_visualizations",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query", default="building")
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--matching-mode", choices=sorted(MaskPairProcessor.MODES), default="overlap_presence")
    parser.add_argument("--overlap-threshold", type=float, default=0.25)
    parser.add_argument("--t12-min-instance-area", type=int, default=0)
    parser.add_argument("--cd-min-instance-area", type=int, default=0)
    parser.add_argument("--model-path", default=str(root / "models/Qwen3-VL-2B-Instruct"))
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--tool-device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--action-retries", type=int, default=3)
    parser.add_argument("--segagent-python", default=str(root / "segagent-env/bin/python"))
    parser.add_argument("--omniovcd-python", default=str(root / "omniovcd-env/bin/python"))
    parser.add_argument("--simpleclick-checkpoint", default=str(root / "models/SimpleClick/cocolvis_vit_large.pth"))
    parser.add_argument("--sam3-checkpoint", default=str(root / "models/sam3/sam3.pt"))
    parser.add_argument("--sam3-bpe", default=str(root / "OmniOVCD/sam3/assets/bpe_simple_vocab_16e6.txt.gz"))
    parser.add_argument("--sam3-resolution", type=int, default=1008)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.monotonic()
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ("predictions", "trajectories", "masks", "verifier_feedback", "logs", "tool_runs"):
        (args.output / name).mkdir()
    _validate_inputs(args)
    manifest = _base_manifest(args)
    (args.output / "run_manifest.md").write_text(_render_manifest(manifest), encoding="utf-8")

    qwen = GroundingModelQwen3VL(
        args.model_path,
        device_map=args.device_map,
        max_new_tokens=args.max_new_tokens,
    )
    rollout_records: dict[str, dict[str, Any]] = {}
    for sample_file in SAMPLES:
        sample = Path(sample_file).stem
        image1 = np.asarray(Image.open(args.input_root / "A" / sample_file).convert("RGB"))
        image2 = np.asarray(Image.open(args.input_root / "B" / sample_file).convert("RGB"))
        tool_root = args.output / "tool_runs" / sample
        point_backend = SubprocessPointBackend(
            args.segagent_python,
            Path(__file__).with_name("segmentation_worker.py"),
            tool_root,
            checkpoint=args.simpleclick_checkpoint,
            pythonpath=(
                Path(__file__).parents[1],
                Path(__file__).parents[2] / "SegAgent",
                Path(__file__).parents[2] / "SegAgent/third_party/SimpleClick",
            ),
            device=args.tool_device,
        )
        box_backend = SubprocessBoxBackend(
            args.omniovcd_python,
            Path(__file__).with_name("segmentation_worker.py"),
            tool_root,
            checkpoint=args.sam3_checkpoint,
            bpe=args.sam3_bpe,
            resolution=args.sam3_resolution,
            pythonpath=(Path(__file__).parents[1], Path(__file__).parents[2] / "OmniOVCD"),
            device=args.tool_device,
        )
        processor = MaskPairProcessor(
            overlap_threshold=args.overlap_threshold,
            matching_mode=args.matching_mode,
            t12_min_instance_area=args.t12_min_instance_area,
            cd_min_instance_area=args.cd_min_instance_area,
        )
        initializer = _cached_initializer(args, sample_file)
        backend = OmniOVCDAdapter(initializer, box_backend.segment_box, processor)
        environment = ChangeAgentEnvironment(
            backend,
            ActionExecutor(point_backend, box_backend),
            RuleBasedVerifier(),
            max_steps=args.max_steps,
            run_metadata={
                "sample": sample_file,
                "query": args.query,
                "agent": "Qwen3-VL-2B-Instruct",
                "verifier": "RuleBasedVerifier",
                "matching_mode": args.matching_mode,
                "overlap_threshold": args.overlap_threshold,
                "gt_available_during_rollout": False,
            },
        )
        observation = environment.reset(image1, image2, args.query)
        invalid_outputs: list[dict[str, str]] = []
        while not environment.done:
            for attempt in range(args.action_retries):
                raw = qwen.generate_raw(observation)
                try:
                    observation, _ = environment.step(raw)
                    break
                except ActionValidationError as error:
                    invalid_outputs.append({"raw_agent_output": raw, "error": str(error)})
                    if attempt + 1 == args.action_retries:
                        raise
        tool_steps = [entry for entry in environment.trajectory.entries if entry.execution.get("tool")]
        if not tool_steps:
            raise RuntimeError(f"{sample_file}: Qwen produced no executable segmentation action")

        trajectory_dir = args.output / "trajectories" / sample
        trajectory_path = environment.trajectory.save(
            trajectory_dir, args.output / "masks" / sample
        )
        feedback_path = args.output / "verifier_feedback" / f"{sample}.json"
        feedback_path.write_text(
            json.dumps(
                [entry.verifier.to_dict() for entry in environment.trajectory.entries],
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        if invalid_outputs:
            (trajectory_dir / "invalid_agent_outputs.json").write_text(
                json.dumps(invalid_outputs, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        best = environment.best_state.change_mask
        Image.fromarray(best.astype(np.uint8) * 255, mode="L").save(
            args.output / "predictions" / sample_file
        )
        rollout_records[sample] = {
            "trajectory": trajectory_path,
            "best_step": environment.trajectory.best_entry.step_index,
            "tool_action_count": len(tool_steps),
            "raw_actions": [entry.raw_action for entry in environment.trajectory.entries[1:]],
        }

    # This marker is written before label_cvt is ever opened. Everything above is the
    # GT-free runtime path; only the offline evaluator below receives the GT directory.
    rollout_marker = args.output / "rollout_complete.json"
    rollout_marker.write_text(
        json.dumps({
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "samples": sorted(rollout_records),
            "gt_loaded": False,
        }, indent=2),
        encoding="utf-8",
    )
    metrics = evaluate_after_rollout(rollout_records, args.input_root / "label_cvt")
    (args.output / "per_sample_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    manifest.update(
        status="success",
        elapsed_seconds=round(time.monotonic() - start, 3),
        rollout_marker=str(rollout_marker),
        per_sample_metrics=str(args.output / "per_sample_metrics.json"),
        rss_peak_mb=round(_rss_mb(), 2),
    )
    (args.output / "run_manifest.md").write_text(_render_manifest(manifest, metrics), encoding="utf-8")
    print(json.dumps({"status": "success", "output": str(args.output), **metrics["aggregate"]}, indent=2))


def _cached_initializer(args: argparse.Namespace, sample_file: str):
    t1_path = args.initial_mask_root / "t1" / sample_file
    t2_path = args.initial_mask_root / "t2" / sample_file

    def initialize(t1_image: np.ndarray, t2_image: np.ndarray, query: str):
        t1_mask = np.asarray(Image.open(t1_path)) > 0
        t2_mask = np.asarray(Image.open(t2_path)) > 0
        if t1_mask.shape != t1_image.shape[:2] or t2_mask.shape != t2_image.shape[:2]:
            raise ValueError("cached OmniOVCD masks do not match input image size")
        return t1_mask, t2_mask, {
            "initializer": "OmniOVCDAdapter.cached_real_omniovcd_masks",
            "initial_t1_mask": str(t1_path),
            "initial_t2_mask": str(t2_path),
            "query": query,
            "target_view_hint": "t2",
        }

    return initialize


def evaluate_after_rollout(
    rollout_records: dict[str, dict[str, Any]], gt_dir: Path
) -> dict[str, Any]:
    samples: dict[str, Any] = {}
    aggregate_counts = np.zeros(4, dtype=np.int64)
    for sample, record in rollout_records.items():
        trajectory_path = Path(record["trajectory"])
        payload = json.loads(trajectory_path.read_text(encoding="utf-8"))
        gt = np.asarray(Image.open(gt_dir / f"{sample}.png")) > 0
        step_metrics = []
        for step in payload["steps"]:
            mask_path = (trajectory_path.parent / step["change_mask_file"]).resolve()
            prediction = np.asarray(np.load(mask_path), dtype=bool)
            values, counts = _metrics(prediction, gt)
            step["offline_metrics"] = values
            step_metrics.append({"step_index": step["step_index"], **values})
            if step["step_index"] == payload["best_step"]:
                best_metrics = values
                aggregate_counts += counts
        trajectory_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        samples[sample] = {
            "best_step": payload["best_step"],
            "tool_action_count": record["tool_action_count"],
            "steps": step_metrics,
            "initial": step_metrics[0],
            "best": {"step_index": payload["best_step"], **best_metrics},
        }
    aggregate, _ = _metrics_from_counts(aggregate_counts)
    return {"samples": samples, "aggregate": aggregate}


def _metrics(prediction: np.ndarray, gt: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    if prediction.shape != gt.shape:
        raise ValueError(f"prediction shape {prediction.shape} != GT shape {gt.shape}")
    counts = np.asarray([
        np.logical_and(prediction, gt).sum(),
        np.logical_and(prediction, ~gt).sum(),
        np.logical_and(~prediction, gt).sum(),
        np.logical_and(~prediction, ~gt).sum(),
    ], dtype=np.int64)
    return _metrics_from_counts(counts)


def _metrics_from_counts(counts: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    tp, fp, fn, tn = (int(value) for value in counts)
    precision = tp / (tp + fp) if tp + fp else 1.0 if tp + fn == 0 else 0.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "iou": round(iou, 8),
        "precision": round(precision, 8),
        "recall": round(recall, 8),
        "f1": round(f1, 8),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }, counts


def _validate_inputs(args: argparse.Namespace) -> None:
    files = [Path(args.model_path), Path(args.simpleclick_checkpoint), Path(args.sam3_checkpoint), Path(args.sam3_bpe)]
    for sample in SAMPLES:
        files.extend([
            args.input_root / "A" / sample,
            args.input_root / "B" / sample,
            args.initial_mask_root / "t1" / sample,
            args.initial_mask_root / "t2" / sample,
        ])
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing required inputs: {missing}")


def _base_manifest(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "samples": list(SAMPLES),
        "query": args.query,
        "input_root": str(args.input_root),
        "initial_mask_root": str(args.initial_mask_root),
        "initialization": "cached masks from prior real OmniOVCD/SAM3 inference, loaded through OmniOVCDAdapter",
        "agent_model": args.model_path,
        "point_executor": f"SimpleClick {args.simpleclick_checkpoint}",
        "box_executor": f"SAM3 {args.sam3_checkpoint}",
        "verifier": "RuleBasedVerifier",
        "matching_mode": args.matching_mode,
        "overlap_threshold": args.overlap_threshold,
        "t12_min_instance_area": args.t12_min_instance_area,
        "cd_min_instance_area": args.cd_min_instance_area,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "gt_policy": "label_cvt is opened only by evaluate_after_rollout after rollout_complete.json",
    }


def _render_manifest(manifest: dict[str, Any], metrics: dict[str, Any] | None = None) -> str:
    lines = ["# LEVIR-CD three-sample full Change-Agent run", ""]
    lines.extend(f"- {key}: `{value}`" for key, value in manifest.items())
    if metrics:
        lines.extend(["", "## Aggregate best-mask metrics", ""])
        lines.extend(f"- {key}: `{value}`" for key, value in metrics["aggregate"].items())
    lines.extend([
        "",
        "## Runtime boundary",
        "",
        "The final entry point is `tools/run_levir_change_agent.py`, not OmniOVCD/eval.py. ",
        "Each sample executes Environment.reset, Qwen action generation, ActionParser, a real ",
        "SimpleClick or SAM3 worker, Environment.step/rebuild, and RuleBasedVerifier feedback. ",
        "Ground truth is loaded only after all trajectories and rollout_complete.json exist.",
    ])
    return "\n".join(lines) + "\n"


def _rss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / 1024.0 if value > 10_000 else value / (1024.0 * 1024.0)


if __name__ == "__main__":
    main()
