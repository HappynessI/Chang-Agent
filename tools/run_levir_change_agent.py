#!/usr/bin/env python3
"""Run the real Qwen → Environment → segmentation-tool loop on three LEVIR samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from change_agent.action_parser import ActionValidationError
from change_agent.adapters.omniovcd_adapter import MaskPairProcessor, OmniOVCDAdapter
from change_agent.adapters.qwen3vl_adapter import GroundingModelQwen3VL
from change_agent.adapters.qwen3vl_verifier import Qwen3VLZeroShotVerifier
from change_agent.adapters.subprocess_adapters import (
    SubprocessBoxBackend,
    SubprocessPointBackend,
    SubprocessSAM3Initializer,
)
from change_agent.environment import ChangeAgentEnvironment
from change_agent.executor import ActionExecutor
from change_agent.state import AgentObservation
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query", default="building")
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument(
        "--selection-policy",
        choices=("verifier_best", "conservative_best", "initial"),
        default="conservative_best",
    )
    parser.add_argument("--selection-epsilon", type=float, default=0.0)
    parser.add_argument("--max-selection-area-delta", type=float, default=0.25)
    parser.add_argument("--max-locality-outside-ratio", type=float, default=0.1)
    parser.add_argument("--max-target-mask-change-ratio", type=float, default=0.25)
    parser.add_argument("--max-component-count-delta", type=int, default=4)
    parser.add_argument("--verifier-max-regions", type=int, default=6)
    parser.add_argument("--verifier-min-region-area", type=int, default=4)
    parser.add_argument("--verifier-region-padding-ratio", type=float, default=0.25)
    parser.add_argument("--matching-mode", choices=sorted(MaskPairProcessor.MODES), default="overlap_presence")
    parser.add_argument("--overlap-threshold", type=float, default=0.25)
    parser.add_argument("--t12-min-instance-area", type=int, default=0)
    parser.add_argument("--cd-min-instance-area", type=int, default=0)
    parser.add_argument("--model-path", default=str(root / "models/Qwen3-VL-2B-Instruct"))
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--verifier", choices=("qwen_zero_shot", "rule"), default="qwen_zero_shot"
    )
    parser.add_argument("--verifier-max-new-tokens", type=int, default=512)
    parser.add_argument("--verifier-accept-threshold", type=float, default=0.82)
    parser.add_argument("--verifier-retries", type=int, default=2)
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
    seed_runtime = _seed_runtime(args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ("predictions", "trajectories", "masks", "verifier_feedback", "logs", "tool_runs"):
        (args.output / name).mkdir()
    for name in ("initial", "verifier_best", "last", "selected"):
        (args.output / "predictions" / name).mkdir()
    _validate_inputs(args)
    manifest = _base_manifest(args, seed_runtime)
    (args.output / "run_manifest.md").write_text(_render_manifest(manifest), encoding="utf-8")

    qwen = GroundingModelQwen3VL(
        args.model_path,
        device_map=args.device_map,
        max_new_tokens=args.max_new_tokens,
    )
    verifier = _build_verifier(args, qwen)
    rollout_records: dict[str, dict[str, Any]] = {}
    for sample_file in SAMPLES:
        sample_start = time.monotonic()
        sample = Path(sample_file).stem
        image1 = np.asarray(Image.open(args.input_root / "A" / sample_file).convert("RGB"))
        image2 = np.asarray(Image.open(args.input_root / "B" / sample_file).convert("RGB"))
        tool_root = args.output / "tool_runs" / sample
        point_backend = SubprocessPointBackend(
            args.segagent_python,
            Path(__file__).with_name("seeded_segmentation_worker.py"),
            tool_root,
            checkpoint=args.simpleclick_checkpoint,
            pythonpath=(
                Path(__file__).parents[1],
                Path(__file__).parents[2] / "SegAgent",
                Path(__file__).parents[2] / "SegAgent/third_party/SimpleClick",
            ),
            device=args.tool_device,
            seed=args.seed,
        )
        box_backend = SubprocessBoxBackend(
            args.omniovcd_python,
            Path(__file__).with_name("seeded_segmentation_worker.py"),
            tool_root,
            checkpoint=args.sam3_checkpoint,
            bpe=args.sam3_bpe,
            resolution=args.sam3_resolution,
            pythonpath=(Path(__file__).parents[1], Path(__file__).parents[2] / "OmniOVCD"),
            device=args.tool_device,
            seed=args.seed,
        )
        initializer = SubprocessSAM3Initializer(
            args.omniovcd_python,
            Path(__file__).with_name("seeded_segmentation_worker.py"),
            tool_root / "sam3_initialization",
            checkpoint=args.sam3_checkpoint,
            bpe=args.sam3_bpe,
            resolution=args.sam3_resolution,
            pythonpath=(Path(__file__).parents[1], Path(__file__).parents[2] / "OmniOVCD"),
            device=args.tool_device,
            timeout_seconds=900,
            seed=args.seed,
        )
        processor = MaskPairProcessor(
            overlap_threshold=args.overlap_threshold,
            matching_mode=args.matching_mode,
            t12_min_instance_area=args.t12_min_instance_area,
            cd_min_instance_area=args.cd_min_instance_area,
        )
        backend = OmniOVCDAdapter(
            initializer.initialize_masks, box_backend.segment_box, processor
        )
        environment = ChangeAgentEnvironment(
            backend,
            ActionExecutor(point_backend, box_backend),
            verifier,
            max_steps=args.max_steps,
            selection_policy=args.selection_policy,
            selection_epsilon=args.selection_epsilon,
            max_selection_area_delta=args.max_selection_area_delta,
            max_locality_outside_ratio=args.max_locality_outside_ratio,
            max_target_mask_change_ratio=args.max_target_mask_change_ratio,
            max_component_count_delta=args.max_component_count_delta,
            verifier_max_regions=args.verifier_max_regions,
            verifier_min_region_area=args.verifier_min_region_area,
            verifier_region_padding_ratio=args.verifier_region_padding_ratio,
            run_metadata={
                "sample": sample_file,
                "query": args.query,
                "agent": "Qwen3-VL-2B-Instruct",
                "verifier": args.verifier,
                "verifier_decision_mode": (
                    "region_classification_then_categorical_pairwise"
                    if args.verifier == "qwen_zero_shot"
                    else "legacy_rule_score"
                ),
                "verifier_max_regions": args.verifier_max_regions,
                "verifier_min_region_area": args.verifier_min_region_area,
                "verifier_region_padding_ratio": args.verifier_region_padding_ratio,
                "coordinate_protocol": "public normalized_0_1000; environment pixel_xy",
                "matching_mode": args.matching_mode,
                "overlap_threshold": args.overlap_threshold,
                "gt_available_during_rollout": False,
            },
        )
        observation = environment.reset(image1, image2, args.query)
        invalid_outputs: list[dict[str, Any]] = []
        episode_stop_reason: str | None = None
        loop_index = 0
        while not environment.done:
            loop_index += 1
            observation, attempt_errors, action_executed = _execute_action_with_retries(
                qwen,
                environment,
                observation,
                args.action_retries,
                loop_index=loop_index,
            )
            invalid_outputs.extend(attempt_errors)
            if not action_executed:
                episode_stop_reason = "action_retry_exhaustion_without_state_change"
                break
        if episode_stop_reason is None:
            last_entry = environment.trajectory.entries[-1]
            episode_stop_reason = (
                "verifier_authorized_finish"
                if last_entry.parsed_action is not None
                and last_entry.parsed_action.action == "finish"
                and last_entry.execution.get("candidate_accepted")
                and last_entry.verifier.stop
                else "max_steps_reached"
            )
        tool_steps = [entry for entry in environment.trajectory.entries if entry.execution.get("tool")]

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
        accepted_count = sum(
            bool(entry.execution.get("candidate_accepted"))
            for entry in environment.trajectory.entries[1:]
        )
        rejected_count = sum(
            entry.execution.get("candidate_accepted") is False
            for entry in environment.trajectory.entries[1:]
        )
        episode_summary = {
            "sample": sample,
            "status": "success",
            "stop_reason": episode_stop_reason,
            "elapsed_seconds": round(time.monotonic() - sample_start, 3),
            "loop_count": loop_index,
            "candidate_count": len(environment.trajectory.entries) - 1,
            "accepted_candidate_count": accepted_count,
            "rejected_candidate_count": rejected_count,
            "tool_action_count": len(tool_steps),
            "action_attempt_count": (
                len(environment.trajectory.entries) - 1 + len(invalid_outputs)
            ),
            "invalid_action_attempt_count": len(invalid_outputs),
            "action_attempt_errors": invalid_outputs,
            "selected_step": environment.trajectory.best_entry.step_index,
            "verifier_best_step": environment.trajectory.verifier_best_entry.step_index,
            "trajectory": str(trajectory_path),
        }
        episode_summary_path = trajectory_dir / "episode_summary.json"
        episode_summary_path.write_text(
            json.dumps(episode_summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        selected = environment.best_state.change_mask
        verifier_best = environment.trajectory.verifier_best_entry.state.change_mask
        initial = environment.trajectory.entries[0].state.change_mask
        last = environment.trajectory.entries[-1].state.change_mask
        Image.fromarray(selected.astype(np.uint8) * 255, mode="L").save(
            args.output / "predictions" / sample_file
        )
        for name, mask in (
            ("selected", selected),
            ("verifier_best", verifier_best),
            ("initial", initial),
            ("last", last),
        ):
            Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(
                args.output / "predictions" / name / sample_file
            )
        rollout_records[sample] = {
            "trajectory": trajectory_path,
            "best_step": environment.trajectory.best_entry.step_index,
            "verifier_best_step": environment.trajectory.verifier_best_entry.step_index,
            "tool_action_count": len(tool_steps),
            "fallback_action_count": 0,
            "fallback_actions": [],
            "episode_stop_reason": episode_stop_reason,
            "episode_summary": episode_summary_path,
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


def _build_verifier(args: argparse.Namespace, qwen: GroundingModelQwen3VL):
    if args.verifier == "rule":
        return RuleBasedVerifier(accept_threshold=args.verifier_accept_threshold)
    return Qwen3VLZeroShotVerifier(
        model=qwen.model,
        processor=qwen.processor,
        max_new_tokens=args.verifier_max_new_tokens,
        accept_threshold=args.verifier_accept_threshold,
        max_retries=args.verifier_retries,
    )


def _execute_action_with_retries(
    qwen: GroundingModelQwen3VL,
    environment: ChangeAgentEnvironment,
    observation: AgentObservation,
    retries: int,
    *,
    loop_index: int = 0,
) -> tuple[AgentObservation, list[dict[str, Any]], bool]:
    """Try only model-produced actions and leave state unchanged on exhaustion."""

    validation_error = None
    previous_raw = None
    errors: list[dict[str, Any]] = []
    for attempt_index in range(1, retries + 1):
        raw = qwen.generate_raw(observation, validation_error, previous_raw)
        try:
            next_observation, _ = environment.step(raw)
            environment.trajectory.entries[-1].execution["action_generation"] = {
                "loop_index": loop_index,
                "attempt_index": attempt_index,
                "prompt_hash": getattr(qwen, "last_prompt_hash", None),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            return next_observation, errors, True
        except ActionValidationError as error:
            errors.append(
                {
                    "loop_index": loop_index,
                    "attempt_index": attempt_index,
                    "raw_agent_output": raw,
                    "error": str(error),
                    "prompt_hash": getattr(qwen, "last_prompt_hash", None),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            validation_error = str(error)
            previous_raw = raw
    return observation, errors, False


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
        selected_metrics = None
        verifier_best_metrics = None
        for step in payload["steps"]:
            mask_path = (trajectory_path.parent / step["change_mask_file"]).resolve()
            prediction = np.asarray(np.load(mask_path), dtype=bool)
            values, counts = _metrics(prediction, gt)
            step["offline_metrics"] = values
            step_metrics.append({"step_index": step["step_index"], **values})
            if step["step_index"] == payload["best_step"]:
                selected_metrics = values
                aggregate_counts += counts
            if step["step_index"] == payload["verifier_best_step"]:
                verifier_best_metrics = values
        trajectory_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        samples[sample] = {
            "verifier_selected_step": payload["best_step"],
            "tool_action_count": record["tool_action_count"],
            "steps": step_metrics,
            "initial": step_metrics[0],
            "verifier_selected": {"step_index": payload["best_step"], **selected_metrics},
            "verifier_best_step": payload["verifier_best_step"],
            "verifier_best": {
                "step_index": payload["verifier_best_step"], **verifier_best_metrics
            },
            "last": step_metrics[-1],
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
            args.input_root / "label_cvt" / sample,
        ])
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing required inputs: {missing}")


def _base_manifest(
    args: argparse.Namespace, seed_runtime: dict[str, Any]
) -> dict[str, Any]:
    return {
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "samples": list(SAMPLES),
        "query": args.query,
        "input_root": str(args.input_root),
        "initialization": "fresh dual-view SAM3 text prompting for every sample; no cached masks",
        "initialization_artifacts": str(args.output / "tool_runs" / "<sample>" / "sam3_initialization"),
        "agent_model": args.model_path,
        "point_executor": f"SimpleClick {args.simpleclick_checkpoint}",
        "box_executor": f"SAM3 {args.sam3_checkpoint}",
        "verifier": args.verifier,
        "verifier_model": "shared Qwen3-VL weights" if args.verifier == "qwen_zero_shot" else None,
        "verifier_decision_mode": (
            "region_classification_then_categorical_pairwise"
            if args.verifier == "qwen_zero_shot"
            else "legacy_rule_score"
        ),
        "verifier_max_regions": args.verifier_max_regions,
        "verifier_min_region_area": args.verifier_min_region_area,
        "verifier_region_padding_ratio": args.verifier_region_padding_ratio,
        "coordinate_protocol": "Agent/Verifier normalized_0_1000; Environment pixel_xy",
        "target_view_policy": "zero-shot visual inference; no alternating pseudo-label",
        "matching_mode": args.matching_mode,
        "overlap_threshold": args.overlap_threshold,
        "t12_min_instance_area": args.t12_min_instance_area,
        "cd_min_instance_area": args.cd_min_instance_area,
        "max_steps": args.max_steps,
        "selection_policy": args.selection_policy,
        "selection_epsilon": args.selection_epsilon,
        "max_selection_area_delta": args.max_selection_area_delta,
        "max_locality_outside_ratio": args.max_locality_outside_ratio,
        "max_target_mask_change_ratio": args.max_target_mask_change_ratio,
        "max_component_count_delta": args.max_component_count_delta,
        "seed": args.seed,
        "seed_runtime": seed_runtime,
        "python": sys.version,
        "platform": platform.platform(),
        "model_identity": _model_identity(Path(args.model_path)),
        "gt_policy": "label_cvt is opened only by evaluate_after_rollout after rollout_complete.json",
    }


def _seed_runtime(seed: int) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    result: dict[str, Any] = {
        "python_random": seed,
        "numpy": seed,
        "torch": None,
        "deterministic_algorithms": False,
    }
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        result["torch"] = seed
        result["torch_version"] = torch.__version__
        result["cuda_version"] = torch.version.cuda
    except ImportError:
        result["torch_version"] = None
        result["cuda_version"] = None
    return result


def _model_identity(model_path: Path) -> dict[str, Any]:
    resolved = model_path.resolve()
    identity: dict[str, Any] = {"path": str(resolved), "exists": resolved.exists()}
    candidates = (
        [resolved]
        if resolved.is_file()
        else [
            resolved / "config.json",
            resolved / "model.safetensors.index.json",
            resolved / "generation_config.json",
        ]
    )
    checksums: dict[str, str] = {}
    for candidate in candidates:
        if candidate.is_file():
            checksums[candidate.name] = hashlib.sha256(candidate.read_bytes()).hexdigest()
    identity["metadata_sha256"] = checksums
    return identity


def _render_manifest(manifest: dict[str, Any], metrics: dict[str, Any] | None = None) -> str:
    lines = ["# LEVIR-CD three-sample full Change-Agent run", ""]
    lines.extend(f"- {key}: `{value}`" for key, value in manifest.items())
    if metrics:
        lines.extend(["", "## Aggregate verifier-selected metrics", ""])
        lines.extend(f"- {key}: `{value}`" for key, value in metrics["aggregate"].items())
    lines.extend([
        "",
        "## Runtime boundary",
        "",
        "The final entry point is `tools/run_levir_change_agent.py`, not OmniOVCD/eval.py. ",
        "Each sample executes Environment.reset, Qwen action generation, ActionParser, a real ",
        "fresh SAM3 initialization, a SimpleClick or SAM3 action worker, Environment.step/rebuild, ",
        "and the configured GT-free Verifier. SAM3 masks, confidence maps, scores, prompts, and ",
        "worker parameters are persisted below tool_runs/<sample>/sam3_initialization. ",
        "Ground truth is loaded only after all trajectories and rollout_complete.json exist.",
    ])
    return "\n".join(lines) + "\n"


def _rss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / 1024.0 if value > 10_000 else value / (1024.0 * 1024.0)


if __name__ == "__main__":
    main()
