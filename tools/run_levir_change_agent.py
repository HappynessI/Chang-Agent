#!/usr/bin/env python3
"""Run the real Qwen → Environment → segmentation-tool loop on three LEVIR samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
from change_agent.adapters.staged_verifier import StagedQwenVerifier
from change_agent.adapters.stage_backends import (
    BailianQwen3VLStageBackend,
    LocalQwen3VLStageBackend,
)
from change_agent.adapters.bailian_adapter import BailianGroundingModelQwen3VL
from change_agent.adapters.direct_verifier import DirectQwenVerifier
from change_agent.adapters.subprocess_adapters import (
    SubprocessBoxBackend,
    SubprocessPointBackend,
    SubprocessSAM3Initializer,
)
from change_agent.environment import ChangeAgentEnvironment
from change_agent.executor import ActionExecutor
from change_agent.state import AgentObservation
from change_agent.trajectory import default_run_metadata
from change_agent.verifier import RuleBasedVerifier


def _staged_protocol_version() -> str:
    return os.environ.get("CHANGE_AGENT_STAGED_PROTOCOL_VERSION", "v11")
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
    parser.add_argument(
        "--samples",
        nargs="+",
        default=list(SAMPLES),
        help="Optional subset of the fixed LEVIR smoke samples (filenames or stems).",
    )
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
    parser.add_argument(
        "--verifier-max-regions",
        type=int,
        default=1,
        help="Maximum initial components per Qwen call; all components are audited.",
    )
    parser.add_argument(
        "--verifier-max-delta-regions-per-batch",
        "--verifier-max-delta-regions",
        dest="verifier_max_delta_regions",
        type=int,
        default=1,
        help="Maximum delta components per Qwen call; all components are still audited.",
    )
    parser.add_argument("--verifier-min-region-area", type=int, default=4)
    parser.add_argument("--verifier-region-padding-ratio", type=float, default=0.25)
    parser.add_argument("--staged-verifier-max-total-regions", type=int, default=128)
    parser.add_argument(
        "--verifier-max-selected-regions",
        type=int,
        default=3,
        help="Maximum existing region IDs selected for each global/local audit.",
    )
    parser.add_argument("--matching-mode", choices=sorted(MaskPairProcessor.MODES), default="overlap_presence")
    parser.add_argument("--overlap-threshold", type=float, default=0.25)
    parser.add_argument("--t12-min-instance-area", type=int, default=0)
    parser.add_argument("--cd-min-instance-area", type=int, default=0)
    parser.add_argument("--model-path", default=str(root / "models/Qwen3-VL-2B-Instruct"))
    parser.add_argument(
        "--agent-backend", choices=("local", "bailian"), default="local"
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--verifier",
        choices=("qwen_zero_shot", "qwen_staged", "rule"),
        default="qwen_zero_shot",
    )
    parser.add_argument(
        "--staged-verifier-backend",
        choices=("local", "bailian"),
        default="local",
    )
    parser.add_argument(
        "--proposal-mode",
        choices=("direct", "proposal", "hybrid"),
        default="hybrid",
        help=(
            "Direct uses full-state Qwen diagnosis/action geometry without Proposals; "
            "Proposal and Hybrid exhaustively audit Environment-numbered regions with "
            "local/exact-component crops; Environment owns execution geometry."
        ),
    )
    parser.add_argument("--bailian-model", default="qwen3-vl-plus")
    parser.add_argument(
        "--verifier-bailian-model",
        default=None,
        help=(
            "Optional hosted verifier model. When omitted, --bailian-model is shared; "
            "setting it isolates verifier-model ablations from the hosted Agent."
        ),
    )
    parser.add_argument("--bailian-base-url", default=None)
    parser.add_argument("--bailian-api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument(
        "--bailian-enable-thinking",
        action="store_true",
        help="Enable hosted hybrid-model reasoning before each structured stage response.",
    )
    parser.add_argument(
        "--bailian-thinking-budget",
        type=int,
        default=None,
        help="Optional positive reasoning-token budget; used only with --bailian-enable-thinking.",
    )
    parser.add_argument("--verifier-max-new-tokens", type=int, default=1024)
    parser.add_argument("--verifier-accept-threshold", type=float, default=0.82)
    parser.add_argument(
        "--verifier-min-visual-confidence",
        type=float,
        default=0.6,
        help=(
            "Deprecated compatibility option. Staged v11 uses grounded discrete "
            "evidence_quality and an atomic runtime audit checklist instead of "
            "model-authored numeric confidence."
        ),
    )
    parser.add_argument("--verifier-retries", type=int, default=2)
    parser.add_argument("--verifier-repetition-penalty", type=float, default=1.05)
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
    parser.add_argument(
        "--visualize",
        action="store_true",
        help=(
            "Save per-step T1/T2 binary masks and each connected-component "
            "instance mask as PNG files."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    staged_protocol_version = _staged_protocol_version()
    start = time.monotonic()
    seed_runtime = _seed_runtime(args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ("predictions", "trajectories", "masks", "verifier_feedback", "logs", "tool_runs"):
        (args.output / name).mkdir()
    for name in ("initial", "verifier_best", "last", "selected"):
        (args.output / "predictions" / name).mkdir()
    if args.visualize:
        (args.output / "visualizations").mkdir()
    _validate_inputs(args)
    manifest = _base_manifest(args, seed_runtime)
    (args.output / "run_manifest.md").write_text(_render_manifest(manifest), encoding="utf-8")

    local_qwen = None
    needs_local_qwen = (
        args.agent_backend == "local"
        or args.verifier == "qwen_zero_shot"
        or (
            args.verifier == "qwen_staged"
            and args.staged_verifier_backend == "local"
        )
    )
    if needs_local_qwen:
        local_qwen = GroundingModelQwen3VL(
            args.model_path,
            device_map=args.device_map,
            max_new_tokens=args.max_new_tokens,
        )
    agent_hosted_client = None
    if args.agent_backend == "bailian":
        agent_hosted_client = BailianQwen3VLStageBackend(
            model=args.bailian_model,
            base_url=args.bailian_base_url,
            api_key_env=args.bailian_api_key_env,
            max_completion_tokens=args.max_new_tokens,
            seed=args.seed,
            enable_thinking=False,
        )
    verifier_hosted_client = None
    if (
        args.verifier == "qwen_staged"
        and args.staged_verifier_backend == "bailian"
    ):
        verifier_hosted_client = BailianQwen3VLStageBackend(
            model=args.verifier_bailian_model or args.bailian_model,
            base_url=args.bailian_base_url,
            api_key_env=args.bailian_api_key_env,
            max_completion_tokens=args.verifier_max_new_tokens,
            seed=args.seed,
            enable_thinking=args.bailian_enable_thinking,
            thinking_budget=args.bailian_thinking_budget,
        )
    qwen = (
        local_qwen
        if args.agent_backend == "local"
        else BailianGroundingModelQwen3VL(client=agent_hosted_client)
    )
    if qwen is None:
        raise RuntimeError("agent backend was not initialized")
    verifier = _build_verifier(args, local_qwen, verifier_hosted_client)
    rollout_records: dict[str, dict[str, Any]] = {}
    sample_files = tuple(
        name if str(name).endswith(".png") else f"{name}.png"
        for name in args.samples
    )
    for sample_file in sample_files:
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
            verifier_max_delta_regions=args.verifier_max_delta_regions,
            verifier_min_region_area=args.verifier_min_region_area,
            verifier_region_padding_ratio=args.verifier_region_padding_ratio,
            enable_verifier_regions=args.proposal_mode != "direct",
            run_metadata={
                "sample": sample_file,
                "query": args.query,
                "agent": (
                    args.model_path
                    if args.agent_backend == "local"
                    else args.bailian_model
                ),
                "action_generation_policy": (
                    "direct_verifier_full_context"
                    if args.proposal_mode == "direct"
                    else "verifier_region_id_programmatic_geometry"
                ),
                "verifier": args.verifier,
                "verifier_model": (
                    args.verifier_bailian_model or args.bailian_model
                    if args.staged_verifier_backend == "bailian"
                    else args.model_path
                ),
                "proposal_mode": args.proposal_mode,
                "proposal_grounding_enabled": args.proposal_mode != "direct",
                "verifier_decision_mode": (
                    "qwen_full_context_direct_binary_rubric"
                    if args.proposal_mode == "direct"
                    else f"qwen_staged_deterministic_target_resolution_{staged_protocol_version}"
                    if args.proposal_mode == "proposal"
                    else f"qwen_staged_deterministic_target_resolution_{staged_protocol_version}_full_context"
                    if args.verifier == "qwen_staged"
                    else "qwen_rich_region_diagnosis_and_global_synthesis"
                    if args.verifier == "qwen_zero_shot"
                    else "legacy_rule_score"
                ),
                "verifier_max_regions": args.verifier_max_regions,
                "verifier_max_selected_regions": args.verifier_max_selected_regions,
                "verifier_max_delta_regions_per_batch": args.verifier_max_delta_regions,
                "verifier_do_sample": False,
                "bailian_enable_thinking": args.bailian_enable_thinking,
                "bailian_thinking_budget": args.bailian_thinking_budget,
                "verifier_repetition_penalty": args.verifier_repetition_penalty,
                "verifier_min_visual_confidence": None,
                "verifier_confidence_policy": (
                    "atomic grounded audit checklist with per-item evidence; no model-authored numeric confidence"
                ),
                "verifier_candidate_evidence_modes": list(
                    Qwen3VLZeroShotVerifier.CANDIDATE_EVIDENCE_MODES
                )
                if args.verifier == "qwen_zero_shot"
                else None,
                "verifier_min_region_area": args.verifier_min_region_area,
                "verifier_region_padding_ratio": args.verifier_region_padding_ratio,
                "coordinate_protocol": "public normalized_0_1000; environment pixel_xy",
                "matching_mode": args.matching_mode,
                "overlap_threshold": args.overlap_threshold,
                "t12_min_instance_area": args.t12_min_instance_area,
                "cd_min_instance_area": args.cd_min_instance_area,
                "gt_available_during_rollout": False,
            },
        )
        observation = environment.reset(image1, image2, args.query)
        invalid_outputs: list[dict[str, Any]] = []
        episode_stop_reason = _initial_verifier_stop_reason(observation)
        loop_index = 0
        while not environment.done and episode_stop_reason is None:
            loop_index += 1
            observation, attempt_errors, action_executed = _execute_verifier_action(
                environment,
                observation,
                loop_index=loop_index,
                source=(
                    "direct_verifier_full_context"
                    if args.proposal_mode == "direct"
                    else "verifier_region_id_programmatic_geometry"
                ),
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
            trajectory_dir,
            args.output / "masks" / sample,
            (
                args.output / "visualizations" / sample
                if args.visualize
                else None
            ),
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


def _build_verifier(
    args: argparse.Namespace,
    local_qwen: GroundingModelQwen3VL | None,
    hosted_client: BailianQwen3VLStageBackend | None,
):
    if args.verifier == "rule":
        return RuleBasedVerifier(accept_threshold=args.verifier_accept_threshold)
    if args.verifier == "qwen_staged":
        if args.staged_verifier_backend == "local":
            if local_qwen is None:
                raise RuntimeError("local staged verifier requires local Qwen weights")
            stage_backend = LocalQwen3VLStageBackend(
                model=local_qwen.model,
                processor=local_qwen.processor,
                max_new_tokens=args.verifier_max_new_tokens,
                do_sample=False,
                repetition_penalty=args.verifier_repetition_penalty,
            )
        else:
            if hosted_client is None:
                raise RuntimeError("BaiLian staged verifier client was not initialized")
            stage_backend = hosted_client
        return StagedQwenVerifier(
            stage_backend,
            accept_threshold=args.verifier_accept_threshold,
            max_regions=args.staged_verifier_max_total_regions,
            max_selected_regions=args.verifier_max_selected_regions,
            max_retries=args.verifier_retries,
            visual_context=args.proposal_mode,
        ) if args.proposal_mode != "direct" else DirectQwenVerifier(
            stage_backend,
            accept_threshold=args.verifier_accept_threshold,
            max_retries=args.verifier_retries,
        )
    if local_qwen is None:
        raise RuntimeError("legacy qwen_zero_shot verifier requires local Qwen weights")
    return Qwen3VLZeroShotVerifier(
        model=local_qwen.model,
        processor=local_qwen.processor,
        max_new_tokens=args.verifier_max_new_tokens,
        accept_threshold=args.verifier_accept_threshold,
        max_retries=args.verifier_retries,
        do_sample=False,
        repetition_penalty=args.verifier_repetition_penalty,
    )


def _initial_verifier_stop_reason(observation: Any) -> str | None:
    feedback = getattr(observation, "feedback", None)
    if feedback is not None and not feedback.verifier_valid:
        return "initial_verifier_invalid"
    if (
        feedback is not None
        and bool(getattr(feedback, "accept", False))
        and bool(getattr(feedback, "stop", False))
        and getattr(feedback, "comparison", None) == "initial"
        and getattr(feedback, "error_type", None) == "none"
    ):
        return "initial_verifier_authorized_finish"
    return None


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


def _execute_verifier_action(
    environment: ChangeAgentEnvironment,
    observation: AgentObservation,
    *,
    loop_index: int,
    source: str = "verifier_region_id_programmatic_geometry",
) -> tuple[AgentObservation, list[dict[str, Any]], bool]:
    """Execute geometry already authorized by the verifier/runtime contract."""

    feedback = observation.feedback
    if (
        feedback is None
        or not feedback.verifier_valid
        or feedback.suggested_action is None
    ):
        return observation, [
            {
                "loop_index": loop_index,
                "error": "verifier did not authorize an executable action",
            }
        ], False
    payload: dict[str, Any] = {
        "target_view": feedback.target_view,
        "action": feedback.suggested_action,
    }
    if feedback.suggested_action in {"positive_point", "negative_point"}:
        if feedback.error_region is None:
            return observation, [
                {
                    "loop_index": loop_index,
                    "error": "verifier point action lacks normalized geometry",
                }
            ], False
        payload["coordinate"] = list(feedback.error_region[:2])
    elif feedback.suggested_action == "box":
        if feedback.error_region is None:
            return observation, [
                {
                    "loop_index": loop_index,
                    "error": "verifier box action lacks normalized geometry",
                }
            ], False
        payload["box"] = list(feedback.error_region)
    raw = json.dumps(payload, separators=(",", ":"))
    try:
        next_observation, _ = environment.step(raw)
    except ActionValidationError as error:
        return observation, [
            {
                "loop_index": loop_index,
                "raw_verifier_action": raw,
                "error": str(error),
            }
        ], False
    environment.trajectory.entries[-1].execution["action_generation"] = {
        "loop_index": loop_index,
        "source": source,
    }
    return next_observation, [], True


def _execute_direct_verifier_action(
    environment: ChangeAgentEnvironment,
    observation: AgentObservation,
    *,
    loop_index: int,
) -> tuple[AgentObservation, list[dict[str, Any]], bool]:
    """Backward-compatible wrapper for the Direct ablation."""

    return _execute_verifier_action(
        environment,
        observation,
        loop_index=loop_index,
        source="direct_verifier_full_context",
    )


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
    if args.proposal_mode != "hybrid" and args.verifier != "qwen_staged":
        raise ValueError("proposal ablations require --verifier qwen_staged")
    needs_local_qwen = (
        args.agent_backend == "local"
        or args.verifier == "qwen_zero_shot"
        or (
            args.verifier == "qwen_staged"
            and args.staged_verifier_backend == "local"
        )
    )
    files = [Path(args.simpleclick_checkpoint), Path(args.sam3_checkpoint), Path(args.sam3_bpe)]
    if needs_local_qwen:
        files.append(Path(args.model_path))
    sample_files = [
        str(sample) if str(sample).endswith(".png") else f"{sample}.png"
        for sample in args.samples
    ]
    for sample in sample_files:
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
    source = default_run_metadata()
    return {
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "samples": [
            str(sample) if str(sample).endswith(".png") else f"{sample}.png"
            for sample in args.samples
        ],
        "query": args.query,
        "input_root": str(args.input_root),
        "initialization": "fresh dual-view SAM3 text prompting for every sample; no cached masks",
        "initialization_artifacts": str(args.output / "tool_runs" / "<sample>" / "sam3_initialization"),
        "agent_model": (
            args.model_path if args.agent_backend == "local" else args.bailian_model
        ),
        "agent_backend": args.agent_backend,
        "action_generation_policy": (
            "direct_verifier_full_context"
            if args.proposal_mode == "direct"
            else "verifier_region_id_programmatic_geometry"
        ),
        "point_executor": f"SimpleClick {args.simpleclick_checkpoint}",
        "box_executor": f"SAM3 {args.sam3_checkpoint}",
        "verifier": args.verifier,
        "proposal_mode": args.proposal_mode,
        "proposal_semantics": {
            "direct": "full-state Qwen diagnosis and model-authored action geometry; no Proposal attachment",
            "proposal": "atomic grounded region audit over marked RGB/change overview plus exact component/local crops, runtime candidate transition rubric, and programmatic geometry",
            "hybrid": "atomic grounded region audit over marked full mask state plus exact component/local crops, runtime candidate transition rubric, and programmatic geometry",
        }[args.proposal_mode],
        "verifier_model": (
            "shared local Qwen3-VL weights"
            if args.verifier == "qwen_zero_shot"
            else args.model_path
            if args.verifier == "qwen_staged"
            and args.staged_verifier_backend == "local"
            else args.verifier_bailian_model or args.bailian_model
            if args.verifier == "qwen_staged"
            else None
        ),
        "staged_verifier_backend": (
            args.staged_verifier_backend if args.verifier == "qwen_staged" else None
        ),
        "verifier_decision_mode": (
            "qwen_rich_region_diagnosis_and_global_synthesis"
            if args.verifier == "qwen_zero_shot"
            else "qwen_full_context_direct_binary_rubric"
            if args.verifier == "qwen_staged" and args.proposal_mode == "direct"
            else f"qwen_staged_deterministic_target_resolution_{_staged_protocol_version()}"
            if args.verifier == "qwen_staged" and args.proposal_mode == "proposal"
            else f"qwen_staged_deterministic_target_resolution_{_staged_protocol_version()}_full_context"
            if args.verifier == "qwen_staged" and args.proposal_mode == "hybrid"
            else "legacy_rule_score"
        ),
        "verifier_max_initial_regions_per_batch": args.verifier_max_regions,
        "verifier_max_selected_regions": args.verifier_max_selected_regions,
        "verifier_max_delta_regions_per_batch": args.verifier_max_delta_regions,
        "verifier_do_sample": False,
        "bailian_enable_thinking": args.bailian_enable_thinking,
        "bailian_thinking_budget": args.bailian_thinking_budget,
        "verifier_repetition_penalty": args.verifier_repetition_penalty,
        "verifier_min_visual_confidence": None,
        "verifier_confidence_policy": (
            "atomic grounded audit checklist with per-item evidence; no model-authored numeric confidence"
        ),
        "verifier_candidate_evidence_modes": list(
            Qwen3VLZeroShotVerifier.CANDIDATE_EVIDENCE_MODES
        )
        if args.verifier == "qwen_zero_shot"
        else None,
        "verifier_min_region_area": args.verifier_min_region_area,
        "verifier_region_padding_ratio": args.verifier_region_padding_ratio,
        "coordinate_protocol": "Agent/Verifier normalized_0_1000; Environment pixel_xy",
        "target_view_policy": "zero-shot visual inference; no alternating pseudo-label",
        "matching_mode": args.matching_mode,
        "overlap_threshold": args.overlap_threshold,
        "t12_min_instance_area": args.t12_min_instance_area,
        "cd_min_instance_area": args.cd_min_instance_area,
        "visualize": args.visualize,
        "visualization_artifacts": (
            str(args.output / "visualizations" / "<sample>" / "step_<index>")
            if args.visualize
            else None
        ),
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
        "model_identity": (
            _model_identity(Path(args.model_path))
            if args.agent_backend == "local"
            or args.verifier == "qwen_zero_shot"
            or (
                args.verifier == "qwen_staged"
                and args.staged_verifier_backend == "local"
            )
            else {"provider": "bailian", "model": args.bailian_model}
        ),
        "verifier_model_identity": (
            {"provider": "bailian", "model": args.verifier_bailian_model or args.bailian_model}
            if args.verifier == "qwen_staged"
            and args.staged_verifier_backend == "bailian"
            else None
        ),
        "git_commit": source["git_commit"],
        "git_dirty": source["git_dirty"],
        "git_worktree_sha256": source["git_worktree_sha256"],
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
