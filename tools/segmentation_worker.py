#!/usr/bin/env python3
"""Execute one real SimpleClick point or SAM3 box action for a parent rollout."""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
from PIL import Image


def rss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / 1024.0 if value > 10_000 else value / (1024.0 * 1024.0)


def point(args: argparse.Namespace, image: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    import torch
    from evaltools.model_loader import SegmentationModel
    from third_party.SimpleClick.isegm.inference import utils
    from third_party.SimpleClick.isegm.inference.predictors import get_predictor

    from change_agent.adapters.segagent_adapter import SimpleClickAdapter

    device = torch.device(args.device)
    model = utils.load_is_model(args.checkpoint, device, False)
    predictor = get_predictor(
        model,
        "NoBRS",
        device,
        prob_thresh=0.49,
        with_flip=False,
        zoom_in_params={"target_size": (448, 448), "skip_clicks": -1},
        predictor_params={"optimize_after_n_clicks": 1},
    )
    segmentation_model = SegmentationModel(predictor)
    initial_mask = np.asarray(np.load(args.initial_mask), dtype=bool)
    result = SimpleClickAdapter(segmentation_model).refine(
        image,
        initial_mask,
        tuple(args.coordinate),
        bool(args.is_positive),
    )
    return result, {
        "tool": "simpleclick",
        "checkpoint": args.checkpoint,
        "coordinate": args.coordinate,
        "is_positive": bool(args.is_positive),
    }


def box(args: argparse.Namespace, image: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    import torch
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    from change_agent.adapters.sam3_adapter import SAM3ProcessorAdapter

    model = build_sam3_image_model(
        bpe_path=args.bpe,
        checkpoint_path=args.checkpoint,
        device=args.device,
        load_from_HF=False,
    )
    model.to(args.device)
    model._device = torch.device(args.device)
    processor = Sam3Processor(model, resolution=args.resolution, device=args.device)
    result = SAM3ProcessorAdapter(processor).segment_box(image, tuple(args.box), args.query)
    return result, {
        "tool": "sam3",
        "checkpoint": args.checkpoint,
        "bpe": args.bpe,
        "resolution": args.resolution,
        "query": args.query,
        "box_cxcywh_normalized": args.box,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("point", "box"))
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-mask", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--initial-mask", type=Path)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--coordinate", nargs=2, type=int)
    parser.add_argument("--is-positive", type=int, choices=(0, 1))
    parser.add_argument("--bpe")
    parser.add_argument("--resolution", type=int, default=1008)
    parser.add_argument("--query")
    parser.add_argument("--box", nargs=4, type=float)
    args = parser.parse_args()
    if args.mode == "point" and (
        args.initial_mask is None or args.coordinate is None or args.is_positive is None
    ):
        parser.error("point requires --initial-mask, --coordinate, and --is-positive")
    if args.mode == "box" and (args.bpe is None or args.query is None or args.box is None):
        parser.error("box requires --bpe, --query, and --box")

    start = time.monotonic()
    before = rss_mb()
    image = np.asarray(Image.open(args.image).convert("RGB"))
    result, details = point(args, image) if args.mode == "point" else box(args, image)
    result = np.asarray(result, dtype=bool)
    if result.shape != image.shape[:2]:
        raise ValueError(f"worker mask shape {result.shape} != image shape {image.shape[:2]}")
    args.output_mask.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_mask, result.astype(np.uint8))
    report = {
        "status": "success",
        "mode": args.mode,
        "device": args.device,
        "elapsed_seconds": round(time.monotonic() - start, 3),
        "rss_before_mb": round(before, 2),
        "rss_peak_mb": round(rss_mb(), 2),
        "output_shape": list(result.shape),
        "output_pixels": int(result.sum()),
        **details,
    }
    try:
        import torch

        report["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            report["cuda"] = {
                "allocated_mb": round(torch.cuda.memory_allocated() / 2**20, 2),
                "reserved_mb": round(torch.cuda.memory_reserved() / 2**20, 2),
                "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 2**20, 2),
            }
    except ImportError:
        report["cuda_available"] = False
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
