"""Stable mask-level boundary around OmniOVCD/SAM3 functionality."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np


@dataclass
class PairUpdate:
    t1_instances: tuple[np.ndarray, ...]
    t2_instances: tuple[np.ndarray, ...]
    matching: tuple[tuple[int, int], ...]
    change_mask: np.ndarray
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class InitializationResult:
    t1_mask: np.ndarray
    t2_mask: np.ndarray
    update: PairUpdate


class MaskPairProcessor:
    """Pure NumPy fallback matching the current OmniOVCD overlap semantics."""

    def __init__(self, overlap_threshold: float = 0.5, min_instance_area: int = 1):
        if not 0 <= overlap_threshold <= 1:
            raise ValueError("overlap_threshold must be in [0, 1]")
        if min_instance_area < 1:
            raise ValueError("min_instance_area must be positive")
        self.overlap_threshold = overlap_threshold
        self.min_instance_area = min_instance_area

    def rebuild(
        self,
        t1_mask: np.ndarray,
        t2_mask: np.ndarray,
        evidence: Mapping[str, Any] | None = None,
    ) -> PairUpdate:
        t1_mask = np.asarray(t1_mask, dtype=bool)
        t2_mask = np.asarray(t2_mask, dtype=bool)
        if t1_mask.shape != t2_mask.shape or t1_mask.ndim != 2:
            raise ValueError("T1/T2 masks must be same-shaped 2-D arrays")
        t1_instances = connected_components(t1_mask, self.min_instance_area)
        t2_instances = connected_components(t2_mask, self.min_instance_area)
        matching = self._match(t1_instances, t2_instances)
        matched_t1 = {left for left, _ in matching}
        matched_t2 = {right for _, right in matching}
        change = np.zeros_like(t1_mask, dtype=bool)
        for index, instance in enumerate(t1_instances):
            if index not in matched_t1:
                change |= instance
        for index, instance in enumerate(t2_instances):
            if index not in matched_t2:
                change |= instance
        return PairUpdate(
            t1_instances=t1_instances,
            t2_instances=t2_instances,
            matching=matching,
            change_mask=change,
            evidence=dict(evidence or {}),
        )

    def _match(
        self,
        t1_instances: tuple[np.ndarray, ...],
        t2_instances: tuple[np.ndarray, ...],
    ) -> tuple[tuple[int, int], ...]:
        candidates: list[tuple[float, int, int]] = []
        for left, mask1 in enumerate(t1_instances):
            area1 = int(mask1.sum())
            for right, mask2 in enumerate(t2_instances):
                intersection = int(np.logical_and(mask1, mask2).sum())
                if not intersection:
                    continue
                area2 = int(mask2.sum())
                overlap = min(intersection / area1, intersection / area2)
                if overlap >= self.overlap_threshold:
                    candidates.append((overlap, left, right))
        # Deterministic one-to-one greedy matching avoids hiding duplicate instances.
        matches: list[tuple[int, int]] = []
        used_left: set[int] = set()
        used_right: set[int] = set()
        for _, left, right in sorted(candidates, reverse=True):
            if left not in used_left and right not in used_right:
                matches.append((left, right))
                used_left.add(left)
                used_right.add(right)
        return tuple(sorted(matches))


def connected_components(mask: np.ndarray, min_area: int = 1) -> tuple[np.ndarray, ...]:
    """Extract 8-connected instances without requiring OpenCV/skimage."""

    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("mask must be 2-D")
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for start_y, start_x in zip(*np.nonzero(mask)):
        if visited[start_y, start_x]:
            continue
        stack = [(int(start_y), int(start_x))]
        visited[start_y, start_x] = True
        pixels: list[tuple[int, int]] = []
        while stack:
            y, x = stack.pop()
            pixels.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and mask[ny, nx]
                        and not visited[ny, nx]
                    ):
                        visited[ny, nx] = True
                        stack.append((ny, nx))
        if len(pixels) >= min_area:
            component = np.zeros_like(mask, dtype=bool)
            ys, xs = zip(*pixels)
            component[np.asarray(ys), np.asarray(xs)] = True
            components.append(component)
    return tuple(components)


class OmniOVCDAdapter:
    """Injectable adapter for expensive OmniOVCD initialization and SAM3 boxes.

    The callbacks are constructed inside the OmniOVCD environment so this package
    never imports incompatible CUDA stacks into the Agent process.
    """

    def __init__(
        self,
        initialize_masks: Callable[[np.ndarray, np.ndarray, str], Any],
        segment_box_callback: Callable[
            [np.ndarray, tuple[float, float, float, float], str], Any
        ],
        pair_processor: MaskPairProcessor | None = None,
    ):
        self.initialize_masks = initialize_masks
        self.segment_box_callback = segment_box_callback
        self.pair_processor = pair_processor or MaskPairProcessor()

    def initialize(
        self, t1_image: np.ndarray, t2_image: np.ndarray, query: str
    ) -> InitializationResult:
        raw = self.initialize_masks(t1_image, t2_image, query)
        if isinstance(raw, Mapping):
            t1_mask, t2_mask = raw["t1_mask"], raw["t2_mask"]
            evidence = dict(raw.get("evidence", {}))
        elif isinstance(raw, tuple) and len(raw) in {2, 3}:
            t1_mask, t2_mask = raw[:2]
            evidence = dict(raw[2]) if len(raw) == 3 else {}
        else:
            raise TypeError("initializer must return a mapping or (t1_mask, t2_mask[, evidence])")
        t1_mask = np.asarray(t1_mask, dtype=bool)
        t2_mask = np.asarray(t2_mask, dtype=bool)
        update = self.pair_processor.rebuild(t1_mask, t2_mask, evidence)
        return InitializationResult(t1_mask=t1_mask, t2_mask=t2_mask, update=update)

    def rebuild(
        self, t1_mask: np.ndarray, t2_mask: np.ndarray, evidence: Mapping[str, Any]
    ) -> PairUpdate:
        return self.pair_processor.rebuild(t1_mask, t2_mask, evidence)

    def segment_box(
        self,
        image: np.ndarray,
        box_cxcywh_normalized: tuple[float, float, float, float],
        query: str,
    ) -> np.ndarray:
        return np.asarray(
            self.segment_box_callback(image, box_cxcywh_normalized, query), dtype=bool
        )

