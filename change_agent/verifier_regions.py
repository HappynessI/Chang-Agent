"""Environment-owned local region proposals for Change Verifier inspection."""

from __future__ import annotations

from typing import Any

import numpy as np

from .adapters.omniovcd_adapter import connected_components
from .coordinates import pixel_box_to_normalized
from .state import ChangeState


def attach_verifier_regions(
    state: ChangeState,
    previous_state: ChangeState | None = None,
    *,
    max_regions: int = 6,
    max_delta_regions: int = 3,
    min_component_area: int = 4,
    padding_ratio: float = 0.25,
) -> list[dict[str, Any]]:
    """Attach a small deterministic proposal set and exact mask facts to ``state``.

    Proposals come from current change components, T1/T2 semantic-mask difference,
    and (for edited candidates) the change-mask delta against the previous accepted
    state. The Verifier selects/classifies these proposals; it does not invent boxes.
    """

    proposals = (
        build_verifier_regions(
            state,
            None,
            max_regions=max_regions,
            min_component_area=min_component_area,
            padding_ratio=padding_ratio,
        )
        if previous_state is None
        else build_candidate_delta_regions(
            state,
            previous_state,
            max_regions=max_delta_regions,
            min_component_area=min_component_area,
            padding_ratio=padding_ratio,
        )
    )
    temporal_difference = np.logical_xor(state.t1_mask, state.t2_mask)
    candidate_delta = (
        np.zeros_like(state.change_mask)
        if previous_state is None
        else np.logical_xor(previous_state.change_mask, state.change_mask)
    )
    state.evidence = dict(state.evidence)
    state.evidence["verifier_region_proposals"] = proposals
    state.evidence["verifier_mask_facts"] = {
        "height": int(state.change_mask.shape[0]),
        "width": int(state.change_mask.shape[1]),
        "change_pixels": int(state.change_mask.sum()),
        "change_area_ratio": float(state.change_mask.mean()),
        "t1_mask_pixels": int(state.t1_mask.sum()),
        "t2_mask_pixels": int(state.t2_mask.sum()),
        "temporal_difference_pixels": int(temporal_difference.sum()),
        "candidate_delta_pixels": int(candidate_delta.sum()),
        "candidate_added_pixels": int(
            np.logical_and(state.change_mask, ~previous_state.change_mask).sum()
        )
        if previous_state is not None
        else 0,
        "candidate_removed_pixels": int(
            np.logical_and(previous_state.change_mask, ~state.change_mask).sum()
        )
        if previous_state is not None
        else 0,
        "candidate_delta_covered_pixels": int(
            sum(item["component_area"] for item in proposals)
        )
        if previous_state is not None
        else 0,
        "proposal_count": len(proposals),
        "proposal_config": {
            "schema_version": "component_delta_v2",
            "max_regions": max_regions if previous_state is None else max_delta_regions,
            "min_component_area": min_component_area,
            "padding_ratio": padding_ratio,
        },
    }
    if previous_state is not None:
        covered = int(state.evidence["verifier_mask_facts"]["candidate_delta_covered_pixels"])
        total = int(candidate_delta.sum())
        state.evidence["verifier_mask_facts"]["candidate_delta_uncovered_pixels"] = max(
            0, total - covered
        )
        state.evidence["verifier_mask_facts"]["candidate_delta_coverage_ratio"] = (
            covered / total if total else 1.0
        )
    return proposals


def build_candidate_delta_regions(
    state: ChangeState,
    previous_state: ChangeState,
    *,
    max_regions: int = 3,
    min_component_area: int = 1,
    padding_ratio: float = 0.25,
) -> list[dict[str, Any]]:
    """Build compact, polarity-preserving component panels for a candidate edit.

    Connected components remain separate, so spatially distant or semantically mixed
    edits cannot be collapsed into one label. Coverage facts force conservative
    rejection when the configured component budget cannot inspect every changed pixel.
    """

    if max_regions < 1:
        raise ValueError("max_regions must be positive")
    if min_component_area < 1:
        raise ValueError("min_component_area must be positive")
    if padding_ratio < 0:
        raise ValueError("padding_ratio must be non-negative")

    change = np.asarray(state.change_mask, dtype=bool)
    previous = np.asarray(previous_state.change_mask, dtype=bool)
    candidate_added = np.logical_and(change, ~previous)
    candidate_removed = np.logical_and(previous, ~change)
    temporal_difference = np.logical_xor(state.t1_mask, state.t2_mask)
    raw: list[tuple[str, np.ndarray]] = []
    for effect_kind, mask in (
        ("added", candidate_added),
        ("removed", candidate_removed),
    ):
        components = sorted(
            connected_components(mask), key=lambda item: int(item.sum()), reverse=True
        )
        raw.extend((effect_kind, component) for component in components)
    raw.sort(key=lambda item: int(item[1].sum()), reverse=True)

    height, width = change.shape
    result: list[dict[str, Any]] = []
    for index, (effect_kind, component) in enumerate(raw[:max_regions]):
        seed_y, seed_x = np.argwhere(component)[0]
        crop_box = _padded_box(_mask_box(component), (height, width), padding_ratio)
        x1, y1, x2, y2 = crop_box
        crop = np.zeros_like(change)
        crop[y1 : y2 + 1, x1 : x2 + 1] = True
        result.append(
            {
                "region_id": f"d{index}",
                "effect_kind": effect_kind,
                "sources": [f"candidate_{effect_kind}"],
                "component_area": int(component.sum()),
                "component_seed_pixels": [int(seed_x), int(seed_y)],
                "box_pixels": list(crop_box),
                "box_normalized": list(
                    pixel_box_to_normalized(crop_box, (width, height))
                ),
                "change_pixels": int(np.logical_and(change, crop).sum()),
                "temporal_difference_pixels": int(
                    np.logical_and(temporal_difference, crop).sum()
                ),
                "candidate_delta_pixels": int(component.sum()),
                "delta_pixels": int(component.sum()),
                "candidate_added_pixels": int(component.sum())
                if effect_kind == "added"
                else 0,
                "candidate_removed_pixels": int(component.sum())
                if effect_kind == "removed"
                else 0,
                "t1_mask_pixels": int(np.logical_and(state.t1_mask, crop).sum()),
                "t2_mask_pixels": int(np.logical_and(state.t2_mask, crop).sum()),
            }
        )
    return result


def build_verifier_regions(
    state: ChangeState,
    previous_state: ChangeState | None = None,
    *,
    max_regions: int = 6,
    min_component_area: int = 4,
    padding_ratio: float = 0.25,
) -> list[dict[str, Any]]:
    if max_regions < 1:
        raise ValueError("max_regions must be positive")
    if min_component_area < 1:
        raise ValueError("min_component_area must be positive")
    if padding_ratio < 0:
        raise ValueError("padding_ratio must be non-negative")

    change = np.asarray(state.change_mask, dtype=bool)
    temporal_difference = np.logical_xor(state.t1_mask, state.t2_mask)
    missing_from_change = np.logical_and(temporal_difference, ~change)
    candidate_added = np.zeros_like(change)
    candidate_removed = np.zeros_like(change)
    if previous_state is not None:
        candidate_added = np.logical_and(change, ~previous_state.change_mask)
        candidate_removed = np.logical_and(previous_state.change_mask, ~change)

    source_masks = [
        ("candidate_added", candidate_added, 0),
        ("candidate_removed", candidate_removed, 0),
        ("change_component", change, 1),
        ("temporal_difference_missing", missing_from_change, 2),
        ("temporal_difference", temporal_difference, 3),
    ]
    raw: list[dict[str, Any]] = []
    for source, source_mask, priority in source_masks:
        components = sorted(
            connected_components(source_mask),
            key=lambda item: int(item.sum()),
            reverse=True,
        )
        if components and int(components[0].sum()) < min_component_area:
            components = components[:1]
        else:
            components = [
                item for item in components if int(item.sum()) >= min_component_area
            ]
        for component in components:
            raw.append(
                {
                    "source": source,
                    "priority": priority,
                    "area": int(component.sum()),
                    "component": component,
                    "box": _mask_box(component),
                }
            )

    raw.sort(key=lambda item: (item["priority"], -item["area"]))
    selected: list[dict[str, Any]] = []
    for item in raw:
        duplicate = next(
            (
                existing
                for existing in selected
                if _box_overlap_fraction(existing["box"], item["box"]) >= 0.8
            ),
            None,
        )
        if duplicate is not None:
            duplicate["sources"].add(item["source"])
            duplicate["component_area"] = max(
                duplicate["component_area"], item["area"]
            )
            continue
        if len(selected) >= max_regions:
            continue
        selected.append(
            {
                "box": item["box"],
                "sources": {item["source"]},
                "component_area": item["area"],
            }
        )

    height, width = change.shape
    result: list[dict[str, Any]] = []
    candidate_delta = np.logical_or(candidate_added, candidate_removed)
    for index, item in enumerate(selected):
        crop_box = _padded_box(item["box"], (height, width), padding_ratio)
        x1, y1, x2, y2 = crop_box
        crop = np.zeros_like(change)
        crop[y1 : y2 + 1, x1 : x2 + 1] = True
        result.append(
            {
                "region_id": f"r{index}",
                "sources": sorted(item["sources"]),
                "component_area": int(item["component_area"]),
                "box_pixels": list(crop_box),
                "box_normalized": list(
                    pixel_box_to_normalized(crop_box, (width, height))
                ),
                "change_pixels": int(np.logical_and(change, crop).sum()),
                "temporal_difference_pixels": int(
                    np.logical_and(temporal_difference, crop).sum()
                ),
                "candidate_delta_pixels": int(
                    np.logical_and(candidate_delta, crop).sum()
                ),
                "candidate_added_pixels": int(
                    np.logical_and(candidate_added, crop).sum()
                ),
                "candidate_removed_pixels": int(
                    np.logical_and(candidate_removed, crop).sum()
                ),
                "t1_mask_pixels": int(np.logical_and(state.t1_mask, crop).sum()),
                "t2_mask_pixels": int(np.logical_and(state.t2_mask, crop).sum()),
            }
        )
    return result


def _mask_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("cannot build a box for an empty component")
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _padded_box(
    box: tuple[int, int, int, int],
    shape: tuple[int, int],
    padding_ratio: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    height, width = shape
    padding = max(4, round(max(x2 - x1 + 1, y2 - y1 + 1) * padding_ratio))
    return (
        max(0, x1 - padding),
        max(0, y1 - padding),
        min(width - 1, x2 + padding),
        min(height - 1, y2 + padding),
    )


def _box_overlap_fraction(
    left: tuple[int, int, int, int], right: tuple[int, int, int, int]
) -> float:
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    intersection = max(0, min(lx2, rx2) - max(lx1, rx1) + 1) * max(
        0, min(ly2, ry2) - max(ly1, ry1) + 1
    )
    if not intersection:
        return 0.0
    left_area = (lx2 - lx1 + 1) * (ly2 - ly1 + 1)
    right_area = (rx2 - rx1 + 1) * (ry2 - ry1 + 1)
    return intersection / min(left_area, right_area)
