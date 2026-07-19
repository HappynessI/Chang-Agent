"""Action dispatch with hard validation at the tool boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .state import AgentAction


class PointBackend(Protocol):
    def refine(
        self,
        image: np.ndarray,
        initial_mask: np.ndarray,
        coordinate: tuple[int, int],
        is_positive: bool,
        click_history: tuple[tuple[tuple[int, int], bool], ...] = (),
    ) -> np.ndarray: ...


class BoxBackend(Protocol):
    def segment_box(
        self,
        image: np.ndarray,
        box_cxcywh_normalized: tuple[float, float, float, float],
        query: str,
    ) -> np.ndarray: ...


@dataclass
class ExecutionResult:
    mask: np.ndarray
    evidence: dict[str, Any]


class ActionExecutor:
    def __init__(
        self,
        point_backend: PointBackend,
        box_backend: BoxBackend,
        *,
        point_roi_fraction: float = 0.125,
    ):
        if not 0 < point_roi_fraction <= 0.5:
            raise ValueError("point_roi_fraction must be in (0, 0.5]")
        self.point_backend = point_backend
        self.box_backend = box_backend
        self.point_roi_fraction = point_roi_fraction

    def execute(
        self,
        action: AgentAction,
        image: np.ndarray,
        initial_mask: np.ndarray,
        query: str,
        *,
        point_session_mask: np.ndarray | None = None,
        point_click_history: tuple[tuple[tuple[int, int], bool], ...] = (),
    ) -> ExecutionResult:
        height, width = image.shape[:2]
        if initial_mask.shape != (height, width):
            raise ValueError("initial mask and target image must have identical spatial size")

        if action.action in {"positive_point", "negative_point"}:
            if action.coordinate is None:
                raise ValueError("validated point action has no coordinate")
            x, y = action.coordinate
            if not (0 <= x < width and 0 <= y < height):
                raise ValueError("pixel coordinate is outside the target image")
            session_mask = (
                initial_mask
                if point_session_mask is None
                else np.asarray(point_session_mask, dtype=bool)
            )
            if session_mask.shape != initial_mask.shape:
                raise ValueError(
                    "point session mask and target mask must have identical spatial size"
                )
            raw_mask = self.point_backend.refine(
                image,
                session_mask,
                action.coordinate,
                action.action == "positive_point",
                point_click_history,
            )
            raw_mask = _validated_tool_mask(raw_mask, initial_mask.shape)
            roi = _point_roi(
                action.coordinate,
                (width, height),
                self.point_roi_fraction,
            )
            if action.action == "positive_point":
                component = _component_containing(raw_mask, action.coordinate)
                mask = np.logical_or(initial_mask, component)
                composition_mode = "merge_clicked_prediction_component"
            else:
                component = _component_containing(initial_mask, action.coordinate)
                mask = np.logical_and(initial_mask, ~component)
                composition_mode = "remove_clicked_initial_component"
            tool = "simpleclick"
            tool_input: dict[str, Any] = {
                "coordinate": list(action.coordinate),
                "is_positive": action.action == "positive_point",
                "accepted_click_history": [
                    {
                        "coordinate": list(history_coordinate),
                        "is_positive": history_is_positive,
                    }
                    for history_coordinate, history_is_positive in point_click_history
                ],
                "session_initial_mask_pixels": int(session_mask.sum()),
            }
        elif action.action == "box":
            if action.box is None:
                raise ValueError("validated box action has no box")
            cxcywh = xyxy_to_normalized_cxcywh(action.box, (width, height))
            raw_mask = self.box_backend.segment_box(image, cxcywh, query)
            raw_mask = _validated_tool_mask(raw_mask, initial_mask.shape)
            x1, y1, x2, y2 = action.box
            mask = np.array(initial_mask, dtype=bool, copy=True)
            mask[y1 : y2 + 1, x1 : x2 + 1] = raw_mask[
                y1 : y2 + 1, x1 : x2 + 1
            ]
            roi = action.box
            composition_mode = "replace_box_roi_only"
            tool = "sam3"
            tool_input = {
                "box_xyxy": list(action.box),
                "box_cxcywh_normalized": list(cxcywh),
            }
        else:
            raise ValueError("finish is handled by the environment, not an executor")

        result = np.asarray(mask, dtype=bool)
        if result.shape != initial_mask.shape:
            raise ValueError(f"tool returned mask shape {result.shape}, expected {initial_mask.shape}")
        backend = self.point_backend if tool == "simpleclick" else self.box_backend
        tool_result = getattr(backend, "last_evidence", None)
        evidence = {"tool": tool, "tool_input": tool_input}
        evidence.update(
            _locality_evidence(
                np.asarray(initial_mask, dtype=bool),
                result,
                roi,
                composition_mode,
            )
        )
        if tool_result:
            evidence["tool_result"] = tool_result
        return ExecutionResult(result, evidence)


def xyxy_to_normalized_cxcywh(
    box: tuple[int, int, int, int], image_size: tuple[int, int]
) -> tuple[float, float, float, float]:
    """Convert pixel XYXY to SAM3 normalized center-size format."""

    x1, y1, x2, y2 = box
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if not (0 <= x1 < x2 < width and 0 <= y1 < y2 < height):
        raise ValueError("box must be ordered and inside the image")
    return (
        ((x1 + x2) / 2.0) / width,
        ((y1 + y2) / 2.0) / height,
        (x2 - x1) / width,
        (y2 - y1) / height,
    )


def _point_roi(
    coordinate: tuple[int, int],
    image_size: tuple[int, int],
    fraction: float,
) -> tuple[int, int, int, int]:
    width, height = image_size
    radius = max(2, round(min(width, height) * fraction))
    x, y = coordinate
    return (
        max(0, x - radius),
        max(0, y - radius),
        min(width - 1, x + radius),
        min(height - 1, y + radius),
    )


def _validated_tool_mask(mask: np.ndarray, expected_shape: tuple[int, int]) -> np.ndarray:
    result = np.asarray(mask, dtype=bool)
    if result.shape != expected_shape:
        raise ValueError(
            f"tool returned mask shape {result.shape}, expected {expected_shape}"
        )
    return result


def _component_containing(
    mask: np.ndarray, coordinate: tuple[int, int]
) -> np.ndarray:
    source = np.asarray(mask, dtype=bool)
    height, width = source.shape
    x, y = coordinate
    component = np.zeros_like(source)
    if not (0 <= x < width and 0 <= y < height) or not source[y, x]:
        return component
    stack = [(x, y)]
    component[y, x] = True
    while stack:
        current_x, current_y = stack.pop()
        for next_x, next_y in (
            (current_x - 1, current_y),
            (current_x + 1, current_y),
            (current_x, current_y - 1),
            (current_x, current_y + 1),
        ):
            if (
                0 <= next_x < width
                and 0 <= next_y < height
                and source[next_y, next_x]
                and not component[next_y, next_x]
            ):
                component[next_y, next_x] = True
                stack.append((next_x, next_y))
    return component


def _locality_evidence(
    before: np.ndarray,
    after: np.ndarray,
    roi: tuple[int, int, int, int],
    composition_mode: str,
) -> dict[str, Any]:
    changed = np.logical_xor(before, after)
    x1, y1, x2, y2 = roi
    inside = np.zeros_like(changed)
    inside[y1 : y2 + 1, x1 : x2 + 1] = True
    changed_pixels = int(changed.sum())
    outside_pixels = int(np.logical_and(changed, ~inside).sum())
    before_components = _component_count(before)
    after_components = _component_count(after)
    largest_changed = max(
        (int(component.sum()) for component in _components(changed)),
        default=0,
    )
    return {
        "locality": {
            "composition_mode": composition_mode,
            "roi_xyxy": [x1, y1, x2, y2],
            "changed_pixels": changed_pixels,
            "outside_roi_pixels": outside_pixels,
            "outside_roi_ratio": (
                outside_pixels / changed_pixels if changed_pixels else 0.0
            ),
            "target_mask_change_ratio": changed_pixels / changed.size,
            "largest_changed_component_pixels": largest_changed,
            "components_before": before_components,
            "components_after": after_components,
            "component_count_delta": after_components - before_components,
        }
    }


def _components(mask: np.ndarray) -> list[np.ndarray]:
    source = np.asarray(mask, dtype=bool)
    remaining = np.array(source, copy=True)
    components: list[np.ndarray] = []
    while remaining.any():
        y, x = np.argwhere(remaining)[0]
        component = _component_containing(remaining, (int(x), int(y)))
        components.append(component)
        remaining[component] = False
    return components


def _component_count(mask: np.ndarray) -> int:
    return len(_components(mask))
