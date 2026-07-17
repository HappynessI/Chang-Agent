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
    def __init__(self, point_backend: PointBackend, box_backend: BoxBackend):
        self.point_backend = point_backend
        self.box_backend = box_backend

    def execute(
        self,
        action: AgentAction,
        image: np.ndarray,
        initial_mask: np.ndarray,
        query: str,
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
            mask = self.point_backend.refine(
                image,
                initial_mask,
                action.coordinate,
                action.action == "positive_point",
            )
            tool = "simpleclick"
            tool_input: dict[str, Any] = {
                "coordinate": list(action.coordinate),
                "is_positive": action.action == "positive_point",
            }
        elif action.action == "box":
            if action.box is None:
                raise ValueError("validated box action has no box")
            cxcywh = xyxy_to_normalized_cxcywh(action.box, (width, height))
            mask = self.box_backend.segment_box(image, cxcywh, query)
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
        return ExecutionResult(result, {"tool": tool, "tool_input": tool_input})


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

