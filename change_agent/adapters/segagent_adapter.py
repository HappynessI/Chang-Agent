"""Thin adapter around SegAgent's SimpleClick wrapper."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import numpy as np


class SimpleClickAdapter:
    """Execute one explicit click while preserving an external initial mask."""

    def __init__(self, segmentation_model: Any, segagent_root: str | Path | None = None):
        self.segmentation_model = segmentation_model
        self.segagent_root = Path(segagent_root).resolve() if segagent_root else None

    def _click_classes(self) -> tuple[type[Any], type[Any]]:
        if self.segagent_root is not None:
            root = str(self.segagent_root)
            if root not in sys.path:
                sys.path.insert(0, root)
        module = importlib.import_module(
            "third_party.SimpleClick.isegm.inference.clicker"
        )
        return module.Click, module.Clicker

    def refine(
        self,
        image: np.ndarray,
        initial_mask: np.ndarray,
        coordinate: tuple[int, int],
        is_positive: bool,
    ) -> np.ndarray:
        Click, Clicker = self._click_classes()
        x, y = coordinate
        clicker = Clicker()
        clicker.add_click(Click(is_positive=is_positive, coords=(y, x)))
        self.segmentation_model.set_input_image(np.asarray(image))
        prediction = self.segmentation_model.get_prediction(
            clicker, mask=np.asarray(initial_mask, dtype=np.float32)
        )
        if isinstance(prediction, tuple):
            prediction = prediction[0]
        return np.asarray(prediction, dtype=bool)

