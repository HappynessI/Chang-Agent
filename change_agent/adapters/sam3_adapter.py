"""Concrete adapter for OmniOVCD's ``SAM3ImageProcessor`` public API."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image


class SAM3ProcessorAdapter:
    """Use SAM3 text prompts for initialization and geometric prompts for boxes.

    The heavy SAM3 model/processor is constructed in the OmniOVCD environment and
    injected here. No private encoder state is exposed to the Agent.
    """

    def __init__(self, processor: Any, mask_threshold: float = 0.0):
        self.processor = processor
        self.mask_threshold = mask_threshold

    def initialize_masks(
        self, t1_image: np.ndarray, t2_image: np.ndarray, query: str
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        t1_mask, t1_evidence = self._text_segment(t1_image, query)
        t2_mask, t2_evidence = self._text_segment(t2_image, query)
        evidence = {
            "sam3_t1": t1_evidence,
            "sam3_t2": t2_evidence,
        }
        confidence1 = t1_evidence.get("confidence_map")
        confidence2 = t2_evidence.get("confidence_map")
        if confidence1 is not None and confidence2 is not None:
            evidence["change_confidence"] = np.maximum(confidence1, confidence2)
        return t1_mask, t2_mask, evidence

    def segment_box(
        self,
        image: np.ndarray,
        box_cxcywh_normalized: tuple[float, float, float, float],
        query: str,
    ) -> np.ndarray:
        state = self.processor.set_image(self._to_pil(image))
        self.processor.reset_all_prompts(state)
        state = self.processor.set_text_prompt(prompt=query, state=state)
        state = self.processor.add_geometric_prompt(
            box=list(box_cxcywh_normalized), label=True, state=state
        )
        return self._mask_from_state(state, image.shape[:2])[0]

    def _text_segment(
        self, image: np.ndarray, query: str
    ) -> tuple[np.ndarray, dict[str, Any]]:
        state = self.processor.set_image(self._to_pil(image))
        self.processor.reset_all_prompts(state)
        state = self.processor.set_text_prompt(prompt=query, state=state)
        mask, confidence = self._mask_from_state(state, image.shape[:2])
        evidence: dict[str, Any] = {"confidence_map": confidence}
        for key in ("presence_score", "object_score", "scores"):
            if key in state:
                evidence[key] = self._numpy(state[key])
        return mask, evidence

    def _mask_from_state(
        self, state: dict[str, Any], shape: tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray]:
        height, width = shape
        logits = state.get("semantic_mask_logits")
        if logits is None:
            logits = state.get("masks_logits")
        if logits is None:
            masks = state.get("masks")
            if masks is None:
                raise KeyError("SAM3 state contains no semantic/instance mask output")
            array = self._numpy(masks).astype(float)
        else:
            array = self._numpy(logits).astype(float)
        while array.ndim > 2:
            array = array.max(axis=0)
        if array.shape != (height, width):
            image = Image.fromarray(array.astype(np.float32), mode="F")
            array = np.asarray(image.resize((width, height), resample=Image.Resampling.BILINEAR))
        mask = array > self.mask_threshold
        # Logits are converted to a bounded confidence map without assuming calibration.
        confidence = 1.0 / (1.0 + np.exp(-np.clip(array, -30, 30)))
        return mask.astype(bool), confidence.astype(np.float32)

    @staticmethod
    def _numpy(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value)

    @staticmethod
    def _to_pil(image: np.ndarray) -> Image.Image:
        array = np.asarray(image)
        if array.dtype != np.uint8:
            if array.max(initial=0) <= 1:
                array = array * 255
            array = np.clip(array, 0, 255).astype(np.uint8)
        return Image.fromarray(array)

