"""Qwen3-VL Agent adapter using the modern multimodal chat-template API."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from PIL import Image

from ..action_parser import ActionParser
from ..state import AgentAction, AgentObservation


class GroundingModelQwen3VL:
    DEFAULT_MODEL = "Qwen/Qwen3-VL-2B-Instruct"

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        *,
        max_new_tokens: int = 160,
        device_map: str | None = "auto",
        dtype: str = "auto",
        model: Any | None = None,
        processor: Any | None = None,
        action_parser: ActionParser | None = None,
    ):
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.device_map = device_map
        self.dtype = dtype
        self.action_parser = action_parser or ActionParser()
        if (model is None) != (processor is None):
            raise ValueError("model and processor must be supplied together")
        if model is None:
            try:
                from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
            except (ImportError, AttributeError) as error:
                raise RuntimeError(
                    "Qwen3-VL requires the isolated qwen3vl dependencies "
                    "(transformers>=4.57.0 and accelerate)."
                ) from error
            load_kwargs: dict[str, Any] = {"dtype": dtype}
            # Transformers accepts ``device_map='auto'`` for CUDA placement, while
            # a CPU-only smoke should use the ordinary single-process loader.
            if device_map != "cpu":
                load_kwargs["device_map"] = device_map
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path, **load_kwargs
            ).eval()
            processor = AutoProcessor.from_pretrained(model_path)
        self.model = model
        self.processor = processor

    def build_messages(self, observation: AgentObservation) -> list[dict[str, Any]]:
        prompt = self._instruction(observation)
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "T1 image (earlier time):"},
                    {"type": "image", "image": self._as_image(observation.t1_image)},
                    {"type": "text", "text": "T2 image (later time):"},
                    {"type": "image", "image": self._as_image(observation.t2_image)},
                    {"type": "text", "text": "Current binary change mask:"},
                    {"type": "image", "image": self._mask_image(observation.change_mask)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    def generate_raw(self, observation: AgentObservation) -> str:
        messages = self.build_messages(observation)
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        device = getattr(self.model, "device", None)
        if device is not None and hasattr(inputs, "to"):
            inputs = inputs.to(device)
        outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        input_ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids
        generated = outputs[:, input_ids.shape[1] :]
        return self.processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

    def act(self, observation: AgentObservation) -> tuple[str, AgentAction]:
        raw = self.generate_raw(observation)
        height, width = observation.t1_image.shape[:2]
        return raw, self.action_parser.parse(raw, (width, height))

    @staticmethod
    def _as_image(value: Any) -> Any:
        if isinstance(value, Image.Image) or isinstance(value, str):
            return value
        array = np.asarray(value)
        if array.dtype != np.uint8:
            if array.max(initial=0) <= 1:
                array = array * 255
            array = np.clip(array, 0, 255).astype(np.uint8)
        return Image.fromarray(array)

    @staticmethod
    def _mask_image(mask: np.ndarray) -> Image.Image:
        return Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255, mode="L")

    @staticmethod
    def _instruction(observation: AgentObservation) -> str:
        feedback = observation.feedback.to_dict() if observation.feedback else None
        history = observation.history_summary or "none"
        has_tool_action = any(
            f"action={name}" in history
            for name in ("positive_point", "negative_point", "box")
        )
        finish_rule = (
            "At least one segmentation tool action has already run; finish is allowed only "
            "if the current mask is credible."
            if has_tool_action
            else "No segmentation tool action has run yet. Finish is forbidden: choose a "
            "positive_point, negative_point, or box action now."
        )
        return (
            "You refine a change-detection result. The three inputs above are explicitly "
            "T1, T2, and the current change mask. Do not invent a final mask. Select one "
            "tool action. All public coordinates, including Verifier error_region and your "
            "output, use normalized [0,1000] XY order; they are not image pixels. For a "
            "256x256 image, pixel center (128,128) is approximately (502,502). Return exactly one "
            "JSON object with target_view ('t1' or 't2') and action "
            "('positive_point', 'negative_point', 'box', or 'finish'). Point actions require "
            "coordinate:[x,y]; box requires box:[x1,y1,x2,y2]; finish requires neither.\n"
            f"{finish_rule}\n"
            f"Query: {observation.query}\n"
            f"Verifier feedback: {json.dumps(feedback, ensure_ascii=False)}\n"
            f"History summary: {history}"
        )
