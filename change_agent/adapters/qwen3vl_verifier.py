"""GT-free zero-shot Change Verifier backed by a shared Qwen3-VL model."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from PIL import Image

from ..coordinates import (
    pixel_box_to_normalized,
    pixel_point_to_normalized,
    validate_normalized_box,
)
from ..state import AgentAction, ChangeState, VerifierOutput


class Qwen3VLZeroShotVerifier:
    """Judge a candidate mask without GT and return structured actionable feedback."""

    ERROR_TYPES = {
        "none",
        "false_positive_change",
        "false_negative",
        "mixed_error",
        "uncertain_region",
    }
    ACTIONS = {"positive_point", "negative_point", "box", "finish"}
    VIEWS = {"t1", "t2"}

    def __init__(
        self,
        *,
        model: Any,
        processor: Any,
        max_new_tokens: int = 256,
        accept_threshold: float = 0.82,
        max_retries: int = 2,
    ):
        if not 0 <= accept_threshold <= 1:
            raise ValueError("accept_threshold must be in [0, 1]")
        if max_retries < 1:
            raise ValueError("max_retries must be positive")
        self.model = model
        self.processor = processor
        self.max_new_tokens = max_new_tokens
        self.accept_threshold = accept_threshold
        self.max_retries = max_retries
        self.last_evidence: dict[str, Any] = {}

    def verify(
        self,
        state: ChangeState,
        previous_score: float | None,
        previous_action: AgentAction | None,
    ) -> VerifierOutput:
        errors: list[str] = []
        raw = ""
        for _ in range(self.max_retries):
            raw = self._generate_raw(state, previous_score, previous_action, errors[-1:])
            try:
                payload = self._extract_json_object(raw)
                output = self._parse_payload(payload, previous_score)
                self.last_evidence = {
                    "type": "qwen3vl_zero_shot",
                    "raw_output": raw,
                    "parsed_output": payload,
                    "validation_errors": errors,
                    "gt_available": False,
                }
                return output
            except (TypeError, ValueError, KeyError) as error:
                errors.append(str(error))
        self.last_evidence = {
            "type": "qwen3vl_zero_shot",
            "raw_output": raw,
            "validation_errors": errors,
            "gt_available": False,
        }
        raise ValueError(f"Qwen3-VL Verifier returned no valid output: {errors}")

    def _generate_raw(
        self,
        state: ChangeState,
        previous_score: float | None,
        previous_action: AgentAction | None,
        previous_errors: list[str],
    ) -> str:
        messages = self.build_messages(state, previous_score, previous_action, previous_errors)
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

    def build_messages(
        self,
        state: ChangeState,
        previous_score: float | None,
        previous_action: AgentAction | None,
        previous_errors: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        previous = self._public_action(previous_action, state.image_size)
        matching = state.evidence.get("matching", {})
        correction = (
            f"Your previous output was invalid: {previous_errors[-1]}. Correct it."
            if previous_errors
            else ""
        )
        prompt = (
            "You are a zero-shot, ground-truth-free verifier for building change detection. "
            "Compare T1 and T2 directly and judge whether the white regions in the candidate "
            "change mask represent real temporal building changes, not merely buildings present "
            "in one image. Diagnose false positives and false negatives. Select target_view as "
            "the temporal semantic mask that should be edited; do not alternate views by rule. "
            "All regions use normalized [0,1000] XYXY coordinates, never image pixels. "
            "Return exactly one JSON object with: quality_score (0.0 to 1.0), error_type "
            "('none', 'false_positive_change', 'false_negative', 'mixed_error', or "
            "'uncertain_region'), target_view ('t1' or 't2'), error_region ([x1,y1,x2,y2] "
            "or null), suggested_action ('positive_point', 'negative_point', 'box', or "
            "'finish'), feedback (one concise sentence), and accept (boolean). Only use "
            "error_type='none', suggested_action='finish', and accept=true when the mask is "
            "credible. Do not use or assume GT.\n"
            f"Query: {state.query}\n"
            f"Change-mask area ratio: {float(state.change_mask.mean()):.6f}\n"
            f"Matching summary: {json.dumps(matching, ensure_ascii=False, default=str)}\n"
            f"Previous score: {previous_score}\n"
            f"Previous action (normalized public coordinates): "
            f"{json.dumps(previous, ensure_ascii=False)}\n"
            f"{correction}"
        )
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "T1 image (earlier time):"},
                    {"type": "image", "image": self._as_image(state.t1_image)},
                    {"type": "text", "text": "T2 image (later time):"},
                    {"type": "image", "image": self._as_image(state.t2_image)},
                    {"type": "text", "text": "Candidate binary change mask:"},
                    {"type": "image", "image": self._mask_image(state.change_mask)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    def _parse_payload(
        self, payload: dict[str, Any], previous_score: float | None
    ) -> VerifierOutput:
        required = {
            "quality_score",
            "error_type",
            "target_view",
            "error_region",
            "suggested_action",
            "feedback",
            "accept",
        }
        if set(payload) != required:
            raise ValueError(
                f"verifier fields must be exactly {sorted(required)}; got {sorted(payload)}"
            )
        score = payload["quality_score"]
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise TypeError("quality_score must be numeric")
        score = float(score)
        if not 0 <= score <= 1:
            raise ValueError("quality_score must be in [0, 1]")
        error_type = payload["error_type"]
        target_view = payload["target_view"]
        suggested_action = payload["suggested_action"]
        if error_type not in self.ERROR_TYPES:
            raise ValueError("unsupported error_type")
        if target_view not in self.VIEWS:
            raise ValueError("target_view must be t1 or t2")
        if suggested_action not in self.ACTIONS:
            raise ValueError("unsupported suggested_action")
        region_value = payload["error_region"]
        region = None
        if region_value is not None:
            if not isinstance(region_value, (list, tuple)) or len(region_value) != 4:
                raise ValueError("error_region must be null or four normalized coordinates")
            if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in region_value):
                raise TypeError("error_region values must be numeric")
            region = validate_normalized_box(tuple(round(float(value)) for value in region_value))
        feedback = payload["feedback"]
        if not isinstance(feedback, str) or not feedback.strip():
            raise TypeError("feedback must be a non-empty string")
        if not isinstance(payload["accept"], bool):
            raise TypeError("accept must be boolean")
        accept = (
            payload["accept"]
            and score >= self.accept_threshold
            and error_type == "none"
            and suggested_action == "finish"
        )
        if not accept and suggested_action != "finish" and region is None:
            raise ValueError("an actionable non-finish diagnosis requires error_region")
        delta = 0.0 if previous_score is None else score - previous_score
        return VerifierOutput(
            quality_score=score,
            score_delta=delta,
            error_type=error_type,
            target_view=target_view,
            error_region=region,
            suggested_action=suggested_action,
            feedback=feedback.strip(),
            accept=accept,
        )

    @staticmethod
    def _extract_json_object(raw: str) -> dict[str, Any]:
        decoder = json.JSONDecoder()
        for index, character in enumerate(raw):
            if character != "{":
                continue
            try:
                value, _ = decoder.raw_decode(raw[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
        raise ValueError("verifier response contains no JSON object")

    @staticmethod
    def _public_action(
        action: AgentAction | None, image_size: tuple[int, int]
    ) -> dict[str, Any] | None:
        if action is None:
            return None
        result: dict[str, Any] = {
            "target_view": action.target_view,
            "action": action.action,
            "coordinate_space": "normalized_0_1000",
        }
        if action.coordinate is not None:
            result["coordinate"] = list(
                pixel_point_to_normalized(action.coordinate, image_size)
            )
        if action.box is not None:
            result["box"] = list(pixel_box_to_normalized(action.box, image_size))
        return result

    @staticmethod
    def _as_image(value: Any) -> Image.Image:
        if isinstance(value, Image.Image):
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
