"""GT-free zero-shot Change Verifier with staged, program-derived actions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from ..coordinates import (
    pixel_box_to_normalized,
    pixel_point_to_normalized,
    validate_normalized_box,
)
from ..state import AgentAction, ChangeState, VerifierOutput


@dataclass(frozen=True)
class _Diagnostic:
    quality_score: float
    error_type: str
    target_view: str
    error_region: tuple[int, int, int, int] | None
    feedback: str


class Qwen3VLZeroShotVerifier:
    """Judge a candidate mask without GT and return safe structured feedback.

    Qwen supplies only the visual diagnosis. Localization is requested separately
    when an actionable diagnosis has no region, and action/accept/stop fields are
    derived deterministically by the runtime.
    """

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
        self._last_valid_output: VerifierOutput | None = None

    def reset(self) -> None:
        """Clear retained feedback when a new Environment episode starts."""

        self._last_valid_output = None
        self.last_evidence = {}

    def on_candidate_rejected(self, previous_feedback: VerifierOutput) -> None:
        """Restore retained feedback to the last accepted Environment state."""

        self._last_valid_output = (
            previous_feedback if previous_feedback.verifier_valid else None
        )

    def verify(
        self,
        state: ChangeState,
        previous_score: float | None,
        previous_action: AgentAction | None,
    ) -> VerifierOutput:
        errors: list[str] = []
        primary_raw = ""
        for _ in range(self.max_retries):
            primary_raw = self._generate_raw(state, previous_score, previous_action, errors[-1:])
            try:
                payload = self._extract_json_object(primary_raw)
                diagnostic = self._parse_diagnostic_payload(payload)
            except (TypeError, ValueError, KeyError) as error:
                errors.append(str(error))
                continue

            if diagnostic.error_type == "none":
                output = self._derive_output(diagnostic, previous_score, localization_valid=True)
                return self._record_valid(
                    output,
                    {
                        "raw_output": primary_raw,
                        "parsed_output": payload,
                        "diagnostic_payload": payload,
                        "validation_errors": errors,
                    },
                )

            if diagnostic.error_region is not None:
                output = self._derive_output(diagnostic, previous_score, localization_valid=True)
                return self._record_valid(
                    output,
                    {
                        "raw_output": primary_raw,
                        "parsed_output": payload,
                        "diagnostic_payload": payload,
                        "validation_errors": errors,
                    },
                )

            # Keep the semantic diagnosis and ask a much smaller second question:
            # only localize the already identified error.
            localization_errors: list[str] = []
            localization_raw = self._generate_localization_raw(state, diagnostic)
            try:
                localization_payload = self._extract_json_object(localization_raw)
                region = self._parse_localization_payload(localization_payload)
                localized = _Diagnostic(
                    diagnostic.quality_score,
                    diagnostic.error_type,
                    diagnostic.target_view,
                    region,
                    diagnostic.feedback,
                )
                output = self._derive_output(
                    localized, previous_score, localization_valid=True
                )
                return self._record_valid(
                    output,
                    {
                        "raw_output": primary_raw,
                        "parsed_output": payload,
                        "diagnostic_payload": payload,
                        "localization_raw": localization_raw,
                        "localization_output": localization_payload,
                        "validation_errors": errors,
                    },
                )
            except (TypeError, ValueError, KeyError) as error:
                localization_errors.append(str(error))
                return self._invalid_output(
                    previous_score,
                    diagnostic,
                    errors + localization_errors,
                    primary_raw,
                    localization_raw,
                    payload,
                )

        return self._invalid_output(
            previous_score, None, errors, primary_raw, None, None
        )

    def _record_valid(
        self, output: VerifierOutput, evidence: dict[str, Any]
    ) -> VerifierOutput:
        self._last_valid_output = output
        self.last_evidence = {
            "type": "qwen3vl_zero_shot",
            "gt_available": False,
            "verifier_valid": True,
            "localization_valid": output.localization_valid,
            **evidence,
        }
        return output

    def _invalid_output(
        self,
        previous_score: float | None,
        diagnostic: _Diagnostic | None,
        errors: list[str],
        primary_raw: str,
        localization_raw: str | None,
        diagnostic_payload: dict[str, Any] | None,
    ) -> VerifierOutput:
        previous = self._last_valid_output
        retained = previous.feedback if previous is not None else None
        current = diagnostic.feedback if diagnostic is not None else None
        messages = ["Verifier invalid; no action is authorized; recheck required."]
        if current:
            messages.append(f"Current diagnosis retained: {current}")
        if retained:
            messages.append(f"Previous valid feedback retained: {retained}")
        base_score = (
            previous.quality_score
            if previous is not None
            else diagnostic.quality_score if diagnostic is not None else previous_score or 0.0
        )
        base_type = (
            previous.error_type
            if previous is not None
            else diagnostic.error_type if diagnostic is not None else "uncertain_region"
        )
        base_view = (
            previous.target_view
            if previous is not None
            else diagnostic.target_view if diagnostic is not None else "t2"
        )
        self.last_evidence = {
            "type": "qwen3vl_zero_shot",
            "raw_output": primary_raw,
            "diagnostic_payload": diagnostic_payload,
            "localization_raw": localization_raw,
            "validation_errors": errors,
            "fallback": True,
            "verifier_valid": False,
            "localization_valid": False,
            "gt_available": False,
        }
        return VerifierOutput(
            quality_score=float(base_score),
            score_delta=0.0,
            error_type=base_type,
            target_view=base_view,
            error_region=previous.error_region if previous is not None else None,
            suggested_action=None,
            feedback=" ".join(messages),
            accept=False,
            verifier_valid=False,
            localization_valid=False,
            stop=False,
        )

    def _derive_output(
        self,
        diagnostic: _Diagnostic,
        previous_score: float | None,
        *,
        localization_valid: bool,
    ) -> VerifierOutput:
        if diagnostic.error_type == "none":
            # The runtime, not Qwen, owns the stop decision.
            suggested_action = "finish"
            region = None
            accept = diagnostic.quality_score >= self.accept_threshold
        else:
            if diagnostic.error_region is None:
                raise ValueError("actionable diagnosis requires a localized error_region")
            region = diagnostic.error_region
            suggested_action = self._suggested_action(diagnostic.error_type)
            accept = False
        delta = (
            0.0
            if previous_score is None
            else diagnostic.quality_score - previous_score
        )
        return VerifierOutput(
            quality_score=diagnostic.quality_score,
            score_delta=delta,
            error_type=diagnostic.error_type,
            target_view=diagnostic.target_view,
            error_region=region,
            suggested_action=suggested_action,
            feedback=diagnostic.feedback,
            accept=accept,
            verifier_valid=True,
            localization_valid=localization_valid,
            stop=accept,
        )

    @staticmethod
    def _suggested_action(error_type: str) -> str:
        if error_type == "false_positive_change":
            return "negative_point"
        if error_type == "false_negative":
            return "positive_point"
        return "box"

    def _generate_raw(
        self,
        state: ChangeState,
        previous_score: float | None,
        previous_action: AgentAction | None,
        previous_errors: list[str],
    ) -> str:
        return self._generate_messages(
            self.build_messages(state, previous_score, previous_action, previous_errors)
        )

    def _generate_localization_raw(
        self, state: ChangeState, diagnostic: _Diagnostic
    ) -> str:
        prompt = (
            "The first-stage verifier diagnosed an actionable error but did not localize it. "
            "Return exactly one JSON object with only error_region: [x1,y1,x2,y2]. "
            "Use normalized [0,1000] XYXY coordinates. Do not output quality, accept, "
            "action, or prose. The region should cover the suspected error in the candidate "
            "change mask. For false_positive_change, localize a region that mainly overlaps "
            "the current white change mask; for false_negative, localize a region mainly "
            "outside the current white mask.\n"
            f"error_type: {diagnostic.error_type}\n"
            f"target_view: {diagnostic.target_view}\n"
            f"diagnosis: {diagnostic.feedback}"
        )
        return self._generate_messages(self._visual_messages(state, prompt))

    def _generate_messages(self, messages: list[dict[str, Any]]) -> str:
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
            f"Your previous diagnostic was invalid: {previous_errors[-1]}. Correct it.\n"
            if previous_errors
            else ""
        )
        prompt = (
            "You are the diagnostic stage of a zero-shot, ground-truth-free verifier for "
            "building change detection. Compare T1 and T2 directly and judge whether the "
            "white regions in the candidate change mask represent real temporal building "
            "changes, not merely buildings present in one image. Diagnose false positives "
            "and false negatives. Select target_view as the temporal semantic mask that "
            "should be edited; do not alternate views by rule. All regions use normalized "
            "[0,1000] XYXY coordinates, never image pixels. Return exactly one JSON object. "
            "The predicted T1/T2 object masks are model outputs, not GT; the current change "
            "mask is reconstructed from those masks and OmniOVCD matching evidence. Use all "
            "five visual inputs to attribute additions, disappearances, and mismatches. "
            "Return only quality_score (0.0 to 1.0), error_type ('none', "
            "'false_positive_change', 'false_negative', 'mixed_error', or 'uncertain_region'), "
            "target_view ('t1' or 't2'), error_region ([x1,y1,x2,y2] or null), and feedback "
            "(one concise sentence). If an error exists but its location is unclear, set "
            "error_region to null; a separate localization request will follow. Do not "
            "output accept or suggested_action. For false_positive_change, localize a "
            "region that mainly overlaps the current white change mask; for false_negative, "
            "localize a region mainly outside the current white mask. Do not use or assume GT.\n"
            f"Query: {state.query}\n"
            f"Change-mask area ratio: {float(state.change_mask.mean()):.6f}\n"
            f"Matching summary: {json.dumps(matching, ensure_ascii=False, default=str)}\n"
            f"Previous score: {previous_score}\n"
            f"Previous action (normalized public coordinates): "
            f"{json.dumps(previous, ensure_ascii=False)}\n"
            f"{correction}"
        )
        return self._visual_messages(state, prompt)

    @staticmethod
    def _visual_messages(state: ChangeState, prompt: str) -> list[dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "T1 original image:"},
                    {"type": "image", "image": Qwen3VLZeroShotVerifier._as_image(state.t1_image)},
                    {"type": "text", "text": "T2 original image:"},
                    {"type": "image", "image": Qwen3VLZeroShotVerifier._as_image(state.t2_image)},
                    {"type": "text", "text": "Predicted T1 object mask:"},
                    {"type": "image", "image": Qwen3VLZeroShotVerifier._mask_image(state.t1_mask)},
                    {"type": "text", "text": "Predicted T2 object mask:"},
                    {"type": "image", "image": Qwen3VLZeroShotVerifier._mask_image(state.t2_mask)},
                    {"type": "text", "text": "Current change mask:"},
                    {"type": "image", "image": Qwen3VLZeroShotVerifier._mask_image(state.change_mask)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    def _parse_diagnostic_payload(self, payload: dict[str, Any]) -> _Diagnostic:
        required = {"quality_score", "error_type", "target_view", "feedback"}
        missing = required - set(payload)
        if missing:
            raise ValueError(f"diagnostic fields missing: {sorted(missing)}")
        score = payload["quality_score"]
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise TypeError("quality_score must be numeric")
        score = float(score)
        if not 0 <= score <= 1:
            raise ValueError("quality_score must be in [0, 1]")
        error_type = payload["error_type"]
        target_view = payload["target_view"]
        if error_type not in self.ERROR_TYPES:
            raise ValueError("unsupported error_type")
        if target_view not in self.VIEWS:
            raise ValueError("target_view must be t1 or t2")
        feedback = payload["feedback"]
        if not isinstance(feedback, str) or not feedback.strip():
            raise TypeError("feedback must be a non-empty string")
        region = None
        region_value = payload.get("error_region")
        if region_value is not None:
            region = self._parse_region(region_value)
        return _Diagnostic(score, error_type, target_view, region, feedback.strip())

    @staticmethod
    def _parse_region(region_value: Any) -> tuple[int, int, int, int]:
        if not isinstance(region_value, (list, tuple)) or len(region_value) != 4:
            raise ValueError("error_region must be null or four normalized coordinates")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in region_value
        ):
            raise TypeError("error_region values must be numeric")
        return validate_normalized_box(tuple(round(float(value)) for value in region_value))

    def _parse_localization_payload(
        self, payload: dict[str, Any]
    ) -> tuple[int, int, int, int]:
        if "error_region" not in payload:
            raise ValueError("localization response must contain error_region")
        return self._parse_region(payload["error_region"])

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
