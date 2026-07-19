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
    progress_score: float
    error_type: str
    feedback: str


class Qwen3VLZeroShotVerifier:
    """Judge a candidate mask without GT and return safe structured feedback.

    Qwen first supplies absolute quality, pairwise progress, error type, and feedback.
    Actionable errors are localized in a separate request, while action/accept/stop
    fields are derived deterministically by the runtime.
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
        max_localization_area_ratio: float = 0.85,
        broad_region_delta_ratio: float = 0.5,
        min_false_positive_white_fraction: float = 0.01,
        max_false_negative_white_fraction: float = 0.5,
    ):
        if not 0 <= accept_threshold <= 1:
            raise ValueError("accept_threshold must be in [0, 1]")
        if max_retries < 1:
            raise ValueError("max_retries must be positive")
        if not 0 < max_localization_area_ratio <= 1:
            raise ValueError("max_localization_area_ratio must be in (0, 1]")
        if not 0 <= broad_region_delta_ratio <= 1:
            raise ValueError("broad_region_delta_ratio must be in [0, 1]")
        if not 0 <= min_false_positive_white_fraction <= 1:
            raise ValueError("min_false_positive_white_fraction must be in [0, 1]")
        if not 0 <= max_false_negative_white_fraction <= 1:
            raise ValueError("max_false_negative_white_fraction must be in [0, 1]")
        self.model = model
        self.processor = processor
        self.max_new_tokens = max_new_tokens
        self.accept_threshold = accept_threshold
        self.max_retries = max_retries
        self.max_localization_area_ratio = max_localization_area_ratio
        self.broad_region_delta_ratio = broad_region_delta_ratio
        self.min_false_positive_white_fraction = min_false_positive_white_fraction
        self.max_false_negative_white_fraction = max_false_negative_white_fraction
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
        previous_state: ChangeState | None = None,
    ) -> VerifierOutput:
        errors: list[str] = []
        primary_raw = ""
        for _ in range(self.max_retries):
            primary_raw = self._generate_raw(
                state,
                previous_score,
                previous_action,
                previous_state,
                errors[-1:],
            )
            try:
                payload = self._extract_json_object(primary_raw)
                diagnostic = self._parse_diagnostic_payload(payload)
                if previous_state is None and diagnostic.progress_score != 0.0:
                    raise ValueError(
                        "progress_score must be 0.0 when no previous valid state is shown"
                    )
            except (TypeError, ValueError, KeyError) as error:
                errors.append(str(error))
                continue

            if diagnostic.error_type == "none":
                output = self._derive_output(
                    diagnostic,
                    previous_score,
                    target_view=(previous_action.target_view if previous_action else "t2"),
                    error_region=None,
                    localization_valid=True,
                )
                return self._record_valid(
                    output,
                    {
                        "raw_output": primary_raw,
                        "parsed_output": payload,
                        "diagnostic_payload": payload,
                        "validation_errors": errors,
                    },
                )

            # Keep scoring separate from localization so the first stage can focus
            # on final-mask quality and pairwise progress.
            localization_errors: list[str] = []
            localization_raw = ""
            localization_attempts: list[dict[str, Any]] = []
            for _ in range(self.max_retries):
                localization_raw = self._generate_localization_raw(
                    state,
                    previous_state,
                    previous_action,
                    diagnostic,
                    localization_errors[-1:],
                )
                try:
                    localization_payload = self._extract_json_object(localization_raw)
                    target_view, region = self._parse_localization_payload(
                        localization_payload
                    )
                    checks = self._validate_localization(
                        state, previous_state, diagnostic.error_type, region
                    )
                    localization_attempts.append(
                        {"raw": localization_raw, "output": localization_payload}
                    )
                    output = self._derive_output(
                        diagnostic,
                        previous_score,
                        target_view=target_view,
                        error_region=region,
                        localization_valid=True,
                    )
                    return self._record_valid(
                        output,
                        {
                            "raw_output": primary_raw,
                            "parsed_output": payload,
                            "diagnostic_payload": payload,
                            "localization_raw": localization_raw,
                            "localization_output": localization_payload,
                            "localization_attempts": localization_attempts,
                            "localization_checks": checks,
                            "validation_errors": errors + localization_errors,
                        },
                    )
                except (TypeError, ValueError, KeyError) as error:
                    localization_errors.append(str(error))
                    localization_attempts.append(
                        {"raw": localization_raw, "error": str(error)}
                    )
            invalid = self._invalid_output(
                previous_score,
                diagnostic,
                errors + localization_errors,
                primary_raw,
                localization_raw,
                payload,
                previous_action,
            )
            self.last_evidence["localization_attempts"] = localization_attempts
            return invalid

        return self._invalid_output(
            previous_score, None, errors, primary_raw, None, None, previous_action
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
        previous_action: AgentAction | None,
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
            else previous_action.target_view if previous_action is not None else "t2"
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
            progress_score=0.0,
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
        target_view: str,
        error_region: tuple[int, int, int, int] | None,
        localization_valid: bool,
    ) -> VerifierOutput:
        if diagnostic.error_type == "none":
            # The runtime, not Qwen, owns the stop decision.
            suggested_action = "finish"
            region = None
            accept = diagnostic.quality_score >= self.accept_threshold
        else:
            if error_region is None:
                raise ValueError("actionable diagnosis requires a localized error_region")
            region = error_region
            suggested_action = self._suggested_action(diagnostic.error_type)
            accept = False
        delta = (
            0.0
            if previous_score is None
            else diagnostic.quality_score - previous_score
        )
        return VerifierOutput(
            quality_score=diagnostic.quality_score,
            progress_score=diagnostic.progress_score,
            score_delta=delta,
            error_type=diagnostic.error_type,
            target_view=target_view,
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
        previous_state: ChangeState | None,
        previous_errors: list[str],
    ) -> str:
        return self._generate_messages(
            self.build_messages(
                state, previous_score, previous_action, previous_state, previous_errors
            )
        )

    def _generate_localization_raw(
        self,
        state: ChangeState,
        previous_state: ChangeState | None,
        previous_action: AgentAction | None,
        diagnostic: _Diagnostic,
        previous_errors: list[str] | None = None,
    ) -> str:
        public_action = self._public_action(previous_action, state.image_size)
        correction = (
            f"Your previous localization was invalid: {previous_errors[-1]}. "
            "Return a smaller, semantically consistent region.\n"
            if previous_errors
            else ""
        )
        prompt = (
            "The first-stage verifier diagnosed an actionable error but did not localize it. "
            "Return exactly one JSON object with only target_view ('t1' or 't2') and "
            "error_region: [x1,y1,x2,y2]. "
            "Use normalized [0,1000] XYXY coordinates. Do not output quality, accept, "
            "action, or prose. The region should cover the suspected error in the candidate "
            "change mask. For false_positive_change, localize a region that mainly overlaps "
            "the current white change mask; for false_negative, localize a region mainly "
            "outside the current white mask.\n"
            f"error_type: {diagnostic.error_type}\n"
            f"diagnosis: {diagnostic.feedback}\n"
            f"Action that produced the candidate: "
            f"{json.dumps(public_action, ensure_ascii=False)}\n"
            f"{correction}"
        )
        return self._generate_messages(
            self._visual_messages(state, prompt, previous_state)
        )

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
        previous_state: ChangeState | None = None,
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
            "building change detection. Use these four explicit temporal relations: "
            "(1) added building = background in T1 and building in T2; "
            "(2) disappeared building = building in T1 and background in T2; "
            "(3) unchanged building = building in both T1 and T2; "
            "(4) unchanged background = background in both T1 and T2. "
            "Only added and disappeared buildings belong in the final change mask; both "
            "unchanged relations must be background in that mask. Avoid the ambiguous phrase "
            "'exists in only one image': identify whether the relation is added or disappeared. "
            "Judge the final candidate change mask first. Predicted T1/T2 object masks are "
            "supporting model outputs, not GT, and an empty T1 or T2 mask does not automatically "
            "mean an error. Compare the original images directly before deciding whether an "
            "empty temporal mask is plausible. Diagnose false positives and false negatives. "
            "Return exactly one JSON object with only quality_score (0.0 to 1.0), "
            "progress_score (-1.0 to 1.0), error_type ('none', "
            "'false_positive_change', 'false_negative', 'mixed_error', or 'uncertain_region'), "
            "and feedback (one concise sentence). quality_score measures the absolute quality "
            "of the candidate final change mask. progress_score independently compares the "
            "candidate with the previous valid state: positive means improved, negative means "
            "worse, and zero means no material change. If no previous valid state is shown, "
            "progress_score must be 0.0. Do not output target_view, error_region, accept, or "
            "suggested_action; actionable errors are localized in a separate stage. Do not use "
            "or assume GT.\n"
            f"Query: {state.query}\n"
            f"Candidate change-mask area ratio: {float(state.change_mask.mean()):.6f}\n"
            f"Previous valid change-mask area ratio: "
            f"{float(previous_state.change_mask.mean()) if previous_state is not None else None}\n"
            f"Matching summary: {json.dumps(matching, ensure_ascii=False, default=str)}\n"
            f"Previous score: {previous_score}\n"
            f"Action that produced the candidate (normalized public coordinates): "
            f"{json.dumps(previous, ensure_ascii=False)}\n"
            f"{correction}"
        )
        return self._visual_messages(state, prompt, previous_state)

    @staticmethod
    def _visual_messages(
        state: ChangeState,
        prompt: str,
        previous_state: ChangeState | None = None,
    ) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = [
            {"type": "text", "text": "Fixed T1 original image:"},
            {"type": "image", "image": Qwen3VLZeroShotVerifier._as_image(state.t1_image)},
            {"type": "text", "text": "Fixed T2 original image:"},
            {"type": "image", "image": Qwen3VLZeroShotVerifier._as_image(state.t2_image)},
        ]
        if previous_state is not None:
            content.extend(
                [
                    {"type": "text", "text": "Previous valid T1 object mask:"},
                    {"type": "image", "image": Qwen3VLZeroShotVerifier._mask_image(previous_state.t1_mask)},
                    {"type": "text", "text": "Previous valid T2 object mask:"},
                    {"type": "image", "image": Qwen3VLZeroShotVerifier._mask_image(previous_state.t2_mask)},
                    {"type": "text", "text": "Previous valid change mask:"},
                    {"type": "image", "image": Qwen3VLZeroShotVerifier._mask_image(previous_state.change_mask)},
                ]
            )
        content.extend(
            [
                {"type": "text", "text": "Candidate T1 object mask:"},
                {"type": "image", "image": Qwen3VLZeroShotVerifier._mask_image(state.t1_mask)},
                {"type": "text", "text": "Candidate T2 object mask:"},
                {"type": "image", "image": Qwen3VLZeroShotVerifier._mask_image(state.t2_mask)},
                {"type": "text", "text": "Candidate final change mask (primary evaluation target):"},
                {"type": "image", "image": Qwen3VLZeroShotVerifier._mask_image(state.change_mask)},
                {"type": "text", "text": prompt},
            ]
        )
        return [
            {
                "role": "user",
                "content": content,
            }
        ]

    def _parse_diagnostic_payload(self, payload: dict[str, Any]) -> _Diagnostic:
        required = {"quality_score", "progress_score", "error_type", "feedback"}
        missing = required - set(payload)
        if missing:
            raise ValueError(f"diagnostic fields missing: {sorted(missing)}")
        extra = set(payload) - required
        if extra:
            raise ValueError(f"unexpected diagnostic fields: {sorted(extra)}")
        score = payload["quality_score"]
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise TypeError("quality_score must be numeric")
        score = float(score)
        if not 0 <= score <= 1:
            raise ValueError("quality_score must be in [0, 1]")
        progress = payload["progress_score"]
        if isinstance(progress, bool) or not isinstance(progress, (int, float)):
            raise TypeError("progress_score must be numeric")
        progress = float(progress)
        if not -1 <= progress <= 1:
            raise ValueError("progress_score must be in [-1, 1]")
        error_type = payload["error_type"]
        if error_type not in self.ERROR_TYPES:
            raise ValueError("unsupported error_type")
        feedback = payload["feedback"]
        if not isinstance(feedback, str) or not feedback.strip():
            raise TypeError("feedback must be a non-empty string")
        return _Diagnostic(score, progress, error_type, feedback.strip())

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
    ) -> tuple[str, tuple[int, int, int, int]]:
        expected = {"target_view", "error_region"}
        extra = set(payload) - expected
        if extra:
            raise ValueError(f"unexpected localization fields: {sorted(extra)}")
        target_view = payload.get("target_view")
        if target_view not in self.VIEWS:
            raise ValueError("localization target_view must be t1 or t2")
        if "error_region" not in payload:
            raise ValueError("localization response must contain error_region")
        return target_view, self._parse_region(payload["error_region"])

    def _validate_localization(
        self,
        state: ChangeState,
        previous_state: ChangeState | None,
        error_type: str,
        region: tuple[int, int, int, int],
    ) -> dict[str, float]:
        width, height = state.image_size
        x1, y1, x2, y2 = region
        pixel_x1 = round(x1 / 1000 * (width - 1))
        pixel_y1 = round(y1 / 1000 * (height - 1))
        pixel_x2 = round(x2 / 1000 * (width - 1))
        pixel_y2 = round(y2 / 1000 * (height - 1))
        region_mask = np.zeros_like(state.change_mask)
        region_mask[pixel_y1 : pixel_y2 + 1, pixel_x1 : pixel_x2 + 1] = True
        region_area_ratio = float(region_mask.mean())
        if previous_state is None:
            delta = np.zeros_like(state.change_mask)
        else:
            delta = np.logical_xor(
                previous_state.change_mask, state.change_mask
            )
        delta_ratio = float(delta.mean())
        if (
            region_area_ratio >= self.max_localization_area_ratio
            and delta_ratio < self.broad_region_delta_ratio
        ):
            raise ValueError(
                "localization region is degenerate/full-image without a broad candidate delta"
            )
        white_fraction = float(state.change_mask[region_mask].mean())
        if (
            error_type == "false_positive_change"
            and white_fraction < self.min_false_positive_white_fraction
        ):
            raise ValueError(
                "false_positive_change region does not overlap enough white candidate change"
            )
        if (
            error_type == "false_negative"
            and white_fraction > self.max_false_negative_white_fraction
        ):
            raise ValueError(
                "false_negative region lies mostly inside the white candidate change mask"
            )
        return {
            "region_area_ratio": region_area_ratio,
            "candidate_delta_ratio": delta_ratio,
            "region_white_fraction": white_fraction,
        }

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
