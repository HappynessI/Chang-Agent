"""Local and hosted model backends for the staged verifier protocol."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Mapping

import numpy as np
from PIL import Image

from ..state import ChangeState
from ..verifier_protocol import StageName, StageProtocolError


class LocalQwen3VLStageBackend:
    """Use already-loaded Transformers Qwen weights for staged JSON calls."""

    def __init__(
        self,
        *,
        model: Any,
        processor: Any,
        max_new_tokens: int = 512,
        do_sample: bool = False,
        repetition_penalty: float = 1.05,
    ):
        self.model = model
        self.processor = processor
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.repetition_penalty = repetition_penalty
        self.last_call: dict[str, Any] = {}
        self.call_history: list[dict[str, Any]] = []

    def reset_audit(self) -> None:
        self.last_call = {}
        self.call_history = []

    def generate_stage(
        self,
        stage: StageName,
        state: ChangeState,
        payload: Mapping[str, Any],
        previous_state: ChangeState | None = None,
    ) -> Mapping[str, Any]:
        return self._complete_stage(stage, state, payload, previous_state, None)

    def repair_stage(
        self,
        stage: StageName,
        state: ChangeState,
        payload: Mapping[str, Any],
        validation_error: str,
        previous_state: ChangeState | None = None,
    ) -> Mapping[str, Any]:
        """Retry one stage with the exact validation failure in the prompt."""

        return self._complete_stage(
            stage, state, payload, previous_state, validation_error
        )

    def _complete_stage(
        self,
        stage: StageName,
        state: ChangeState,
        payload: Mapping[str, Any],
        previous_state: ChangeState | None,
        validation_error: str | None,
    ) -> Mapping[str, Any]:
        messages = _local_messages(
            stage, state, payload, previous_state, validation_error
        )
        prompt = _stage_prompt(stage, payload, validation_error)
        started = time.monotonic()
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
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            repetition_penalty=self.repetition_penalty,
        )
        input_ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids
        generated = outputs[:, input_ids.shape[1] :]
        raw = self.processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        audit = {
            "backend": "local_transformers",
            "stage": stage,
            "latency_seconds": round(time.monotonic() - started, 6),
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "attempt_kind": "repair" if validation_error else "initial",
            "raw_response": _audit_text(raw),
        }
        try:
            result = _extract_stage_json(raw, stage)
        except StageProtocolError as error:
            audit["parse_error"] = str(error)
            self.last_call = audit
            self.call_history.append(audit)
            raise
        audit["parsed_output"] = result
        self.last_call = audit
        self.call_history.append(audit)
        return result


class BailianQwen3VLStageBackend:
    """Minimal OpenAI-compatible BaiLian client with no credential logging.

    The API key is read only from ``api_key_env``.  ``base_url`` may be a full
    ``.../chat/completions`` URL or an OpenAI-compatible ``.../v1`` base URL.
    Network calls are made only when ``generate_stage`` is explicitly invoked.
    """

    DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def __init__(
        self,
        *,
        model: str = "qwen3-vl-plus",
        base_url: str | None = None,
        api_key_env: str = "DASHSCOPE_API_KEY",
        timeout_seconds: float = 120.0,
        max_completion_tokens: int = 512,
        seed: int = 42,
        opener: Any | None = None,
    ):
        self.model = model
        self.base_url = (base_url or os.environ.get("DASHSCOPE_BASE_URL") or self.DEFAULT_BASE_URL).rstrip("/")
        self.api_key_env = api_key_env
        self.timeout_seconds = timeout_seconds
        self.max_completion_tokens = max_completion_tokens
        self.seed = seed
        self.opener = opener or urllib.request.urlopen
        self.last_call: dict[str, Any] = {}
        self.call_history: list[dict[str, Any]] = []

    def reset_audit(self) -> None:
        self.last_call = {}
        self.call_history = []

    @property
    def endpoint(self) -> str:
        return (
            self.base_url
            if self.base_url.endswith("/chat/completions")
            else self.base_url + "/chat/completions"
        )

    def generate_stage(
        self,
        stage: StageName,
        state: ChangeState,
        payload: Mapping[str, Any],
        previous_state: ChangeState | None = None,
    ) -> Mapping[str, Any]:
        messages = _hosted_messages(stage, state, payload, previous_state, None)
        prompt = _stage_prompt(stage, payload, None)
        return self.complete_json(messages, prompt=prompt, call_kind=stage)

    def repair_stage(
        self,
        stage: StageName,
        state: ChangeState,
        payload: Mapping[str, Any],
        validation_error: str,
        previous_state: ChangeState | None = None,
    ) -> Mapping[str, Any]:
        messages = _hosted_messages(
            stage, state, payload, previous_state, validation_error
        )
        prompt = _stage_prompt(stage, payload, validation_error)
        return self.complete_json(messages, prompt=prompt, call_kind=stage)

    def complete_json(
        self,
        messages: list[dict[str, Any]],
        *,
        prompt: str,
        call_kind: str,
    ) -> Mapping[str, Any]:
        """Submit one JSON-mode chat completion using the configured endpoint."""

        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"BaiLian API key environment variable {self.api_key_env!r} is not configured"
            )
        body = {
            "model": self.model,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "max_completion_tokens": self.max_completion_tokens,
            "seed": self.seed,
            "stream": False,
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        started = time.monotonic()
        try:
            response = self.opener(request, timeout=self.timeout_seconds)
            raw_body = response.read()
        except urllib.error.HTTPError as error:
            # Keep only a short provider diagnostic.  Never include request data
            # or credentials; this is useful for distinguishing payload/schema
            # validation errors from endpoint/authentication failures.
            try:
                detail = error.read(1024).decode("utf-8", errors="replace")
            except Exception:
                detail = ""
            detail = " ".join(detail.split())
            if "sk-" in detail:
                detail = detail.replace("sk-", "sk-[redacted]-")
            suffix = f": {detail[:600]}" if detail else ""
            audit = {
                "backend": "bailian_openai_compatible",
                "model": self.model,
                "stage": call_kind,
                "latency_seconds": round(time.monotonic() - started, 6),
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "http_status": error.code,
                "provider_error": detail[:600],
            }
            self.last_call = audit
            self.call_history.append(audit)
            raise RuntimeError(f"BaiLian request failed with HTTP {error.code}{suffix}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"BaiLian request failed: {error.reason}") from error
        try:
            envelope = json.loads(raw_body.decode("utf-8"))
            choice = envelope["choices"][0]["message"]["content"]
            if isinstance(choice, list):
                choice = next(
                    (
                        item.get("text")
                        for item in choice
                        if isinstance(item, Mapping) and isinstance(item.get("text"), str)
                    ),
                    None,
                )
            result = (
                _extract_stage_json(choice, call_kind)
                if call_kind in _STAGE_EXPECTED_KEYS
                else _extract_json_object(choice)
            )
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            KeyError,
            IndexError,
            TypeError,
            StageProtocolError,
        ) as error:
            audit = {
                "backend": "bailian_openai_compatible",
                "model": self.model,
                "stage": call_kind,
                "latency_seconds": round(time.monotonic() - started, 6),
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "parse_error": str(error),
                "raw_response": _audit_text(locals().get("choice", "")),
            }
            self.last_call = audit
            self.call_history.append(audit)
            raise StageProtocolError("BaiLian response does not contain a valid chat JSON result") from error
        self.last_call = {
            "backend": "bailian_openai_compatible",
            "model": self.model,
            "stage": call_kind,
            "latency_seconds": round(time.monotonic() - started, 6),
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "request_id": envelope.get("id"),
            "usage": envelope.get("usage"),
            "raw_response": _audit_text(choice),
            "parsed_output": result,
        }
        self.call_history.append(dict(self.last_call))
        return result


def _local_messages(
    stage: StageName,
    state: ChangeState,
    payload: Mapping[str, Any],
    previous_state: ChangeState | None,
    validation_error: str | None = None,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for label, image in _stage_images(stage, state, previous_state, payload):
        content.extend(
            [
                {"type": "text", "text": label},
                {"type": "image", "image": image},
            ]
        )
    content.append(
        {"type": "text", "text": _stage_prompt(stage, payload, validation_error)}
    )
    return [{"role": "user", "content": content}]


def _hosted_messages(
    stage: StageName,
    state: ChangeState,
    payload: Mapping[str, Any],
    previous_state: ChangeState | None,
    validation_error: str | None = None,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for label, image in _stage_images(stage, state, previous_state, payload):
        content.extend(
            [
                {"type": "text", "text": label},
                {"type": "image_url", "image_url": {"url": _image_data_url(image)}},
            ]
        )
    content.append(
        {"type": "text", "text": _stage_prompt(stage, payload, validation_error)}
    )
    return [
        {
            "role": "system",
            "content": "You are a strict change-detection verifier. Return JSON only.",
        },
        {"role": "user", "content": content},
    ]


def _stage_images(
    stage: StageName,
    state: ChangeState,
    previous_state: ChangeState | None,
    payload: Mapping[str, Any],
) -> list[tuple[str, Image.Image]]:
    if stage in {"evidence", "candidate_evidence"}:
        images = [
            ("T1 earlier RGB image", _as_image(state.t1_image)),
            ("T2 later RGB image", _as_image(state.t2_image)),
            ("Current predicted T1 object mask", _mask_image(state.t1_mask)),
            ("Current predicted T2 object mask", _mask_image(state.t2_mask)),
            ("Current final change mask", _mask_image(state.change_mask)),
        ]
        region = payload.get("region")
        if isinstance(region, Mapping):
            box = region.get("box_normalized_1000")
            if isinstance(box, (list, tuple)) and len(box) == 4:
                crop_box = _normalized_crop_box(box, state.image_size)
                images.extend(
                    [
                        ("Exact T1 proposal crop", _as_image(state.t1_image).crop(crop_box)),
                        ("Exact T2 proposal crop", _as_image(state.t2_image).crop(crop_box)),
                        ("Exact proposal change-mask crop", _mask_image(state.change_mask).crop(crop_box)),
                    ]
                )
        return images
    if stage == "decision" and previous_state is not None:
        return [
            ("Previous accepted final change mask", _mask_image(previous_state.change_mask)),
            ("Candidate final change mask", _mask_image(state.change_mask)),
        ]
    return []


def _stage_prompt(
    stage: StageName,
    payload: Mapping[str, Any],
    validation_error: str | None = None,
) -> str:
    context = json.dumps(
        {"environment_facts": payload},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    region = payload.get("region")
    region_id = (
        str(region.get("region_id"))
        if isinstance(region, Mapping) and region.get("region_id") is not None
        else "r0"
    )
    if stage in {"evidence", "candidate_evidence"}:
        schema = (
            f'{{"region_id":"{region_id}","visual_judgment":{{'
            '"t1_state":"background",'
            '"t2_state":"building",'
            '"visual_confidence":0.0,"evidence_quality":"clear"}}}'
        )
        task = (
            "Read only the supplied local visual evidence and classify T1/T2 content. "
            "Each state must be one of building, background, mixed, uncertain; evidence_quality "
            "must be clear, ambiguous, or insufficient."
        )
    elif stage in {"diagnosis", "candidate_diagnosis"}:
        schema = (
            f'{{"region_id":"{region_id}","diagnosis":{{'
            '"error_type":"none","target_view":null,"confidence":0.0}}}'
        )
        task = (
            "Classify the current mask error. If T1/T2 visual states differ and the change mask "
            "is already white, the region is normally correct and error_type is none. If that "
            "difference is missing from a black change mask, use false_negative and target the "
            "view that should gain an object-mask correction. error_type must be one of none, "
            "false_positive_change, false_negative, mixed_error, uncertain_region. target_view "
            "must be JSON null when error_type is none; otherwise use the JSON string t1 or t2."
        )
    elif stage == "plan":
        diagnosis = payload.get("diagnosis")
        target_view = (
            str(diagnosis.get("target_view"))
            if isinstance(diagnosis, Mapping)
            and diagnosis.get("target_view") in {"t1", "t2"}
            else "t2"
        )
        editable = (
            region.get("editable_seed_white", {})
            if isinstance(region, Mapping)
            else {}
        )
        seed_white = (
            bool(editable.get(target_view, False))
            if isinstance(editable, Mapping)
            else False
        )
        point_action = "negative_point" if seed_white else "positive_point"
        seed = (
            region.get("component_seed_normalized_1000", [0, 0])
            if isinstance(region, Mapping)
            else [0, 0]
        )
        if not isinstance(seed, (list, tuple)) or len(seed) != 2:
            seed = [0, 0]
        schema = (
            f'{{"region_id":"{region_id}","plan":{{'
            f'"action":"{point_action}","target_view":"{target_view}",'
            f'"coordinate_normalized_1000":[{int(seed[0])},{int(seed[1])}],'
            '"box_normalized_1000":null}}}'
        )
        task = (
            "Choose only an allowed executable action. negative_point requires a white seed in "
            "the selected object mask; positive_point requires a black seed. Use supplied region "
            "geometry. For a point action, copy the exact supplied component seed shown in the "
            "output contract. Use exactly one of coordinate_normalized_1000 and "
            "box_normalized_1000."
        )
    else:
        candidate_mode = payload.get("mode") == "candidate"
        comparison_example = "uncertain" if candidate_mode else "initial"
        schema = (
            f'{{"decision":{{"comparison":"{comparison_example}","quality_score":0.0,'
            '"progress_score":0.0,"accept":false,"stop":false,'
            '"feedback":"short explanation"}}'
        )
        task = (
            "Judge the complete initial state or compare the candidate with the previous accepted "
            "state. comparison must be initial, better, worse, unchanged, or uncertain. accept "
            "may be true for an initial state only when no error remains, and for a candidate "
            "only when comparison is better. stop requires accept=true and no remaining error."
        )
    repair = ""
    if validation_error:
        repair = (
            "\nREPAIR: The previous response was rejected for this exact reason: "
            f"{validation_error[:500]} Do not repeat the rejected structure."
        )
    return (
        "OUTPUT CONTRACT: Return exactly one JSON object, no markdown and no explanation. "
        "The top-level keys must exactly match this template:\n"
        f"{schema}\n"
        f"TASK: {task}"
        f"{repair}\n"
        "Environment region IDs and normalized [0,1000] geometry are authoritative; never alter "
        "them. T1 is earlier and T2 is later. Do not copy the Environment envelope into the "
        "answer.\n<ENVIRONMENT_FACTS>\n"
        f"{context}\n"
        "</ENVIRONMENT_FACTS>"
    )


_STAGE_EXPECTED_KEYS: dict[str, frozenset[str]] = {
    "evidence": frozenset({"region_id", "visual_judgment"}),
    "candidate_evidence": frozenset({"region_id", "visual_judgment"}),
    "diagnosis": frozenset({"region_id", "diagnosis"}),
    "candidate_diagnosis": frozenset({"region_id", "diagnosis"}),
    "plan": frozenset({"region_id", "plan"}),
    "decision": frozenset({"decision"}),
}


def _extract_stage_json(raw: Any, stage: str) -> Mapping[str, Any]:
    """Select the JSON envelope for ``stage`` instead of the first object."""

    expected = _STAGE_EXPECTED_KEYS.get(stage)
    if expected is None:
        raise StageProtocolError(f"unsupported staged JSON call: {stage!r}")
    candidates = [raw] if isinstance(raw, Mapping) else list(_json_objects(raw))
    for candidate in candidates:
        if isinstance(candidate, Mapping) and frozenset(candidate) == expected:
            return candidate
    keys = [sorted(str(key) for key in item) for item in candidates if isinstance(item, Mapping)]
    raise StageProtocolError(
        f"{stage} response contains no JSON object with exact keys {sorted(expected)}; "
        f"candidate_keys={keys}"
    )


def _extract_json_object(raw: Any) -> Mapping[str, Any]:
    if isinstance(raw, Mapping):
        return raw
    if not isinstance(raw, str):
        raise StageProtocolError("model response must be JSON text or an object")
    start = raw.find("{")
    if start < 0:
        raise StageProtocolError("model response contains no JSON object")
    try:
        value, _ = json.JSONDecoder().raw_decode(raw[start:])
    except json.JSONDecodeError as error:
        raise StageProtocolError("model response contains incomplete JSON") from error
    if not isinstance(value, Mapping):
        raise StageProtocolError("model response JSON must be an object")
    return value


def _json_objects(raw: Any):
    if not isinstance(raw, str):
        raise StageProtocolError("model response must be JSON text or an object")
    decoder = json.JSONDecoder()
    cursor = 0
    found = False
    while True:
        start = raw.find("{", cursor)
        if start < 0:
            break
        try:
            value, consumed = decoder.raw_decode(raw[start:])
        except json.JSONDecodeError:
            cursor = start + 1
            continue
        cursor = start + max(consumed, 1)
        if isinstance(value, Mapping):
            found = True
            yield value
    if not found:
        raise StageProtocolError("model response contains no complete JSON object")


def _audit_text(raw: Any, limit: int = 12000) -> str:
    value = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
    return value if len(value) <= limit else value[:limit] + "...[truncated]"


def _image_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return "data:image/png;base64," + encoded


def _as_image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value
    array = np.asarray(value)
    if array.dtype != np.uint8:
        if array.max(initial=0) <= 1:
            array = array * 255
        array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array)


def _mask_image(mask: np.ndarray) -> Image.Image:
    return Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255, mode="L")


def _normalized_crop_box(
    box: list[int] | tuple[int, ...], image_size: tuple[int, int]
) -> tuple[int, int, int, int]:
    width, height = image_size
    x1, y1, x2, y2 = (int(value) for value in box)
    left = max(0, min(width - 1, round(x1 * (width - 1) / 1000)))
    top = max(0, min(height - 1, round(y1 * (height - 1) / 1000)))
    right = max(left + 1, min(width, round(x2 * (width - 1) / 1000) + 1))
    bottom = max(top + 1, min(height, round(y2 * (height - 1) / 1000) + 1))
    return left, top, right, bottom
