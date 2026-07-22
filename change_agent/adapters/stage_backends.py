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
from PIL import Image, ImageDraw

from ..coordinates import normalized_point_to_pixel
from .omniovcd_adapter import connected_components
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
    if stage == "direct":
        images = [
            ("T1 earlier RGB image", _as_image(state.t1_image)),
            ("T2 later RGB image", _as_image(state.t2_image)),
            ("Current predicted T1 object mask", _mask_image(state.t1_mask)),
            ("Current predicted T2 object mask", _mask_image(state.t2_mask)),
            ("Current final change mask", _mask_image(state.change_mask)),
        ]
        if previous_state is not None:
            rejected = payload.get("mode") == "replan"
            state_label = "Rejected candidate" if rejected else "Previous accepted"
            images.extend(
                [
                    (f"{state_label} T1 object mask", _mask_image(previous_state.t1_mask)),
                    (f"{state_label} T2 object mask", _mask_image(previous_state.t2_mask)),
                    (f"{state_label} final change mask", _mask_image(previous_state.change_mask)),
                ]
            )
        return images
    if stage == "select":
        catalog = payload.get("proposal_catalog", [])
        if not isinstance(catalog, list):
            raise StageProtocolError("proposal_catalog must be a list")
        return [
            (
                "Numbered global proposal overview: T1 | T2 | current change mask",
                _proposal_overview(state, previous_state, catalog),
            )
        ]
    if stage in {"evidence", "candidate_evidence", "diagnosis", "candidate_diagnosis"}:
        visual_context = str(payload.get("visual_context", "hybrid"))
        if visual_context not in {"proposal", "hybrid"}:
            raise StageProtocolError(
                "visual_context must be proposal or hybrid for staged regional calls"
            )
        images: list[tuple[str, Image.Image]] = []
        catalog = payload.get("proposal_catalog", [])
        if isinstance(catalog, list) and catalog:
            active = payload.get("region", {})
            active_id = (
                str(active.get("region_id"))
                if isinstance(active, Mapping) and active.get("region_id")
                else None
            )
            images.append(
                (
                    "Global numbered overview; yellow marks the active local region",
                    _proposal_overview(
                        state,
                        previous_state,
                        catalog,
                        active_region_id=active_id,
                    ),
                )
            )
        region = payload.get("region")
        if isinstance(region, Mapping):
            box = region.get("box_normalized_1000")
            if isinstance(box, (list, tuple)) and len(box) == 4:
                crop_box = _normalized_crop_box(box, state.image_size)
                images.extend(
                    [
                        ("Exact T1 proposal crop", _as_image(state.t1_image).crop(crop_box)),
                        ("Exact T2 proposal crop", _as_image(state.t2_image).crop(crop_box)),
                        ("Exact proposal T1 object-mask crop", _mask_image(state.t1_mask).crop(crop_box)),
                        ("Exact proposal T2 object-mask crop", _mask_image(state.t2_mask).crop(crop_box)),
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


_MARK_COLORS = (
    (255, 70, 70),
    (40, 220, 90),
    (60, 150, 255),
    (255, 80, 220),
    (30, 220, 220),
    (255, 150, 30),
)


def _proposal_overview(
    state: ChangeState,
    previous_state: ChangeState | None,
    catalog: list[Any],
    *,
    active_region_id: str | None = None,
) -> Image.Image:
    """Render deterministic Set-of-Mark contours from Environment proposals."""

    panels = [
        _as_image(state.t1_image).convert("RGB"),
        _as_image(state.t2_image).convert("RGB"),
        _mask_image(state.change_mask).convert("RGB"),
    ]
    width, height = state.image_size
    for index, proposal in enumerate(catalog):
        if not isinstance(proposal, Mapping):
            continue
        region_id = str(proposal.get("region_id", ""))
        if not region_id:
            continue
        color = (
            (255, 255, 0)
            if region_id == active_region_id
            else _MARK_COLORS[index % len(_MARK_COLORS)]
        )
        component = _proposal_component(state, previous_state, proposal)
        seed = proposal.get("component_seed_normalized_1000", [0, 0])
        try:
            seed_x, seed_y = normalized_point_to_pixel(seed, state.image_size)
        except (TypeError, ValueError):
            continue
        boundary = _mask_boundary(component) if component is not None else None
        for panel in panels:
            array = np.asarray(panel).copy()
            if boundary is not None and boundary.any():
                array[boundary] = color
                panel.paste(Image.fromarray(array))
            else:
                box = proposal.get("box_normalized_1000")
                if isinstance(box, (list, tuple)) and len(box) == 4:
                    left, top, right, bottom = _normalized_crop_box(
                        tuple(int(value) for value in box), state.image_size
                    )
                    ImageDraw.Draw(panel).rectangle(
                        (left, top, max(left, right - 1), max(top, bottom - 1)),
                        outline=color,
                        width=2,
                    )
            draw = ImageDraw.Draw(panel)
            radius = 3
            draw.ellipse(
                (
                    seed_x - radius,
                    seed_y - radius,
                    seed_x + radius,
                    seed_y + radius,
                ),
                fill=color,
                outline=(0, 0, 0),
            )
            label_x = min(max(0, seed_x + 5), max(0, width - 24))
            label_y = min(max(0, seed_y - 8), max(0, height - 12))
            label_box = draw.textbbox((label_x, label_y), region_id)
            draw.rectangle(label_box, fill=(0, 0, 0))
            draw.text((label_x, label_y), region_id, fill=color)
    canvas = Image.new("RGB", (width * len(panels), height), (0, 0, 0))
    for index, panel in enumerate(panels):
        canvas.paste(panel, (index * width, 0))
    return canvas


def _proposal_component(
    state: ChangeState,
    previous_state: ChangeState | None,
    proposal: Mapping[str, Any],
) -> np.ndarray | None:
    kind = str(proposal.get("audit_kind", "mixed"))
    temporal_xor = np.logical_xor(state.t1_mask, state.t2_mask)
    if kind == "present":
        source = state.change_mask
    elif kind == "missing":
        source = np.logical_and(temporal_xor, ~state.change_mask)
    elif kind == "delta_added" and previous_state is not None:
        source = np.logical_and(state.change_mask, ~previous_state.change_mask)
    elif kind == "delta_removed" and previous_state is not None:
        source = np.logical_and(previous_state.change_mask, ~state.change_mask)
    else:
        source = np.logical_or(state.change_mask, temporal_xor)
    seed = proposal.get("component_seed_normalized_1000", [0, 0])
    try:
        seed_x, seed_y = normalized_point_to_pixel(seed, state.image_size)
    except (TypeError, ValueError):
        return None
    for component in connected_components(source, min_area=1):
        if component[seed_y, seed_x]:
            return component
    return None


def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(np.asarray(mask, dtype=bool), 1, constant_values=False)
    eroded = np.ones_like(mask, dtype=bool)
    for offset_y in range(3):
        for offset_x in range(3):
            eroded &= padded[
                offset_y : offset_y + mask.shape[0],
                offset_x : offset_x + mask.shape[1],
            ]
    return np.logical_and(mask, ~eroded)


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
    if stage == "select":
        max_selected = int(payload.get("max_selected_regions", 1))
        schema = (
            '{"selection":{"region_ids":["r0"],'
            '"reason":"short observable reason"}}'
        )
        task = (
            "Inspect the numbered global overview and choose the regions most likely to contain "
            "a material target-class change-mask error. Return one or more supplied region IDs, "
            f"at most {max_selected}; never output coordinates. Selection only prioritizes local "
            "inspection and does not assert that unselected regions are correct."
        )
    elif stage in {"evidence", "candidate_evidence"}:
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
            '{"region_id":"' + region_id + '","diagnosis":{'
            '"error_type":<ERROR_TYPE>,"target_view":<TARGET_VIEW_OR_NULL>,'
            '"confidence":<CONFIDENCE_0_TO_1>}}'
        )
        task = (
            "Classify the current mask error from supplied RGB evidence, predicted masks, "
            "proposal crop, and Environment facts. Do not infer correctness from the change-mask "
            "color or from a temporal difference alone. A proposal may contain both correct and "
            "incorrect pixels. Use false_positive_change when predicted white change lacks real "
            "T1/T2 change, false_negative when a black region contains real change, mixed_error "
            "Use mixed_error when one proposal contains both supported and unsupported predicted "
            "pixels or a partial omission, and uncertain_region when evidence is insufficient. "
            "Use none only "
            "when the whole audited region is supported by the RGB evidence and mask coverage. "
            "Inspect boundaries and internal gaps, not only the dominant object. error_type must "
            "be one of none, false_positive_change, false_negative, mixed_error, "
            "uncertain_region. target_view must be JSON null when error_type is none; otherwise "
            "use the JSON string t1 or t2 for the object mask that needs correction. confidence "
            "means confidence in this diagnosis; for none, report confidence that the whole "
            "region is correct. Replace every schema placeholder with a valid JSON value."
        )
    elif stage == "direct":
        mode = str(payload.get("mode", "initial"))
        candidate_mode = mode == "candidate"
        replan_mode = mode == "replan"
        candidate_effect = (
            '{"intended_error_improved":<BOOLEAN>,'
            '"introduced_false_positive":<BOOLEAN>,'
            '"introduced_false_negative":<BOOLEAN>,'
            '"boundary_or_artifact_worsened":<BOOLEAN>,'
            '"evidence":"short paired evidence"}'
            if candidate_mode
            else "null"
        )
        schema = (
            '{"verdict":{"rubric":{'
            '"evidence_sufficient":{"pass":<BOOLEAN>,"evidence":"short evidence"},'
            '"target_class_only":{"pass":<BOOLEAN>,"evidence":"short evidence"},'
            '"change_semantic_precision":{"pass":<BOOLEAN>,"evidence":"short evidence"},'
            '"change_semantic_recall":{"pass":<BOOLEAN>,"evidence":"short evidence"},'
            '"changed_object_extent":{"pass":<BOOLEAN>,"evidence":"short evidence"},'
            '"change_boundary_alignment":{"pass":<BOOLEAN>,"evidence":"short evidence"},'
            '"change_artifact_control":{"pass":<BOOLEAN>,"evidence":"short evidence"}},'
            '"candidate_effect":' + candidate_effect + ','
            '"error_type":<ERROR_TYPE>,"target_view":<TARGET_VIEW_OR_NULL>,'
            '"suggested_action":<ACTION>,"coordinate_normalized_1000":<POINT_OR_NULL>,'
            '"box_normalized_1000":<BOX_OR_NULL>,"feedback":"short explanation"}}'
        )
        task = (
            "Apply the binary rubric to the complete T1/T2 pair and final change mask without "
            "Proposal geometry. The only semantic target is target_class in Environment facts. "
            "For target_class=building, roads, parking areas, vehicles, vegetation, bare ground, "
            "shadows, illumination, and registration differences are context, never target "
            "changes. Predicted T1/T2 object masks are fallible aids, not ground truth; do not "
            "demand that they segment non-target content or unchanged target objects when the "
            "final change mask is correct. White change pixels are correct only for a real "
            "appearance, disappearance, construction, or demolition of target objects. "
            "Set evidence_sufficient for visual judgeability. target_class_only records whether "
            "your reasoning stayed scoped to target_class; set it true when you correctly call "
            "roads or vehicles non-target false positives, even if the mask contains them. Set "
            "it false only when your reasoning treats non-target content as target change. Set "
            "change_semantic_precision when no "
            "material white region is unsupported, change_semantic_recall when no obvious target "
            "change is missing, changed_object_extent when changed targets have materially "
            "complete coverage, change_boundary_alignment when boundaries follow changed target "
            "objects, and change_artifact_control when no material fragments, holes, or noise "
            "remain. Give one short observable evidence string per item; do not provide hidden "
            "chain-of-thought. Runtime computes quality, progress, comparison, and acceptance, "
            "so never output those fields. error_type must be exactly one of none, "
            "false_positive_change, false_negative, mixed_error, or uncertain_region. Every "
            "rubric item passing requires none; none requires every quality item and target scope "
            "to pass. A failed evidence_sufficient gate requires uncertain_region; a failed "
            "target_class_only item is an auditable scope error and may still carry an actionable "
            "false-positive or mixed diagnosis. For none, "
            "use target_view null, suggested_action finish, and null geometry. "
            "For an error, choose t1 or t2 and one executable positive_point, negative_point, "
            "or box; suggested_action must be exactly one of positive_point, negative_point, "
            "box, or finish (never use revise/correct/edit). A point needs normalized [0,1000] coordinate; a box needs normalized "
            "[0,1000] XYXY box. For a negative_point, the selected coordinate must be white "
            "in the selected current object mask. A positive_point may be on either mask state "
            "because its predicted component can still expand the current mask. "
            "For a candidate, compare actual candidate versus previous "
            "accepted masks: intended_error_improved means the attempted target-class error was "
            "materially reduced; the three harm flags report newly introduced semantic or "
            "shape damage."
        )
        if replan_mode:
            task += (
                " This is a rollback replan: current images and masks are the accepted state; "
                "the additional rejected-candidate masks and Environment facts explain why the "
                "previous action failed. Do not repeat the rejected action or geometry. Choose a "
                "materially different executable correction, or finish only if no error remains. "
                "Use the bounded rejection_history to avoid all recently failed actions. "
                "This is action replanning, not candidate evaluation; candidate_effect must be "
                "JSON null. Runtime comparison is uncertain until the replacement action is "
                "executed; comparison must be uncertain in this replan response. "
                ""
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
        if stage == "direct" and isinstance(payload.get("repair_context"), Mapping):
            repair += (
                " Preserve the previous response's rubric pass booleans, error_type, "
                "and candidate_effect flags. Change only the invalid target/action/geometry "
                "fields needed to satisfy the tool contract; do not turn an actionable error "
                "into none/finish merely because the first geometry was invalid."
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
    "select": frozenset({"selection"}),
    "evidence": frozenset({"region_id", "visual_judgment"}),
    "candidate_evidence": frozenset({"region_id", "visual_judgment"}),
    "diagnosis": frozenset({"region_id", "diagnosis"}),
    "candidate_diagnosis": frozenset({"region_id", "diagnosis"}),
    "decision": frozenset({"decision"}),
    "direct": frozenset({"verdict"}),
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
    """Convert a normalized inclusive box to a provider-safe PIL crop box.

    BaiLian's vision endpoint rejects images whose height or width is below
    11 pixels. Region proposals can legitimately be smaller than that after
    connected-component extraction, so expand only the crop sent to the
    provider while keeping it inside the original image. The proposal's
    normalized coordinates and mask facts remain unchanged.
    """

    width, height = image_size
    x1, y1, x2, y2 = (int(value) for value in box)
    left = max(0, min(width - 1, round(x1 * (width - 1) / 1000)))
    top = max(0, min(height - 1, round(y1 * (height - 1) / 1000)))
    right = max(left + 1, min(width, round(x2 * (width - 1) / 1000) + 1))
    bottom = max(top + 1, min(height, round(y2 * (height - 1) / 1000) + 1))
    return _expand_crop_min_side(
        (left, top, right, bottom),
        image_size,
        min_side=11,
    )


def _expand_crop_min_side(
    crop_box: tuple[int, int, int, int],
    image_size: tuple[int, int],
    *,
    min_side: int,
) -> tuple[int, int, int, int]:
    """Expand a PIL crop box to a minimum side length without leaving the image."""

    if min_side < 1:
        raise ValueError("min_side must be positive")
    width, height = image_size
    left, top, right, bottom = crop_box
    target_width = min(width, min_side)
    target_height = min(height, min_side)

    if right - left < target_width:
        right = min(width, left + target_width)
        left = max(0, right - target_width)
    if bottom - top < target_height:
        bottom = min(height, top + target_height)
        top = max(0, bottom - target_height)
    return left, top, right, bottom
