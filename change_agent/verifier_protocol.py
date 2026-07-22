"""Typed, stage-oriented protocol for GT-free change verification.

The Environment owns geometry and editability.  A model is only asked to select
existing region IDs and make local semantic decisions. Keeping these records
typed prevents a model from silently changing a region id, coordinate frame,
or mask fact between diagnosis and programmatic action resolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Protocol, Sequence

from .state import ChangeState

StageName = Literal[
    "select",
    "evidence",
    "diagnosis",
    "candidate_evidence",
    "candidate_diagnosis",
    "decision",
    "direct",
]

ERROR_TYPES = {
    "none",
    "false_positive_change",
    "false_negative",
    "mixed_error",
    "uncertain_region",
}
TARGET_VIEWS = {"t1", "t2"}
ACTIONS = {"positive_point", "negative_point", "box", "finish"}
COMPARISONS = {"initial", "better", "worse", "unchanged", "uncertain"}
RGB_STATES = {"building", "background", "mixed", "uncertain"}
AUDIT_KINDS = {"present", "missing", "delta_added", "delta_removed", "mixed"}


class StageProtocolError(ValueError):
    """Raised when a model response cannot be safely used by the runtime."""


@dataclass(frozen=True)
class EvidenceRecord:
    """Authoritative Environment facts for one proposed region."""

    region_id: str
    audit_kind: str
    change_mask_state: Literal["white", "black", "mixed"]
    temporal_difference_state: Literal["present", "absent", "mixed"]
    box_normalized_1000: tuple[int, int, int, int]
    component_seed_normalized_1000: tuple[int, int]
    t1_mask_pixels: int = 0
    t2_mask_pixels: int = 0
    component_area: int = 0
    editable_seed_white: Mapping[str, bool] = field(default_factory=dict)
    allowed_actions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.region_id:
            raise StageProtocolError("region_id must be non-empty")
        if self.audit_kind not in AUDIT_KINDS:
            raise StageProtocolError(f"unsupported audit_kind: {self.audit_kind!r}")
        if self.change_mask_state not in {"white", "black", "mixed"}:
            raise StageProtocolError("unsupported change_mask_state")
        if self.temporal_difference_state not in {"present", "absent", "mixed"}:
            raise StageProtocolError("unsupported temporal_difference_state")
        _validate_box(self.box_normalized_1000)
        _validate_point(self.component_seed_normalized_1000)
        if any(value < 0 for value in (self.t1_mask_pixels, self.t2_mask_pixels, self.component_area)):
            raise StageProtocolError("mask pixel counts must be non-negative")
        for view, value in self.editable_seed_white.items():
            if view not in TARGET_VIEWS or not isinstance(value, bool):
                raise StageProtocolError("editable_seed_white must map t1/t2 to bool")
        for action in self.allowed_actions:
            if action not in ACTIONS - {"finish"}:
                raise StageProtocolError(f"unsupported allowed action: {action!r}")

    @classmethod
    def from_proposal(cls, proposal: Mapping[str, Any]) -> "EvidenceRecord":
        """Convert an Environment proposal without allowing model-authored facts."""

        region_id = str(proposal["region_id"])
        seed = proposal.get("component_seed_normalized") or proposal.get(
            "component_seed_normalized_1000"
        )
        box = proposal.get("box_normalized") or proposal.get("box_normalized_1000")
        if seed is None or box is None:
            raise StageProtocolError(f"proposal {region_id} lacks canonical geometry")
        if "audit_kind" in proposal:
            audit_kind = str(proposal["audit_kind"])
        elif proposal.get("effect_kind") == "added":
            audit_kind = "delta_added"
        elif proposal.get("effect_kind") == "removed":
            audit_kind = "delta_removed"
        else:
            audit_kind = "mixed"
        change_state = (
            "white"
            if audit_kind in {"present", "delta_added"}
            else "black"
            if audit_kind in {"missing", "delta_removed"}
            else "mixed"
        )
        facts = proposal.get("mask_facts", {})
        editable = {
            view: bool(proposal.get(f"component_seed_{view}_mask_white", False))
            for view in TARGET_VIEWS
        }
        allowed = tuple(
            action
            for action in ("positive_point", "negative_point", "box")
            if action == "box"
            or any(
                editable.get(view, False) if action == "negative_point" else not editable.get(view, False)
                for view in TARGET_VIEWS
            )
        )
        return cls(
            region_id=region_id,
            audit_kind=audit_kind,
            change_mask_state=change_state,
            temporal_difference_state=(
                "present"
                if int(proposal.get("temporal_difference_pixels", 0)) > 0
                else "absent"
            ),
            box_normalized_1000=tuple(int(value) for value in box),
            component_seed_normalized_1000=tuple(int(value) for value in seed),
            t1_mask_pixels=int(proposal.get("t1_mask_pixels", facts.get("t1_mask_pixels", 0))),
            t2_mask_pixels=int(proposal.get("t2_mask_pixels", facts.get("t2_mask_pixels", 0))),
            component_area=int(proposal.get("component_area", 0)),
            editable_seed_white=editable,
            allowed_actions=allowed,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "audit_kind": self.audit_kind,
            "change_mask_state": self.change_mask_state,
            "temporal_difference_state": self.temporal_difference_state,
            "box_normalized_1000": list(self.box_normalized_1000),
            "component_seed_normalized_1000": list(self.component_seed_normalized_1000),
            "t1_mask_pixels": self.t1_mask_pixels,
            "t2_mask_pixels": self.t2_mask_pixels,
            "component_area": self.component_area,
            "editable_seed_white": dict(self.editable_seed_white),
            "allowed_actions": list(self.allowed_actions),
        }


@dataclass(frozen=True)
class EvidenceJudgment:
    region_id: str
    t1_state: str
    t2_state: str
    confidence: float
    evidence_quality: Literal["clear", "ambiguous", "insufficient"]


@dataclass(frozen=True)
class Diagnosis:
    region_id: str
    error_type: str
    target_view: str | None
    confidence: float

    def __post_init__(self) -> None:
        if self.error_type not in ERROR_TYPES:
            raise StageProtocolError(f"unsupported error_type: {self.error_type!r}")
        if self.target_view is not None and self.target_view not in TARGET_VIEWS:
            raise StageProtocolError("target_view must be t1, t2, or null")
        if self.error_type == "none" and self.target_view is not None:
            raise StageProtocolError("none diagnosis must not target a view")
        _validate_confidence(self.confidence)


@dataclass(frozen=True)
class ActionPlan:
    region_id: str | None
    action: str
    target_view: str | None
    coordinate_normalized_1000: tuple[int, int] | None = None
    box_normalized_1000: tuple[int, int, int, int] | None = None

    def __post_init__(self) -> None:
        if self.action not in ACTIONS:
            raise StageProtocolError(f"unsupported action: {self.action!r}")
        if self.action == "finish":
            if self.region_id is not None or self.target_view is not None:
                raise StageProtocolError("finish plan must not select a region or view")
            if self.coordinate_normalized_1000 is not None or self.box_normalized_1000 is not None:
                raise StageProtocolError("finish plan must not contain geometry")
            return
        if self.region_id is None or self.target_view not in TARGET_VIEWS:
            raise StageProtocolError("tool plan requires region_id and target_view")
        if self.action in {"positive_point", "negative_point"}:
            if self.coordinate_normalized_1000 is None or self.box_normalized_1000 is not None:
                raise StageProtocolError("point plan requires only coordinate_normalized_1000")
            _validate_point(self.coordinate_normalized_1000)
        if self.action == "box":
            if self.box_normalized_1000 is None or self.coordinate_normalized_1000 is not None:
                raise StageProtocolError("box plan requires only box_normalized_1000")
            _validate_box(self.box_normalized_1000)


@dataclass(frozen=True)
class Decision:
    comparison: str
    quality_score: float
    progress_score: float
    accept: bool
    stop: bool
    feedback: str = ""

    def __post_init__(self) -> None:
        if self.comparison not in COMPARISONS:
            raise StageProtocolError(f"unsupported comparison: {self.comparison!r}")
        if not 0 <= self.quality_score <= 1:
            raise StageProtocolError("quality_score must be in [0,1]")
        if not -1 <= self.progress_score <= 1:
            raise StageProtocolError("progress_score must be in [-1,1]")
        if self.stop and not self.accept:
            raise StageProtocolError("stop requires accept=true")


@dataclass(frozen=True)
class StageTrace:
    """Serializable intermediate state retained for audit and replay."""

    mode: Literal["initial", "candidate"]
    evidence: tuple[EvidenceRecord, ...]
    selected_region_ids: tuple[str, ...] = ()
    judgments: tuple[EvidenceJudgment, ...] = ()
    diagnoses: tuple[Diagnosis, ...] = ()
    plan: ActionPlan | None = None
    decision: Decision | None = None
    replan_evidence: tuple[EvidenceRecord, ...] = ()
    replan_selected_region_ids: tuple[str, ...] = ()
    replan_judgments: tuple[EvidenceJudgment, ...] = ()
    replan_diagnoses: tuple[Diagnosis, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "evidence": [item.to_dict() for item in self.evidence],
            "selected_region_ids": list(self.selected_region_ids),
            "judgments": [item.__dict__ for item in self.judgments],
            "diagnoses": [item.__dict__ for item in self.diagnoses],
            "plan": self.plan.__dict__ if self.plan else None,
            "decision": self.decision.__dict__ if self.decision else None,
            "replan_evidence": [item.to_dict() for item in self.replan_evidence],
            "replan_selected_region_ids": list(self.replan_selected_region_ids),
            "replan_judgments": [item.__dict__ for item in self.replan_judgments],
            "replan_diagnoses": [item.__dict__ for item in self.replan_diagnoses],
        }


class StageBackend(Protocol):
    """Model-independent interface used by local and hosted MLLM backends."""

    def generate_stage(
        self,
        stage: StageName,
        state: ChangeState,
        payload: Mapping[str, Any],
        previous_state: ChangeState | None = None,
    ) -> Mapping[str, Any]: ...


def _validate_point(point: Sequence[int]) -> None:
    if len(point) != 2 or any(not isinstance(value, int) for value in point):
        raise StageProtocolError("normalized point must contain two integers")
    if any(value < 0 or value > 1000 for value in point):
        raise StageProtocolError("normalized point must be within [0,1000]")


def _validate_box(box: Sequence[int]) -> None:
    if len(box) != 4 or any(not isinstance(value, int) for value in box):
        raise StageProtocolError("normalized box must contain four integers")
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
        raise StageProtocolError("normalized box must be ordered and within [0,1000]")


def _validate_confidence(value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= float(value) <= 1:
        raise StageProtocolError("confidence must be in [0,1]")
