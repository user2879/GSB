"""Serializable fault injection configuration.

Two layers:
  * specification layer: faults that enter through declared inputs and
    parameters (bounds, manifest truth, tone-map constants), validated with
    fixed error codes before any pixel is processed;
  * pipeline layer: replacement of a specific frame's intermediate stage
    array, applied only through the engine's InjectionContext.

Each item carries ``expect_detected`` (the oracle). A run is classified into
detected / missed / false_positive by the detector against this oracle.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional

from .config import (
    ACES_A, GAMMA_MAX, GAMMA_MIN, STAGES, TONEMAP_INPUT_MAX,
)
from .errors import ErrorCode
from .hashing import canonical_json

SPEC_INJECTIONS = (
    "peak_out_of_bounds",
    "illegal_transfer_param",
    "truth_sample_tamper",
    "tonemap_constant_tamper",
)
PIPELINE_INJECTIONS = ("stage_replacement",)


@dataclass
class InjectionItem:
    item_id: str
    kind: str
    expect_detected: bool = True
    frame: Optional[int] = None
    stage: Optional[str] = None
    payload: dict = field(default_factory=dict)
    note: str = ""

    def validate_shape(self):
        if self.kind in SPEC_INJECTIONS:
            if self.kind == "stage_replacement":
                raise ValueError("kind mismatch")
        if self.kind == "stage_replacement":
            if self.frame is None or self.stage not in STAGES:
                raise ValueError(
                    "stage_replacement requires a frame and a valid stage")
        if self.kind == "truth_sample_tamper" and self.frame is None:
            raise ValueError("truth_sample_tamper requires a frame")
        return self


@dataclass
class InjectionConfig:
    config_id: str
    items: list = field(default_factory=list)

    def to_dict(self):
        return {
            "config_id": self.config_id,
            "items": [asdict(it) for it in self.items],
        }

    def to_json(self):
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, data):
        items = [InjectionItem(**it) for it in data["items"]]
        for it in items:
            it.validate_shape()
        return cls(config_id=data["config_id"], items=items)

    def get(self, item_id):
        for it in self.items:
            if it.item_id == item_id:
                return it
        raise KeyError(item_id)


# ---------------------------------------------------------------------------
# Specification-layer builders
# ---------------------------------------------------------------------------

def peak_out_of_bounds(item_id="inj_peak", *, value=1.0e9,
                       expect_detected=True):
    """Truth manifest peak luma above the declared HDR physical bound."""
    return InjectionItem(
        item_id=item_id,
        kind="peak_out_of_bounds",
        expect_detected=expect_detected,
        payload={"peak_luma": value, "legal_max": TONEMAP_INPUT_MAX},
        note="manifest peak luma exceeds physical HDR bound",
    )


def illegal_transfer_param(item_id="inj_gamma", *, gamma=GAMMA_MAX * 10.0,
                           expect_detected=True):
    return InjectionItem(
        item_id=item_id,
        kind="illegal_transfer_param",
        expect_detected=expect_detected,
        payload={"transfer_gamma": gamma,
                 "legal_range": [GAMMA_MIN, GAMMA_MAX]},
        note="transfer gamma outside legal domain",
    )


def truth_sample_tamper(item_id, frame, row, col, *, delta=0.5,
                        expect_detected=True):
    return InjectionItem(
        item_id=item_id,
        kind="truth_sample_tamper",
        expect_detected=expect_detected,
        frame=frame,
        payload={"row": row, "col": col, "delta": delta},
        note="reference sample linear RGB tampered",
    )


def tonemap_constant_tamper(item_id="inj_aces_a", *, constant="tonemap_a",
                            factor=1.001, expect_detected=True):
    return InjectionItem(
        item_id=item_id,
        kind="tonemap_constant_tamper",
        expect_detected=expect_detected,
        payload={"constant": constant, "factor": factor,
                 "baseline": ACES_A if constant == "tonemap_a" else None},
        note="frozen tone-map constant altered",
    )


def stage_replacement(item_id, frame, stage, replacement, *,
                      expect_detected=True, note=""):
    """Pipeline-layer injection: replace one frame's stage array."""
    return InjectionItem(
        item_id=item_id,
        kind="stage_replacement",
        expect_detected=expect_detected,
        frame=frame,
        stage=stage,
        payload={"replacement": replacement},
        note=note,
    )


def apply_manifest_tamper(manifest, item):
    """Return a deep-ish copy of the manifest with a spec injection applied."""
    import copy

    patched = copy.deepcopy(manifest)
    if item.kind == "peak_out_of_bounds":
        frame_idx = item.payload.get("frame", 0)
        patched["frames"][frame_idx]["peak_luma"] = float(
            item.payload["peak_luma"]
        )
    elif item.kind == "truth_sample_tamper":
        entry = patched["frames"][item.frame]
        target = None
        for sample in entry["samples"]:
            if sample["row"] == item.payload["row"] and \
                    sample["col"] == item.payload["col"]:
                target = sample
                break
        if target is None:
            raise KeyError("tamper target sample not found")
        target["rgb"] = [v + float(item.payload["delta"]) for v in target["rgb"]]
    else:
        raise ValueError(f"{item.kind} does not tamper the manifest")
    return patched


def apply_state_tamper(state, item):
    """Return a tampered ProcessingState for parameter/constant injections."""
    from dataclasses import replace

    if item.kind == "illegal_transfer_param":
        return replace(state, transfer_gamma=float(item.payload["transfer_gamma"]))
    if item.kind == "tonemap_constant_tamper":
        const = item.payload["constant"]
        factor = float(item.payload["factor"])
        return replace(state, **{const: float(getattr(state, const)) * factor})
    raise ValueError(f"{item.kind} does not tamper the state")


def expected_error_code(item):
    return {
        "peak_out_of_bounds": ErrorCode.INJECT_BOUNDS,
        "illegal_transfer_param": ErrorCode.INJECT_BOUNDS,
        "truth_sample_tamper": ErrorCode.HASH_MISMATCH,
        "tonemap_constant_tamper": ErrorCode.HASH_MISMATCH,
        "stage_replacement": ErrorCode.HASH_MISMATCH,
    }[item.kind]
