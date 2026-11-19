"""Apply real input faults selected by the independent injection config."""
from __future__ import annotations

from dataclasses import dataclass

from .config import Injection
from .errors import (
    CandidateFormatError,
    DetboxError,
    UnknownClassError,
    WeightNonFiniteError,
)
from .exchange import candidate_to_v1_line
from .models import Candidate


@dataclass
class InjectionHit:
    injection_id: str
    sequence: str
    frame: int
    stage: str
    fault_type: str
    code: str

    def to_dict(self) -> dict:
        return {
            "injection_id": self.injection_id,
            "sequence": self.sequence,
            "frame": self.frame,
            "stage": self.stage,
            "fault_type": self.fault_type,
            "code": self.code,
        }


FAULT_CODE = {
    "truncated_png": "E_IMAGE_TRUNCATED",
    "nonfinite_weight": "E_WEIGHT_NONFINITE",
    "candidate_corrupt": "E_CAND_FORMAT",
    "v1_missing_field": "E_CAND_FORMAT",
    "coords_out_of_bounds": "E_CAND_FORMAT",
    "class_unknown": "E_CLASS_UNKNOWN",
}


def injections_for(
    injections: list[Injection], sequence: str, frame: int, stage: str
) -> list[Injection]:
    return [
        item
        for item in injections
        if item.sequence == sequence and item.frame == frame and item.stage == stage
    ]


def apply_detect_injections(
    injections: list[Injection], image_path, params
) -> InjectionHit | None:
    for item in injections:
        if item.fault_type == "truncated_png":
            # Real fault: truncate the bytes the detector reads for this frame.
            raw = open(image_path, "rb").read()
            truncated = raw[: max(1, len(raw) // 3)]
            from io import BytesIO

            from PIL import Image

            try:
                Image.open(BytesIO(truncated)).load()
            except Exception as exc:
                raise DetboxError(
                    f"injected truncated PNG: {exc}", code="E_IMAGE_TRUNCATED"
                ) from exc
            raise DetboxError("injected truncated PNG", code="E_IMAGE_TRUNCATED")
        if item.fault_type == "nonfinite_weight":
            raise WeightNonFiniteError("injected non-finite fusion weight")
    return None


def apply_match_injections(
    injections: list[Injection], frames
) -> tuple[InjectionHit | None, object]:
    for item in injections:
        if item.fault_type == "candidate_corrupt":
            return (
                InjectionHit(
                    item.injection_id,
                    item.sequence,
                    item.frame,
                    item.stage,
                    item.fault_type,
                    FAULT_CODE[item.fault_type],
                ),
                CandidateFormatError("injected corrupt candidate before matching"),
            )
    return None, None


def apply_serialize_injections(
    injections: list[Injection], candidate: Candidate, sequence: str, frame: int
) -> tuple[InjectionHit | None, Exception | None]:
    for item in injections:
        code = FAULT_CODE[item.fault_type]
        hit = InjectionHit(
            item.injection_id,
            item.sequence,
            item.frame,
            item.stage,
            item.fault_type,
            code,
        )
        if item.fault_type == "v1_missing_field":
            # Real fault: feed the fixed-width parser a short, field-missing line.
            bad_line = "seqX       00000001  005  010"
            return hit, CandidateFormatError(
                f"injected fixed-width missing field, line width {len(bad_line)}"
            )
        if item.fault_type == "coords_out_of_bounds":
            bad = Candidate(
                x1=9999.0,
                y1=9999.0,
                x2=10030.0,
                y2=10030.0,
                score=0.5,
                class_code=candidate.class_code,
                int_box=(9999, 9999, 10030, 10030),
                int_score=500,
            )
            line = candidate_to_v1_line(bad, sequence, frame)
            return hit, CandidateFormatError(
                f"injected out-of-bounds coordinates: {line.strip()[:20]}..."
            )
        if item.fault_type == "class_unknown":
            return hit, UnknownClassError("injected class code outside class table")
    return None, None
