"""Deterministic fault injection plan and stage-specific fault mutation."""

from dataclasses import dataclass
from typing import Optional


INJ_TIMESTAMP_ORDER = "timestamp_reorder"
INJ_SELF_REFERENCE = "relation_self_reference"
INJ_TYPE_UNKNOWN = "evidence_type_unknown"
INJ_WIDTH_OVERFLOW = "fixed_width_overflow"

INJ_KINDS = (INJ_TIMESTAMP_ORDER, INJ_SELF_REFERENCE,
             INJ_TYPE_UNKNOWN, INJ_WIDTH_OVERFLOW)


@dataclass(frozen=True)
class Injection:
    inj_id: str
    kind: str
    tid: str
    stage: str
    target_seq: Optional[int] = None

    @classmethod
    def from_dict(cls, data: dict) -> "Injection":
        return cls(
            inj_id=data["inj_id"],
            kind=data["kind"],
            tid=data["tid"],
            stage=data["stage"],
            target_seq=(None if data.get("target_seq") is None
                        else int(data["target_seq"])),
        )
