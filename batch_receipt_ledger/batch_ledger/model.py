from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .hashing import stable_ref

MISSING = "$missing"

CORE_FIELDS = ("batch_id", "request_id", "output_digest")
VERSION_2_FIELDS = ("param_version", "elapsed_ms", "worker_id")
VERSION_3_FIELDS = ("rank_in_batch", "supersedes_ref", "param_digest")
OPTIONAL_FIELDS = VERSION_2_FIELDS + VERSION_3_FIELDS
ALL_DATA_FIELDS = CORE_FIELDS + OPTIONAL_FIELDS


def is_missing(value: Any) -> bool:
    return value == MISSING


def receipt_ref_for(batch_id: Any, request_id: Any, output_digest: Any) -> str:
    return stable_ref("rr", [batch_id, request_id, output_digest])


@dataclass(frozen=True)
class CanonicalReceipt:
    source_version: str
    batch_id: str
    request_id: str
    output_digest: str
    param_version: Any = MISSING
    elapsed_ms: Any = MISSING
    worker_id: Any = MISSING
    rank_in_batch: Any = MISSING
    supersedes_ref: Any = MISSING
    param_digest: Any = MISSING
    missing_fields: tuple[str, ...] = ()
    completeness_label: str = "unknown"
    receipt_ref: str = ""
    origin_version: str = ""

    def __post_init__(self) -> None:
        for field_name in OPTIONAL_FIELDS:
            value = getattr(self, field_name)
            if value == MISSING and value is not MISSING:
                object.__setattr__(self, field_name, MISSING)
        if not self.receipt_ref:
            object.__setattr__(
                self,
                "receipt_ref",
                receipt_ref_for(self.batch_id, self.request_id, self.output_digest),
            )
        if not self.origin_version:
            object.__setattr__(self, "origin_version", self.source_version)
        missing = tuple(name for name in ALL_DATA_FIELDS[3:] if is_missing(getattr(self, name)))
        if tuple(self.missing_fields) != missing:
            object.__setattr__(self, "missing_fields", missing)
        present = len(ALL_DATA_FIELDS) - len(missing)
        if not self.completeness_label or self.completeness_label == "unknown":
            if not missing:
                label = "complete_r3"
            elif all(is_missing(getattr(self, name)) for name in VERSION_3_FIELDS) and not any(
                is_missing(getattr(self, name)) for name in VERSION_2_FIELDS
            ):
                label = "r2_core"
            elif all(is_missing(getattr(self, name)) for name in OPTIONAL_FIELDS):
                label = "r1_core"
            else:
                label = "partial"
            object.__setattr__(self, "completeness_label", label)
        elif self.completeness_label == "complete" and missing:
            object.__setattr__(self, "completeness_label", "partial")
        # present is retained through completeness_score() to avoid mutable state.
        _ = present

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["missing_fields"] = list(self.missing_fields)
        data["completeness_score"] = self.completeness_score
        return data

    @property
    def completeness_score(self) -> int:
        return len(ALL_DATA_FIELDS) - len(self.missing_fields)

    def semantic_dict(self) -> dict[str, Any]:
        """Fields used by the same-physical-receipt equivalence assertion."""
        data = self.to_dict()
        data.pop("source_version", None)
        return data


@dataclass(frozen=True)
class Failure:
    code: str
    message: str
    source: str = ""
    record_index: int = -1
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def sort_key(self) -> tuple[Any, ...]:
        return (self.source, self.record_index, self.code, self.message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "source": self.source,
            "record_index": self.record_index,
            "details": dict(self.details),
        }


@dataclass
class LedgerReport:
    """Thin wrapper used by callers that prefer a typed return."""

    report: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return self.report
