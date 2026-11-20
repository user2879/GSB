"""Business errors. Every user-visible failure carries a stable code."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


ERROR_DESCRIPTIONS = {
    "E_SCHEMA_VERSION": "Unsupported or inconsistent schema/version declaration.",
    "E_ENDIAN_MISMATCH": "Binary byte-order marker or declared endianness is inconsistent.",
    "E_INDEX_OVERFLOW": "Index width cannot represent the declared frame/segment index.",
    "E_METRIC_VERSION": "Metric encoding version is unsupported or mislabeled.",
    "E_AUDIO_DRIFT": "Audio offset exceeds the declared timebase tolerance.",
    "E_EQUIV_MISMATCH": "Carrier conclusions are not equivalent under the canonical model.",
    "E_CLOSURE_MISMATCH": "Executed recomputation set differs from the invalidation closure.",
    "E_SNAPSHOT_STALE": "Ledger commit is based on a stale snapshot revision.",
    "E_FIELD_BITMAP": "Metric field bitmap is invalid or inconsistent with the version.",
    "E_SCHEMA": "Required schema fields or values are missing/invalid.",
    "E_CARRIER_PARSE": "A carrier could not be parsed completely.",
    "E_DECODE": "A frame or audio unit could not be decoded.",
    "E_DROPPED_FRAME": "A frame was removed by injected input parameters.",
    "E_TIMESTAMP_JUMP": "A frame timestamp exceeds the declared tolerance.",
    "E_AUDIO_SAMPLE_RATE": "Declared audio sample rate differs from the WAV header.",
    "E_INJECTION": "Injection configuration is invalid.",
    "E_ASSET": "Synthetic asset is missing, corrupt, or has an unexpected hash.",
    "E_LEDGER_CORRUPT": "Committed ledger snapshot failed hash validation.",
    "E_RECOVERY": "Committed ledger recovery could not be completed.",
    "E_CONTENT_HASH": "Content hash does not match the declared asset set.",
    "E_IO": "Local file operation failed.",
}


@dataclass
class BusinessError(Exception):
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__init__(f"{self.code}: {self.message}")
        if self.code not in ERROR_DESCRIPTIONS:
            ERROR_DESCRIPTIONS[self.code] = self.message

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "description": ERROR_DESCRIPTIONS.get(self.code, self.message),
            "details": self.details,
        }


def as_error(exc: BaseException) -> BusinessError:
    if isinstance(exc, BusinessError):
        return exc
    return BusinessError("E_IO", str(exc), {"type": type(exc).__name__})
