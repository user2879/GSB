"""Business error codes and failure-record helpers."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

E_AUDIO_FORMAT = "E_AUDIO_FORMAT"
E_CONFIG_INVALID = "E_CONFIG_INVALID"
E_DECODE_FAILED = "E_DECODE_FAILED"
E_FEATURE_FAILED = "E_FEATURE_FAILED"
E_REDECODE_BUDGET_PENDING = "E_REDECODE_BUDGET_PENDING"
E_SNAPSHOT_INVALID = "E_SNAPSHOT_INVALID"
E_SIGNAL_EXIT = "E_SIGNAL_EXIT"


class BusinessError(Exception):
    """Exception carrying a stable business code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class FailureRecord:
    source: str
    stage: str
    code: str
    exception_type: str
    message: str
    chunk_index: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)
