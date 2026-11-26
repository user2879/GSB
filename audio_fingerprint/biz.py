from __future__ import annotations

from dataclasses import dataclass
from typing import Any


OK = "BIZ_OK"
WAV_FILE_MISSING = "E1001"
WAV_HEADER_INVALID = "E1002"
SAMPLE_RATE_NOT_DECLARED = "E1003"
BIT_DEPTH_NOT_DECLARED = "E1004"
DATA_CHUNK_MISMATCH = "E1005"
MANIFEST_SCHEMA_INVALID = "E2001"
DUPLICATE_TRACK_VERSION = "E2002"
COMMAND_SCHEMA_INVALID = "E2003"
INVALID_PEAK_THRESHOLD = "E3001"
INVALID_BUCKET_CAPACITY = "E3002"
INVALID_TIME_GEAR = "E3003"
INVALID_BUDGET = "E3004"
INVALID_CONFIG = "E3005"
CLIP_QUOTA_EXCEEDED = "B4001"
BATCH_BUDGET_EXHAUSTED = "B4002"
FRAME_FAILED = "B4003"
GRACEFUL_SHUTDOWN = "B4004"


@dataclass(frozen=True)
class BusinessIssue:
    code: str
    target: str
    message: str
    detail: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "target": self.target,
            "message": self.message,
            "detail": self.detail or {},
        }


def issue_dict(code: str, target: str, message: str, **detail: Any) -> dict[str, Any]:
    return {"code": code, "target": target, "message": message, "detail": detail}
