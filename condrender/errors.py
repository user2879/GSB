from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


class Codes:
    OK = "E_OK"
    VOCAB_UNKNOWN_TERM = "E_VOCAB_UNKNOWN_TERM"
    VOCAB_EMPTY_PROMPT = "E_VOCAB_EMPTY_PROMPT"
    COND_DIM_MISMATCH = "E_COND_DIM_MISMATCH"
    PNG_SIGNATURE = "E_PNG_SIGNATURE"
    PNG_HEADER = "E_PNG_HEADER"
    PNG_RESOLUTION = "E_PNG_RESOLUTION"
    PNG_TRUNCATED = "E_PNG_TRUNCATED"
    PNG_CRC = "E_PNG_CRC"
    MOTION_GRID = "E_MOTION_GRID"
    MOTION_BLOCK_MISMATCH = "E_MOTION_BLOCK_MISMATCH"
    SEED_RANGE = "E_SEED_RANGE"
    POSITIVE_INT = "E_POSITIVE_INT"
    TILE_ROW_RANGE = "E_TILE_ROW_RANGE"
    QUANT_ENUM = "E_QUANT_ENUM"
    FRAC_BITS_ENUM = "E_FRAC_BITS_ENUM"
    ROUNDING_ENUM = "E_ROUNDING_ENUM"
    NOISE_VERSION = "E_NOISE_VERSION"
    NOISE_VERSION_MISSING = "E_NOISE_VERSION_MISSING"
    NONFINITE_CONDITION = "E_NONFINITE_CONDITION"
    STALE_PREVIEW = "E_STALE_PREVIEW"
    SNAPSHOT_SCHEMA = "E_SNAPSHOT_SCHEMA"
    SNAPSHOT_CORRUPT = "E_SNAPSHOT_CORRUPT"
    BUDGET_EXCEEDED = "E_BUDGET_EXCEEDED"
    COMMAND_INVALID = "E_COMMAND_INVALID"
    INJECTION_UNKNOWN = "E_INJECTION_UNKNOWN"
    CONFIG_INVALID = "E_CONFIG_INVALID"
    FILE_NOT_FOUND = "E_FILE_NOT_FOUND"


@dataclass
class Failure:
    code: str
    message: str
    path: str | None = None
    frame: int | None = None
    stage: str | None = None
    injection_id: str | None = None
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if not self.details:
            data.pop("details", None)
        return data


class ValidationError(Exception):
    def __init__(self, failure: Failure):
        super().__init__(failure.message)
        self.failure = failure


class StageFailure(Exception):
    def __init__(self, failure: Failure):
        super().__init__(failure.message)
        self.failure = failure


class StalePreview(Exception):
    def __init__(self, summary: dict[str, Any]):
        super().__init__("interaction budget expired before a fresh summary")
        self.summary = summary
