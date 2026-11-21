from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


E_FIELD_TRUNCATED = "E_FIELD_TRUNCATED"
E_VERSION_MISMATCH = "E_VERSION_MISMATCH"
E_COORD_INVALID = "E_COORD_INVALID"
E_SCORE_INVALID = "E_SCORE_INVALID"
E_CLASS_UNKNOWN = "E_CLASS_UNKNOWN"
E_NON_FINITE = "E_NON_FINITE"
E_IMAGE_INVALID = "E_IMAGE_INVALID"
E_CONFIG_INVALID = "E_CONFIG_INVALID"
E_SNAPSHOT_INVALID = "E_SNAPSHOT_INVALID"
E_MATCH_INVALID = "E_MATCH_INVALID"
E_SERIALIZE_INVALID = "E_SERIALIZE_INVALID"
E_IO = "E_IO"
E_UNEXPECTED = "E_UNEXPECTED"


@dataclass
class BusinessError(Exception):
    code: str
    message: str
    stage: str
    series_name: str | None = None
    frame_index: int | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        location = ""
        if self.series_name is not None:
            location = f" [{self.series_name}:{self.frame_index}]"
        return f"{self.code}{location}: {self.message}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_code": self.code,
            "message": self.message,
            "stage": self.stage,
            "series_name": self.series_name,
            "frame_index": self.frame_index,
            "details": self.details,
        }


def as_business_error(
    exc: BaseException,
    *,
    stage: str,
    series_name: str | None = None,
    frame_index: int | None = None,
) -> BusinessError:
    if isinstance(exc, BusinessError):
        if exc.series_name is None:
            exc.series_name = series_name
        if exc.frame_index is None:
            exc.frame_index = frame_index
        return exc
    return BusinessError(
        code=E_UNEXPECTED,
        message=str(exc) or exc.__class__.__name__,
        stage=stage,
        series_name=series_name,
        frame_index=frame_index,
        details={"exception_type": exc.__class__.__name__},
    )
