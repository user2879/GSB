from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import json
from pathlib import Path
from typing import Any, Mapping


class ErrorCode(StrEnum):
    E_FORMAT = "E_FORMAT"
    E_INTEGRITY = "E_INTEGRITY"
    E_BUDGET = "E_BUDGET"
    E_PAYLOAD = "E_PAYLOAD"
    E_REPLAY = "E_REPLAY"


@dataclass(slots=True)
class EngineError(Exception):
    code: ErrorCode
    message: str
    path: str | None = None
    record_index: int | None = None
    page_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        Exception.__init__(self, self.message)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "code": self.code.value,
            "message": self.message,
            "details": _stable(self.details),
        }
        if self.path is not None:
            result["path"] = self.path
        if self.record_index is not None:
            result["record_index"] = self.record_index
        if self.page_id is not None:
            result["page_id"] = self.page_id
        return _stable(result)


def _stable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _stable(value[k]) for k in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_stable(item) for item in value]
    if isinstance(value, bytes):
        return value.hex()
    return value


def error_json(error: EngineError) -> str:
    return json.dumps(error.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_structured_log(log_path: str | Path, event: str, error: EngineError | None = None, **extra: Any) -> None:
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"event": event, "status": "error" if error else "ok"}
    payload.update(extra)
    if error is not None:
        payload.update(error.to_dict())
    line = json.dumps(_stable(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line + "\n")
        handle.flush()
