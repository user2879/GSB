from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Any


E_SOURCE = "E_SOURCE"
E_FORMAT = "E_FORMAT"
E_EMPTY = "E_EMPTY"
E_REPLAY = "E_REPLAY"


@dataclass
class SlowDiagError(Exception):
    code: str
    message: str
    path: str | None = None
    source_id: str | None = None
    row_no: int | None = None
    query_id: str | None = None
    metric: str | None = None
    field: str | None = None
    details: dict[str, Any] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
        }
        if self.path is not None:
            payload["path"] = self.path
        if self.source_id is not None:
            payload["source_id"] = self.source_id
        if self.row_no is not None:
            payload["row_no"] = self.row_no
        if self.query_id is not None:
            payload["query_id"] = self.query_id
        if self.metric is not None:
            payload["metric"] = self.metric
        if self.field is not None:
            payload["field"] = self.field
        if self.details:
            payload["details"] = self.details
        return payload


def source_error(message: str, path: str, **kw: Any) -> SlowDiagError:
    return SlowDiagError(E_SOURCE, message, path=path, **kw)


def format_error(
    message: str,
    *,
    source_id: str,
    row_no: int | None,
    path: str | None = None,
    **kw: Any,
) -> SlowDiagError:
    return SlowDiagError(
        E_FORMAT,
        message,
        path=path,
        source_id=source_id,
        row_no=row_no,
        **kw,
    )
