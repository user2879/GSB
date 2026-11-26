from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any


@dataclass
class CkptError(Exception):
    """Structured error used by every externally visible failure path."""

    code: str
    detail: str
    item: str | None = None
    line: int | None = None
    shard_id: str | None = None
    batch_id: str | None = None
    field: str | None = None
    extra: dict[str, Any] = dc_field(default_factory=dict)

    def __post_init__(self) -> None:
        Exception.__init__(self, self.detail)

    def __str__(self) -> str:
        where: list[str] = [self.code]
        if self.item is not None:
            where.append(f"item={self.item}")
        if self.line is not None:
            where.append(f"line={self.line}")
        if self.shard_id is not None:
            where.append(f"shard_id={self.shard_id}")
        if self.batch_id is not None:
            where.append(f"batch_id={self.batch_id}")
        if self.field is not None:
            where.append(f"field={self.field}")
        return " ".join(where) + f": {self.detail}"

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code, "detail": self.detail}
        for name in ("item", "line", "shard_id", "batch_id", "field"):
            value = getattr(self, name)
            if value is not None:
                payload[name] = value
        if self.extra:
            payload.update(self.extra)
        return payload


def spec_error(detail: str, *, item: str | None = None, line: int | None = None) -> CkptError:
    return CkptError("E_SPEC", detail, item=item, line=line)


def protocol_error(code: str, detail: str, **kwargs: Any) -> CkptError:
    return CkptError(code, detail, **kwargs)
