"""Typed error codes used by the shard mesh demo."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass
class ShardMeshError(Exception):
    code: str
    message: str
    category: str | None = None
    index: int | None = None
    seq: int | None = None
    field_name: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        parts = [self.code]
        location = []
        if self.category is not None:
            location.append(f"category={self.category}")
        if self.index is not None:
            location.append(f"index={self.index}")
        if self.seq is not None:
            location.append(f"seq={self.seq}")
        if self.field_name is not None:
            location.append(f"field={self.field_name}")
        if location:
            parts.append("(" + " ".join(location) + ")")
        parts.append(self.message)
        return ": ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": False,
            "code": self.code,
            "message": self.message,
            "category": self.category,
            "index": self.index,
            "seq": self.seq,
            "field": self.field_name,
            "details": dict(self.details),
        }


def spec_error(
    index: int | None,
    field_name: str,
    message: str,
    *,
    category: str = "spec",
    seq: int | None = None,
    details: Mapping[str, Any] | None = None,
) -> ShardMeshError:
    return ShardMeshError(
        "E_SPEC",
        message,
        category=category,
        index=index,
        seq=seq,
        field_name=field_name,
        details=details or {},
    )
