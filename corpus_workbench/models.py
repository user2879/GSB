"""Shared immutable data records and stable serialisation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .codes import STAGE_RANK
from .util import sha256_text


@dataclass(frozen=True)
class Document:
    source: str
    doc_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    fingerprint: str = ""
    file: str = ""
    line_no: int = 0
    injection_id: str | None = None

    @property
    def stable_key(self) -> str:
        return f"{self.source}\x1f{self.doc_id}"

    def to_dict(self, include_text: bool = False) -> dict[str, Any]:
        value = {
            "source": self.source,
            "doc_id": self.doc_id,
            "fingerprint": self.fingerprint,
            "metadata": self.metadata,
            "file": self.file,
            "line_no": self.line_no,
            "injection_id": self.injection_id,
            "stable_key": self.stable_key,
        }
        if include_text:
            value["text"] = self.text
        return value


@dataclass(frozen=True)
class SampleRecord:
    source: str
    doc_id: str
    sample_index: int
    order: int

    @property
    def stable_key(self) -> str:
        return f"{self.source}\x1f{self.doc_id}\x1f{self.sample_index:08d}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "doc_id": self.doc_id,
            "sample_index": self.sample_index,
            "order": self.order,
            "stable_key": self.stable_key,
        }


@dataclass(frozen=True)
class Failure:
    source: str
    doc_id: str
    stage: str
    code: str
    message: str
    injection_id: str | None = None
    file: str | None = None
    line_no: int | None = None

    @property
    def stable_key(self) -> str:
        return "\x1f".join(
            [
                f"{STAGE_RANK.get(self.stage, 99):02d}",
                self.source,
                self.doc_id,
                "" if self.line_no is None else f"{self.line_no:012d}",
                self.code,
                self.injection_id or "",
            ]
        )

    def sort_key(self) -> tuple[Any, ...]:
        return (
            STAGE_RANK.get(self.stage, 99),
            self.source,
            self.doc_id,
            -1 if self.line_no is None else self.line_no,
            self.code,
            self.injection_id or "",
            self.message,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "doc_id": self.doc_id,
            "stage": self.stage,
            "code": self.code,
            "message": self.message,
            "injection_id": self.injection_id,
            "file": self.file,
            "line_no": self.line_no,
            "stable_key": self.stable_key,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Failure":
        return cls(
            source=str(value.get("source", "")),
            doc_id=str(value.get("doc_id", "")),
            stage=str(value.get("stage", "")),
            code=str(value.get("code", "")),
            message=str(value.get("message", "")),
            injection_id=value.get("injection_id"),
            file=value.get("file"),
            line_no=value.get("line_no"),
        )


@dataclass(frozen=True)
class SignatureRecord:
    source: str
    doc_id: str
    fingerprint: str
    shingle_length: int
    signature_dim: int
    point_cost: int
    signature: tuple[int, ...]
    cache_key: str
    cache_hit: bool = False
    reused_from_snapshot: bool = False

    @property
    def stable_key(self) -> str:
        return f"{self.source}\x1f{self.doc_id}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "doc_id": self.doc_id,
            "fingerprint": self.fingerprint,
            "shingle_length": self.shingle_length,
            "signature_dim": self.signature_dim,
            "point_cost": self.point_cost,
            "signature": list(self.signature),
            "cache_key": self.cache_key,
            "cache_hit": self.cache_hit,
            "reused_from_snapshot": self.reused_from_snapshot,
            "stable_key": self.stable_key,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SignatureRecord":
        return cls(
            source=str(value["source"]),
            doc_id=str(value["doc_id"]),
            fingerprint=str(value["fingerprint"]),
            shingle_length=int(value["shingle_length"]),
            signature_dim=int(value["signature_dim"]),
            point_cost=int(value["point_cost"]),
            signature=tuple(int(part) for part in value["signature"]),
            cache_key=str(value["cache_key"]),
            cache_hit=bool(value.get("cache_hit", False)),
            reused_from_snapshot=bool(value.get("reused_from_snapshot", False)),
        )


def sorted_failures(failures: list[Failure]) -> list[Failure]:
    return sorted(failures, key=Failure.sort_key)


def content_fingerprint(text: str) -> str:
    """Fingerprint the exact UTF-8 document text."""
    return sha256_text(text)
