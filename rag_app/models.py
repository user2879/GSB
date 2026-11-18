"""Core data models for documents, queries, configuration and failures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .fingerprint import canonical_json, fingerprint


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    body: str
    source: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Document":
        return cls(
            doc_id=str(data["doc_id"]),
            title=str(data.get("title", "")),
            body=str(data["body"]),
            source=str(data.get("source", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "body": self.body,
            "source": self.source,
        }


@dataclass(frozen=True)
class QueryTurn:
    session_id: str
    turn_no: int
    query: str
    is_followup: bool
    context_ref: tuple[int, ...]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QueryTurn":
        refs = data.get("context_ref") or []
        return cls(
            session_id=str(data["session_id"]),
            turn_no=int(data["turn_no"]),
            query=str(data["query"]),
            is_followup=bool(data["is_followup"]),
            context_ref=tuple(int(r) for r in refs),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "turn_no": self.turn_no,
            "query": self.query,
            "is_followup": self.is_followup,
            "context_ref": list(self.context_ref),
        }


@dataclass(frozen=True)
class Command:
    seq: int
    op: str
    args: dict[str, Any]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Command":
        return cls(
            seq=int(data["seq"]),
            op=str(data["op"]),
            args=dict(data.get("args") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"seq": self.seq, "op": self.op, "args": self.args}


@dataclass
class Failure:
    """A validation or business failure with a deterministic sort key."""

    code: str
    scope: str
    message: str
    sort_key: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "scope": self.scope,
            "message": self.message,
            "sort_key": self.sort_key,
        }


@dataclass(frozen=True)
class RetrievalConfig:
    normalization: dict[str, bool]
    bm25: dict[str, float]
    recall_top_k: int
    answer_window: int
    min_citations: int
    fusion_weights: dict[str, float]
    first_response_budget_ms: int
    max_concurrency: int
    snapshot_commit_interval: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RetrievalConfig":
        return cls(
            normalization=dict(data["normalization"]),
            bm25=dict(data["bm25"]),
            recall_top_k=int(data["recall_top_k"]),
            answer_window=int(data["answer_window"]),
            min_citations=int(data["min_citations"]),
            fusion_weights=dict(data["fusion_weights"]),
            first_response_budget_ms=int(data["first_response_budget_ms"]),
            max_concurrency=int(data["max_concurrency"]),
            snapshot_commit_interval=int(data["snapshot_commit_interval"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "normalization": dict(self.normalization),
            "bm25": dict(self.bm25),
            "recall_top_k": self.recall_top_k,
            "answer_window": self.answer_window,
            "min_citations": self.min_citations,
            "fusion_weights": dict(self.fusion_weights),
            "first_response_budget_ms": self.first_response_budget_ms,
            "max_concurrency": self.max_concurrency,
            "snapshot_commit_interval": self.snapshot_commit_interval,
        }

    def fingerprint(self) -> str:
        return fingerprint(self.to_dict())

    def replace(self, **changes: Any) -> "RetrievalConfig":
        data = self.to_dict()
        data.update(changes)
        return RetrievalConfig.from_dict(data)

    def __str__(self) -> str:  # pragma: no cover - debugging convenience
        return canonical_json(self.to_dict())


def turn_sort_key(session_id: str, turn_no: int) -> str:
    """Stable lexicographic sort key derived from session id and turn no."""
    return f"{session_id}#{turn_no:04d}"


def turn_tuple(sort_key: str) -> tuple[str, int]:
    session_id, turn_no = sort_key.split("#")
    return session_id, int(turn_no)


@dataclass
class MutableState:
    """All engine inputs that can be mutated by interaction commands."""

    docs: list[Document]
    turns: dict[tuple[str, int], QueryTurn] = field(default_factory=dict)

    def ordered_turns(self) -> list[QueryTurn]:
        return [self.turns[key] for key in sorted(self.turns)]

