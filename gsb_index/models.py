"""Runtime models for configuration, documents, revisions and results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class Vocabulary:
    terms: tuple[str, ...]
    stopwords: frozenset[str]

    @property
    def term_set(self) -> frozenset[str]:
        return frozenset(self.terms)


@dataclass(frozen=True)
class Document:
    doc_id: str
    version: int
    shard_id: str
    fields: Mapping[str, tuple[str, ...]]
    field_stopwords: Mapping[str, bool]

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "version": self.version,
            "shard_id": self.shard_id,
            "fields": {name: list(values) for name, values in self.fields.items()},
            "field_stopwords": dict(self.field_stopwords),
        }


@dataclass(frozen=True)
class ShardSpec:
    shard_id: str
    port: int
    term_bucket_capacity: int
    concurrency_limit: int
    seed: int
    revision: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "shard_id": self.shard_id,
            "port": self.port,
            "term_bucket_capacity": self.term_bucket_capacity,
            "concurrency_limit": self.concurrency_limit,
            "seed": self.seed,
            "revision": self.revision,
        }


@dataclass(frozen=True)
class CompressionProfile:
    bucket_count: int
    bucket_weights: tuple[float, ...]
    encoding: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "bucket_count": self.bucket_count,
            "bucket_weights": list(self.bucket_weights),
            "encoding": self.encoding,
        }


@dataclass(frozen=True)
class BuildBudget:
    global_term_entry_budget: int
    per_shard_term_entry_limit: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "global_term_entry_budget": self.global_term_entry_budget,
            "per_shard_term_entry_limit": self.per_shard_term_entry_limit,
        }


@dataclass(frozen=True)
class ScoringProfile:
    bit_width: int = 32
    fraction_bits: int = 8
    quantum_step: int = 1
    title_weight: int = 3
    body_weight: int = 2
    tf_saturation_k: int = 4

    def to_dict(self) -> dict[str, Any]:
        return {
            "bit_width": self.bit_width,
            "fraction_bits": self.fraction_bits,
            "quantum_step": self.quantum_step,
            "title_weight": self.title_weight,
            "body_weight": self.body_weight,
            "tf_saturation_k": self.tf_saturation_k,
        }


@dataclass(frozen=True)
class Query:
    query_id: str
    terms: tuple[str, ...]
    top_k: int = 5

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "terms": list(self.terms),
            "top_k": self.top_k,
        }


@dataclass(frozen=True)
class RevisionNode:
    version: int
    parent: int | None
    profile: CompressionProfile
    changes: Mapping[str, Any] = field(default_factory=dict)
    encoding: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "parent": self.parent,
            "profile": self.profile.to_dict(),
            "changes": dict(self.changes),
        }


@dataclass(frozen=True)
class Project:
    vocabulary: Vocabulary
    documents: tuple[Document, ...]
    shards: tuple[ShardSpec, ...]
    compression: CompressionProfile
    budget: BuildBudget
    revisions: tuple[RevisionNode, ...]
    current_revision: int
    scoring: ScoringProfile
    queries: tuple[Query, ...]
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def shard_map(self) -> dict[str, ShardSpec]:
        return {shard.shard_id: shard for shard in self.shards}

    def document_by_id(self) -> dict[str, Document]:
        return {doc.doc_id: doc for doc in self.documents}

    def documents_by_shard(self) -> dict[str, list[Document]]:
        result = {shard.shard_id: [] for shard in self.shards}
        for doc in self.documents:
            result.setdefault(doc.shard_id, []).append(doc)
        return result

    def profile_for_revision(self, version: int) -> CompressionProfile:
        for node in self.revisions:
            if node.version == version:
                return node.profile
        if version == self.current_revision:
            return self.compression
        raise KeyError(version)

    def config_payload(self) -> dict[str, Any]:
        return {
            "vocabulary": {
                "terms": list(self.vocabulary.terms),
                "stopwords": sorted(self.vocabulary.stopwords),
            },
            "shards": [shard.to_dict() for shard in self.shards],
            "compression": self.profile_for_revision(self.current_revision).to_dict(),
            "budget": self.budget.to_dict(),
            "revisions": [node.to_dict() for node in self.revisions],
            "current_revision": self.current_revision,
            "scoring": self.scoring.to_dict(),
            "queries": [query.to_dict() for query in self.queries],
        }

