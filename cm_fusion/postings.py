from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, Iterable

from .config import Config
from .errors import PipelineError
from .loaders import Document
from .util import sha256_json


FIELDS = ("title", "body")


@dataclass(frozen=True)
class Posting:
    doc_id: str
    tf: int
    positions: tuple[int, ...] | None

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "tf": self.tf,
            "positions": None if self.positions is None else list(self.positions),
        }


@dataclass(frozen=True)
class PostingBlock:
    field: str
    term: str
    block_index: int
    postings: tuple[Posting, ...]
    lane: str

    @property
    def key(self) -> str:
        return f"{self.field}\x1f{self.term}\x1f{self.block_index:04d}"

    def to_dict(self) -> dict:
        return {
            "field": self.field,
            "term": self.term,
            "block_index": self.block_index,
            "lane": self.lane,
            "postings": [posting.to_dict() for posting in self.postings],
        }


@dataclass(frozen=True)
class LogicalIndex:
    documents: tuple[Document, ...]
    postings: dict[str, dict[str, tuple[Posting, ...]]]
    doc_ordinals: dict[str, int]
    term_refs: dict[str, int]
    keep_positions: bool

    def logical_digest(self) -> str:
        return logical_postings_digest(self.postings, self.keep_positions)

    def document_count(self) -> int:
        return len(self.documents)

    def posting_count(self) -> int:
        return sum(len(items) for field in FIELDS for items in self.postings[field].values())

    def block_count(self, block_size: int) -> int:
        return sum(_chunk_count(len(items), block_size) for field in FIELDS for items in self.postings[field].values())

    def blocks(self, config: Config, tier_id: int | None = None) -> list[PostingBlock]:
        tier = config.tier(tier_id)
        block_size = config.posting_block_size
        blocks: list[PostingBlock] = []
        for field in FIELDS:
            for term in sorted(self.postings[field]):
                items = self.postings[field][term]
                lane = "lowfreq" if tier.low_frequency_filter and len(items) < tier.low_frequency_threshold else "main"
                for start in range(0, len(items), block_size):
                    chunk = tuple(sorted(items[start : start + block_size], key=lambda posting: posting.doc_id))
                    blocks.append(PostingBlock(field, term, start // block_size, chunk, lane))
        return blocks


def stable_doc_ordinal(doc_id: str) -> int:
    digest = hashlib.sha256(doc_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _chunk_count(size: int, block_size: int) -> int:
    return (size + block_size - 1) // block_size


def build_logical_index(
    documents: Iterable[Document],
    config: Config,
    checkpoint: Callable[[], None] | None = None,
) -> LogicalIndex:
    doc_list = sorted(documents, key=lambda doc: doc.doc_id)
    ordinals: dict[str, int] = {}
    reverse: dict[int, str] = {}
    for doc in doc_list:
        ordinal = stable_doc_ordinal(doc.doc_id)
        if ordinal in reverse:
            raise PipelineError(
                "E_INDEX_INVALID",
                "document ID hash collision",
                {"doc_ids": [reverse[ordinal], doc.doc_id]},
            )
        ordinals[doc.doc_id] = ordinal
        reverse[ordinal] = doc.doc_id

    postings: dict[str, dict[str, list[Posting]]] = {field: {} for field in FIELDS}
    for doc in doc_list:
        if checkpoint is not None:
            checkpoint()
        for field, tokens in (("title", doc.title_terms), ("body", doc.body_terms)):
            positions_by_term: dict[str, list[int]] = {}
            for position, term in enumerate(tokens):
                positions_by_term.setdefault(term, []).append(position)
            for term, positions in positions_by_term.items():
                postings[field].setdefault(term, []).append(
                    Posting(
                        doc_id=doc.doc_id,
                        tf=len(positions),
                        positions=tuple(positions) if config.keep_positions else None,
                    )
                )

    frozen: dict[str, dict[str, tuple[Posting, ...]]] = {}
    for field in FIELDS:
        frozen[field] = {
            term: tuple(sorted(items, key=lambda posting: posting.doc_id))
            for term, items in sorted(postings[field].items())
        }
    term_keys = sorted((field, term) for field in FIELDS for term in frozen[field])
    term_refs = {f"{field}\x1f{term}": index for index, (field, term) in enumerate(term_keys)}
    return LogicalIndex(tuple(doc_list), frozen, ordinals, term_refs, config.keep_positions)


def logical_index_from_postings(
    documents: Iterable[Document],
    postings: dict[str, dict[str, list[Posting] | tuple[Posting, ...]]],
    keep_positions: bool,
) -> LogicalIndex:
    doc_list = sorted(documents, key=lambda doc: doc.doc_id)
    ordinals = {doc.doc_id: stable_doc_ordinal(doc.doc_id) for doc in doc_list}
    frozen = {
        field: {
            term: tuple(sorted(items, key=lambda posting: posting.doc_id))
            for term, items in sorted(postings.get(field, {}).items())
        }
        for field in FIELDS
    }
    term_keys = sorted((field, term) for field in FIELDS for term in frozen[field])
    term_refs = {f"{field}\x1f{term}": index for index, (field, term) in enumerate(term_keys)}
    return LogicalIndex(tuple(doc_list), frozen, ordinals, term_refs, keep_positions)


def canonical_postings_digest(postings: dict[str, dict[str, tuple[Posting, ...]]]) -> str:
    payload = {
        field: {
            term: [posting.to_dict() for posting in sorted(items, key=lambda p: p.doc_id)]
            for term, items in sorted(postings[field].items())
        }
        for field in FIELDS
    }
    return sha256_json(payload)


def logical_postings_digest(
    postings: dict[str, dict[str, tuple[Posting, ...]]],
    keep_positions: bool,
) -> str:
    return sha256_json(
        {
            "fields": {
                field: {
                    term: [posting.to_dict() for posting in sorted(items, key=lambda p: p.doc_id)]
                    for term, items in sorted(postings[field].items())
                }
                for field in FIELDS
            },
            "keep_positions": keep_positions,
        }
    )
