"""Typed domain objects shared by validation, storage, and the UI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PageDecl:
    page_id: str
    size_bytes: int
    initial_version: int


@dataclass(frozen=True)
class ReplayBudget:
    global_slots: int
    per_tx_quota: int
    page_byte_budget: int
    checkpoint_interval: int

    def as_dict(self) -> dict[str, int]:
        return {
            "global_slots": self.global_slots,
            "per_tx_quota": self.per_tx_quota,
            "page_byte_budget": self.page_byte_budget,
            "checkpoint_interval": self.checkpoint_interval,
        }


@dataclass(frozen=True)
class WalEntry:
    lsn: int
    txid: str
    page_id: str
    content_digest: str
    commit: bool
    persist: bool


@dataclass(frozen=True)
class PreparedEntry:
    lsn: int
    txid: str
    page_id: str
    content_digest: str
    commit: bool
    persist: bool
    target_version: int
    generated_digest: str
    visible: bool


@dataclass(frozen=True)
class SampleSpec:
    pages: tuple[PageDecl, ...]
    budget: ReplayBudget
    entries: tuple[PreparedEntry, ...]
    crash_point_lsn: int
    committed_txids: tuple[str, ...]
    source_sample: str
    log_file: str
    wal_text: str
    wal_digest: str
    pages_digest: str
    budget_digest: str
    sample_digest: str
    raw_document: dict[str, Any] = field(repr=False, default_factory=dict)

    @property
    def last_lsn(self) -> int:
        return self.entries[-1].lsn if self.entries else 0

    @property
    def initial_cursor(self) -> int:
        return self.entries[0].lsn - 1 if self.entries else 0


@dataclass(frozen=True)
class ReplayTaskSpec:
    task_key: str
    txid: str
    page_id: str
    chunk_index: int
    entries: tuple[PreparedEntry, ...]
    planned_bytes: int

    @property
    def lsn_start(self) -> int:
        return self.entries[0].lsn

    @property
    def lsn_end(self) -> int:
        return self.entries[-1].lsn
