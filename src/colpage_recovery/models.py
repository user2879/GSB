from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


SUPPORTED_DTYPES = {
    "uint8": ("<u1", 1, 1),
    "int32": ("<i4", 4, 2),
    "int64": ("<i8", 8, 3),
    "float32": ("<f4", 4, 4),
    "float64": ("<f8", 8, 5),
}


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    dtype: str

    @property
    def canonical_dtype(self) -> str:
        return SUPPORTED_DTYPES[self.dtype][0]

    @property
    def item_size(self) -> int:
        return SUPPORTED_DTYPES[self.dtype][1]

    @property
    def type_code(self) -> int:
        return SUPPORTED_DTYPES[self.dtype][2]


@dataclass(frozen=True)
class PageSpec:
    page_id: str
    row_count: int
    columns: tuple[ColumnSpec, ...]
    null_bitmap: tuple[bool, ...]


@dataclass(frozen=True)
class WalEntry:
    lsn: int
    tx_id: str
    page_id: str
    row_start: int
    row_end: int
    seed: int
    commit_flag: bool
    persist_flag: bool

    @property
    def width(self) -> int:
        return self.row_end - self.row_start


@dataclass(frozen=True)
class Budget:
    global_slots: int = 2
    per_tx_quota: int = 1
    page_byte_budget: int = 65536
    checkpoint_interval: int = 3
    ledger_batch_entries: int = 2
    marker_protocol: str = "steal-no-force"
    work_delay_ms: int = 0


@dataclass
class Bundle:
    pages: list[PageSpec]
    entries: list[WalEntry]
    budget: Budget
    pages_payload: dict[str, Any] = field(default_factory=dict)
    wal_payload: dict[str, Any] = field(default_factory=dict)
    budget_payload: dict[str, Any] = field(default_factory=dict)

    @property
    def page_map(self) -> dict[str, PageSpec]:
        return {page.page_id: page for page in self.pages}

    def batches(self, batch_size: int | None = None) -> list[list[WalEntry]]:
        size = batch_size or self.budget.ledger_batch_entries
        if size <= 0:
            raise ValueError("batch size must be positive")
        return [self.entries[i : i + size] for i in range(0, len(self.entries), size)]
