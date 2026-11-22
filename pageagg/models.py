"""Core immutable models shared by loader, planner and execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ColumnDef:
    name: str
    column_type: str

    def payload(self) -> dict[str, str]:
        return {"name": self.name, "type": self.column_type}


@dataclass(frozen=True)
class PageData:
    page_id: int
    rows: int
    path: Path
    content_sha256: str
    columns: dict[str, np.ndarray]
    null_maps: dict[str, np.ndarray]

    def group_keys(self, group_key: str) -> np.ndarray:
        return self.columns[group_key]


@dataclass(frozen=True)
class PageCatalog:
    path: Path
    raw: dict[str, Any]
    digest: str
    dataset_id: str
    rows_per_page: int
    group_key: str
    batch_size: int
    overflow_ratio_threshold_ppm: int
    rows_reference: int
    boundary_margin_ppm: int
    columns: tuple[ColumnDef, ...]
    pages: dict[int, PageData]
    scan_sequence: tuple[int, ...]

    @property
    def metric_columns(self) -> tuple[ColumnDef, ...]:
        return tuple(column for column in self.columns if column.name != self.group_key)

    @property
    def distinct_page_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self.pages))

    @property
    def access_count(self) -> int:
        return len(self.scan_sequence)

