"""Immutable domain models shared by validation, engine, and ledger."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .canonical import object_digest, typed_group_key


@dataclass(frozen=True)
class ColumnDef:
    name: str
    type: str


@dataclass(frozen=True)
class PageMeta:
    page_id: str
    file: str
    start_row: int
    end_row: int

    @property
    def row_count(self) -> int:
        return self.end_row - self.start_row


@dataclass(frozen=True)
class Filter:
    column: str
    op: str
    value: Any = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"column": self.column, "op": self.op}
        if self.op not in ("is_null", "is_not_null"):
            payload["value"] = self.value
        return payload


@dataclass(frozen=True)
class ViewDefinition:
    view_id: str
    group_by: tuple[str, ...]
    aggregates: tuple[str, ...]
    value_column: str | None
    filters: tuple[Filter, ...]
    refresh_mode: str

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "view_id": self.view_id,
            "group_by": list(self.group_by),
            "aggregates": list(self.aggregates),
            "filters": [item.to_dict() for item in self.filters],
            "refresh_mode": self.refresh_mode,
        }
        if self.value_column is not None:
            payload["value_column"] = self.value_column
        return payload

    @property
    def digest(self) -> str:
        return object_digest(self.to_dict())

    @property
    def needs_value_column(self) -> bool:
        return any(item in ("sum", "avg", "mean") for item in self.aggregates)

    def aggregate_order(self) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for item in self.aggregates:
            name = "avg" if item == "mean" else item
            if name not in seen:
                seen.add(name)
                result.append(name)
        return result


@dataclass
class BaseTable:
    name: str
    row_count: int
    row_key: str
    columns: tuple[ColumnDef, ...]
    pages: tuple[PageMeta, ...]
    base_dir: Path
    page_by_id: dict[str, PageMeta] = field(default_factory=dict)

    @property
    def column_map(self) -> dict[str, ColumnDef]:
        return {column.name: column for column in self.columns}

    def page_for_global_row(self, row_index: int) -> PageMeta:
        for page in self.pages:
            if page.start_row <= row_index < page.end_row:
                return page
        raise IndexError(f"row index {row_index} is outside page directory")


@dataclass(frozen=True)
class RuntimeConfig:
    table: BaseTable
    views: tuple[ViewDefinition, ...]
    scale_power: int
    catalog_digest: str
    views_digest: str
    source_catalog_path: str
    source_views_path: str

    @property
    def view_map(self) -> dict[str, ViewDefinition]:
        return {view.view_id: view for view in self.views}

    def scheme_fingerprint(self, budget: int) -> str:
        payload = {
            "catalog_digest": self.catalog_digest,
            "views_digest": self.views_digest,
            "scale_power": self.scale_power,
            "budget": budget,
            "normalized_views": [view.to_dict() for view in self.views],
        }
        return object_digest(payload)


@dataclass(frozen=True)
class Change:
    seq: int
    op: str
    row_id: int
    row: dict[str, Any] | None


@dataclass(frozen=True)
class GroupRef:
    view_id: str
    values: tuple[Any, ...]

    @property
    def key_sort(self) -> str:
        from .canonical import group_sort_key

        return group_sort_key(self.values)

    @property
    def typed_values(self) -> list[list[Any]]:
        return typed_group_key(self.values)

    @property
    def identity(self) -> tuple[str, str]:
        return (self.view_id, self.key_sort)

    def to_dict(self) -> dict[str, Any]:
        return {
            "view_id": self.view_id,
            "group_key": [typed_group_key([value])[0] for value in self.values],
            "group_key_values": list(self.values),
        }


def stable_group_refs(refs: Sequence[GroupRef]) -> list[GroupRef]:
    unique: dict[tuple[str, str], GroupRef] = {}
    for ref in refs:
        unique[ref.identity] = ref
    return sorted(unique.values(), key=lambda ref: (ref.view_id, ref.key_sort))
