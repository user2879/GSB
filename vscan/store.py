from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .canonical import (
    SplitMix64,
    atomic_write,
    canonical_json,
    digest_value,
    stable_column_seed,
)
from .codec import ColumnFileInfo, file_info, write_column
from .errors import VScanError
from .spec import ColumnSpec, SegmentSpec, ScanSpec


STORE_FORMAT_VERSION = 1


class StoreError(VScanError):
    code = "E_STORE"


def segment_dir(root: str | Path, segment_id: str) -> Path:
    return Path(root) / "segments" / segment_id


def column_path(root: str | Path, segment_id: str, column: str) -> Path:
    return segment_dir(root, segment_id) / f"{column}.col"


def manifest_path(root: str | Path) -> Path:
    return Path(root) / "manifest.json"


def generate_column_values(
    segment: SegmentSpec,
    column: ColumnSpec,
    overrides: Mapping[int, int | None] | None = None,
) -> list[int | None]:
    rng = SplitMix64(stable_column_seed(segment.seed, column.name, column.seed))
    span = column.value_max - column.value_min + 1
    values: list[int | None] = []
    for row in range(segment.rows):
        null_draw = rng.next_u64()
        value_draw = rng.next_u64()
        value = column.value_min + int(value_draw % span)
        if null_draw % 1000 < column.null_per_mille:
            value = None
        if overrides is not None and row in overrides:
            value = overrides[row]
        values.append(value)
    return values


@dataclass(frozen=True)
class ColumnState:
    name: str
    encoding: str
    file: str
    file_digest: str
    column_digest: str
    rows: int
    non_null_count: int
    payload_bytes: int
    file_bytes: int

    @classmethod
    def from_info(cls, name: str, file_name: str, info: ColumnFileInfo) -> "ColumnState":
        body = {
            "name": name,
            "encoding": info.encoding,
            "file_digest": info.file_digest,
            "rows": info.rows,
            "non_null_count": info.non_null_count,
        }
        return cls(
            name=name,
            encoding=info.encoding,
            file=file_name,
            file_digest=info.file_digest,
            column_digest=digest_value(body),
            rows=info.rows,
            non_null_count=info.non_null_count,
            payload_bytes=info.payload_bytes,
            file_bytes=info.file_bytes,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "encoding": self.encoding,
            "file": self.file,
            "file_digest": self.file_digest,
            "column_digest": self.column_digest,
            "rows": self.rows,
            "non_null_count": self.non_null_count,
            "payload_bytes": self.payload_bytes,
            "file_bytes": self.file_bytes,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ColumnState":
        return cls(
            name=str(data["name"]),
            encoding=str(data["encoding"]),
            file=str(data["file"]),
            file_digest=str(data["file_digest"]),
            column_digest=str(data["column_digest"]),
            rows=int(data["rows"]),
            non_null_count=int(data["non_null_count"]),
            payload_bytes=int(data["payload_bytes"]),
            file_bytes=int(data["file_bytes"]),
        )


@dataclass(frozen=True)
class SegmentState:
    id: str
    rows: int
    seed: int
    segment_digest: str
    columns: tuple[ColumnState, ...]

    def column_map(self) -> dict[str, ColumnState]:
        return {column.name: column for column in self.columns}

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "rows": self.rows,
            "seed": self.seed,
            "segment_digest": self.segment_digest,
            "columns": [column.to_dict() for column in self.columns],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SegmentState":
        return cls(
            id=str(data["id"]),
            rows=int(data["rows"]),
            seed=int(data["seed"]),
            segment_digest=str(data["segment_digest"]),
            columns=tuple(
                sorted(
                    (ColumnState.from_dict(column) for column in data["columns"]),
                    key=lambda column: column.name,
                )
            ),
        )


@dataclass(frozen=True)
class StoreState:
    format_version: int
    segments: tuple[SegmentState, ...]
    content_digest: str

    def segment_map(self) -> dict[str, SegmentState]:
        return {segment.id: segment for segment in self.segments}

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "content_digest": self.content_digest,
            "segments": [segment.to_dict() for segment in self.segments],
        }

    @property
    def snapshot_digest(self) -> str:
        return digest_value(self.canonical_dict())

    def to_dict(self) -> dict[str, Any]:
        return {**self.canonical_dict(), "snapshot_digest": self.snapshot_digest}


def _segment_digest(segment: SegmentSpec, columns: Iterable[ColumnState]) -> str:
    return digest_value(
        {
            "id": segment.id,
            "rows": segment.rows,
            "seed": segment.seed,
            "columns": [
                {
                    "name": column.name,
                    "encoding": column.encoding,
                    "column_digest": column.column_digest,
                }
                for column in sorted(columns, key=lambda column: column.name)
            ],
        }
    )


def _store_digest(segments: Iterable[SegmentState]) -> str:
    return digest_value(
        {
            "format_version": STORE_FORMAT_VERSION,
            "segments": [
                {
                    "id": segment.id,
                    "segment_digest": segment.segment_digest,
                }
                for segment in sorted(segments, key=lambda segment: segment.id)
            ],
        }
    )


def build_column(
    root: str | Path,
    segment: SegmentSpec,
    column: ColumnSpec,
    overrides: Mapping[int, int | None] | None = None,
) -> ColumnState:
    values = generate_column_values(segment, column, overrides)
    path = column_path(root, segment.id, column.name)
    info = write_column(path, values, column.encoding)
    return ColumnState.from_info(column.name, path.name, info)


def build_segment(
    root: str | Path,
    segment: SegmentSpec,
    overrides: Mapping[str, Mapping[int, int | None]] | None = None,
    *,
    columns: Iterable[str] | None = None,
    previous: SegmentState | None = None,
) -> SegmentState:
    selected = set(columns) if columns is not None else None
    column_specs = {column.name: column for column in segment.columns}
    states: dict[str, ColumnState] = {}
    if previous is not None and selected is not None:
        for state in previous.columns:
            if state.name not in selected:
                states[state.name] = state
    for name in sorted(column_specs):
        if selected is not None and name not in selected:
            continue
        column_overrides = None if overrides is None else overrides.get(name)
        states[name] = build_column(root, segment, column_specs[name], column_overrides)
    missing = set(column_specs) - set(states)
    if missing:
        raise StoreError(f"segment {segment.id!r} is missing column state {sorted(missing)[0]!r}")
    ordered = tuple(states[name] for name in sorted(states))
    return SegmentState(
        id=segment.id,
        rows=segment.rows,
        seed=segment.seed,
        segment_digest=_segment_digest(segment, ordered),
        columns=ordered,
    )


def build_store(
    spec: ScanSpec,
    root: str | Path,
    overrides: Mapping[str, Mapping[str, Mapping[int, int | None]]] | None = None,
    *,
    previous: StoreState | None = None,
    columns_by_segment: Mapping[str, Iterable[str]] | None = None,
) -> StoreState:
    previous_segments = previous.segment_map() if previous is not None else {}
    segments: list[SegmentState] = []
    for segment in sorted(spec.segments, key=lambda item: item.id):
        segment_overrides = None if overrides is None else overrides.get(segment.id)
        selected = None if columns_by_segment is None else columns_by_segment.get(segment.id, ())
        segments.append(
            build_segment(
                root,
                segment,
                segment_overrides,
                columns=selected,
                previous=previous_segments.get(segment.id),
            )
        )
    ordered = tuple(sorted(segments, key=lambda segment: segment.id))
    state = StoreState(
        format_version=STORE_FORMAT_VERSION,
        segments=ordered,
        content_digest=_store_digest(ordered),
    )
    atomic_write(
        manifest_path(root),
        (canonical_json(state.canonical_dict()) + "\n").encode("utf-8"),
    )
    return state


def load_store(root: str | Path) -> StoreState:
    path = manifest_path(root)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StoreError(f"cannot load store manifest: {exc}") from exc
    try:
        state = StoreState(
            format_version=int(data["format_version"]),
            segments=tuple(
                sorted(
                    (SegmentState.from_dict(item) for item in data["segments"]),
                    key=lambda segment: segment.id,
                )
            ),
            content_digest=str(data["content_digest"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise StoreError(f"invalid store manifest: {exc}") from exc
    if state.format_version != STORE_FORMAT_VERSION:
        raise StoreError(
            f"unsupported store format version {state.format_version}; expected {STORE_FORMAT_VERSION}"
        )
    if state.content_digest != _store_digest(state.segments):
        raise StoreError("store manifest content digest mismatch")
    for segment in state.segments:
        for column in segment.columns:
            path = column_path(root, segment.id, column.name)
            if not path.exists():
                raise StoreError(f"column file is missing: {path}")
            info = file_info(path)
            if info.file_digest != column.file_digest:
                raise StoreError(
                    f"column file digest mismatch: segment={segment.id} column={column.name}"
                )
    return state


def column_info_dicts(state: StoreState) -> list[dict[str, Any]]:
    return [
        {
            "segment_id": segment.id,
            "column": column.name,
            "encoding": column.encoding,
            "non_null_count": column.non_null_count,
            "column_digest": column.column_digest,
        }
        for segment in state.segments
        for column in segment.columns
    ]
