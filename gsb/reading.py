"""Reading-order assembly and deterministic conflict adjudication."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gsb.config import RunConfig
from gsb.constants import (
    CONFLICT_CROSS_COLUMN,
    CONFLICT_MISORDER,
    CONFLICT_MISSING,
    CONFLICT_PRIORITY,
)
from gsb.keypoints import KeypointResult
from gsb.models import ColumnSpec
from gsb.util import sha256_obj, task_sort_key


@dataclass(frozen=True)
class ReadingEntry:
    position: int
    keypoint_id: str
    column_observed: int
    column_source: str
    x: int
    y: int
    reading_index_truth: int
    cache_key: str
    sort_key: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": self.position,
            "keypoint_id": self.keypoint_id,
            "column": self.column_observed,
            "column_source": self.column_source,
            "x": self.x,
            "y": self.y,
            "reading_index_truth": self.reading_index_truth,
            "cache_key": self.cache_key,
            "sort_key": self.sort_key,
        }


@dataclass(frozen=True)
class Conflict:
    sequence: int
    after_keypoint_id: str | None
    keypoint_id: str
    conflict_type: str
    types: tuple[str, ...]
    priority: int
    previous_column: int | None
    current_column: int
    previous_truth_index: int | None
    current_truth_index: int
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "after_keypoint_id": self.after_keypoint_id,
            "keypoint_id": self.keypoint_id,
            "conflict_type": self.conflict_type,
            "types": list(self.types),
            "priority": self.priority,
            "previous_column": self.previous_column,
            "current_column": self.current_column,
            "previous_truth_index": self.previous_truth_index,
            "current_truth_index": self.current_truth_index,
            "message": self.message,
        }


@dataclass(frozen=True)
class ReadingResult:
    entries: tuple[ReadingEntry, ...]
    conflicts: tuple[Conflict, ...]
    order_cache_key: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "entries": [entry.to_dict() for entry in self.entries],
            "conflicts": [conflict.to_dict() for conflict in self.conflicts],
            "order_cache_key": self.order_cache_key,
        }


def _assign_column(
    keypoint: KeypointResult,
    columns: dict[int, ColumnSpec],
) -> tuple[int, str]:
    for column in sorted(columns.values(), key=lambda item: item.index):
        if column.contains(keypoint.x):
            return column.index, "interval"
    if not columns:
        return -1, "missing"
    nearest = min(
        columns.values(),
        key=lambda item: (
            abs(keypoint.x - (item.x_min + item.x_max) // 2),
            item.index,
        ),
    )
    return nearest.index, "nearest"


def assemble_reading_order(
    keypoints: list[KeypointResult],
    *,
    columns: tuple[ColumnSpec, ...],
    config: RunConfig,
    image_fingerprint: str,
    parameter_fingerprint: str,
    template_fingerprint: str,
    file_name: str,
) -> ReadingResult:
    column_lookup = {column.index: column for column in columns}
    assigned: list[tuple[KeypointResult, int, str]] = []
    for keypoint in keypoints:
        observed, source = _assign_column(keypoint, column_lookup)
        assigned.append((keypoint, observed, source))
    assigned.sort(
        key=lambda item: (
            item[1],
            item[0].y,
            -item[0].x,
            item[0].reading_index_truth,
            item[0].keypoint_id,
        )
    )
    order_payload = {
        "stage": "reading",
        "image": image_fingerprint,
        "parameters": parameter_fingerprint,
        "template_fingerprint": template_fingerprint,
        "template_version": config.recognition.template_version,
        "quantization": config.quantization.level,
        "keypoints": [keypoint.cache_key for keypoint, _, _ in assigned],
        "priority": list(config.recognition.reading_conflict_priority),
    }
    order_cache_key = sha256_obj(order_payload)
    entries: list[ReadingEntry] = []
    for position, (keypoint, observed_column, source) in enumerate(assigned, start=1):
        entry_cache = sha256_obj(
            {
                "stage": "reading_entry",
                "image": image_fingerprint,
                "parameters": parameter_fingerprint,
                "template_fingerprint": template_fingerprint,
                "template_version": config.recognition.template_version,
                "quantization": config.quantization.level,
                "keypoint": keypoint.cache_key,
                "position": position,
                "column": observed_column,
            }
        )
        entries.append(
            ReadingEntry(
                position=position,
                keypoint_id=keypoint.keypoint_id,
                column_observed=observed_column,
                column_source=source,
                x=keypoint.x,
                y=keypoint.y,
                reading_index_truth=keypoint.reading_index_truth,
                cache_key=entry_cache,
                sort_key=task_sort_key(
                    Path(file_name).name,
                    "reading",
                    observed_column,
                    position,
                    position,
                ),
            )
        )

    conflicts: list[Conflict] = []
    previous: tuple[KeypointResult, int, str] | None = None
    for current in assigned:
        keypoint, observed_column, _ = current
        if previous is not None:
            previous_keypoint, previous_column, _ = previous
            types: list[str] = []
            if observed_column < previous_column or observed_column != keypoint.column_truth:
                types.append(CONFLICT_CROSS_COLUMN)
            if (
                keypoint.reading_index_truth <= previous_keypoint.reading_index_truth
                or (
                    observed_column == previous_column
                    and keypoint.y < previous_keypoint.y
                )
            ):
                types.append(CONFLICT_MISORDER)
            if keypoint.reading_index_truth != previous_keypoint.reading_index_truth + 1:
                types.append(CONFLICT_MISSING)
            if types:
                priority_lookup = {
                    conflict_type: index
                    for index, conflict_type in enumerate(
                        config.recognition.reading_conflict_priority
                    )
                }
                types.sort(key=lambda item: (priority_lookup[item], CONFLICT_PRIORITY[item]))
                selected = types[0]
                conflicts.append(
                    Conflict(
                        sequence=len(conflicts) + 1,
                        after_keypoint_id=previous_keypoint.keypoint_id,
                        keypoint_id=keypoint.keypoint_id,
                        conflict_type=selected,
                        types=tuple(types),
                        priority=priority_lookup[selected] + 1,
                        previous_column=previous_column,
                        current_column=observed_column,
                        previous_truth_index=previous_keypoint.reading_index_truth,
                        current_truth_index=keypoint.reading_index_truth,
                        message=(
                            f"{selected}: observed ({observed_column},{keypoint.y}) after "
                            f"({previous_column},{previous_keypoint.y}) with truth sequence "
                            f"{previous_keypoint.reading_index_truth}->{keypoint.reading_index_truth}"
                        ),
                    )
                )
        previous = current
    return ReadingResult(
        entries=tuple(entries),
        conflicts=tuple(conflicts),
        order_cache_key=order_cache_key,
    )
