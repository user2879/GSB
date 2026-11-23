"""YAML/JSONL/JSON loading and strict input validation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Config
from .errors import (
    E_DUPLICATE_ID,
    E_EDIT_TYPE,
    E_EMPTY_SEQUENCE,
    E_INDEX_RANGE,
    E_LOAD,
    E_OBS_VALUE,
    E_SEGMENT_ID,
    E_SEGMENT_LENGTH,
    E_SEQUENCE_ID,
    E_TIMESTAMP_INTERVAL,
    Failure,
)
from .models import EditBatch, LibraryData, SequenceRecord


@dataclass
class LibraryLoadResult:
    library: LibraryData
    failures: list[Failure]


@dataclass
class EditPlanLoadResult:
    batches: list[EditBatch]
    failures: list[Failure]


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    return isinstance(value, float) and math.isfinite(value)


def _is_missing(value: Any, marker: Any) -> bool:
    if value is None:
        return True
    if marker is None:
        return False
    try:
        return value == marker
    except Exception:  # pragma: no cover
        return False


def validate_sequence_record(
    raw: Any,
    config: Config,
    *,
    record_index: int,
    stage: str = "library",
) -> tuple[SequenceRecord | None, list[Failure]]:
    failures: list[Failure] = []
    if not isinstance(raw, dict):
        return None, [
            Failure(
                E_LOAD,
                stage,
                "sequence record must be a JSON object",
                record_index=record_index,
            )
        ]

    seq_id = raw.get("id", raw.get("sequence_id"))
    if not isinstance(seq_id, str) or not seq_id.strip():
        failures.append(
            Failure(
                E_SEQUENCE_ID,
                stage,
                "sequence id must be a non-empty string",
                record_index=record_index,
                detail={"value": seq_id},
            )
        )
        seq_id = None

    observations = raw.get("observations", raw.get("values"))
    if not isinstance(observations, list) or not observations:
        failures.append(
            Failure(
                E_EMPTY_SEQUENCE,
                stage,
                "observations must be a non-empty list",
                sequence_id=seq_id,
                record_index=record_index,
            )
        )
        observations = []

    checked_observations: list[Any] = []
    for index, value in enumerate(observations):
        if _is_missing(value, config.missing_marker):
            checked_observations.append(None if value is None else value)
            continue
        if not _is_finite_number(value):
            failures.append(
                Failure(
                    E_OBS_VALUE,
                    stage,
                    "observation must be finite or the configured missing marker",
                    sequence_id=seq_id,
                    record_index=record_index,
                    detail={"index": index, "value": repr(value)},
                )
            )
            continue
        checked_observations.append(value)

    timestamp_interval = raw.get("timestamp_interval", 1.0)
    if not _is_finite_number(timestamp_interval) or float(timestamp_interval) <= 0:
        failures.append(
            Failure(
                E_TIMESTAMP_INTERVAL,
                stage,
                "timestamp_interval must be a finite positive number",
                sequence_id=seq_id,
                record_index=record_index,
                detail={"value": repr(timestamp_interval)},
            )
        )

    segment_ids = raw.get("segment_ids", raw.get("segments"))
    if segment_ids is None:
        segment_ids = [0] * len(observations)
    if not isinstance(segment_ids, list):
        failures.append(
            Failure(
                E_SEGMENT_LENGTH,
                stage,
                "segment_ids must be a list",
                sequence_id=seq_id,
                record_index=record_index,
            )
        )
        segment_ids = []
    elif len(segment_ids) != len(observations):
        failures.append(
            Failure(
                E_SEGMENT_LENGTH,
                stage,
                "segment_ids length must equal observations length",
                sequence_id=seq_id,
                record_index=record_index,
                detail={"segment_count": len(segment_ids), "point_count": len(observations)},
            )
        )
    for index, segment_id in enumerate(segment_ids):
        if isinstance(segment_id, bool) or not isinstance(segment_id, (str, int)):
            failures.append(
                Failure(
                    E_SEGMENT_ID,
                    stage,
                    "segment id must be a string or integer",
                    sequence_id=seq_id,
                    record_index=record_index,
                    detail={"index": index, "value": repr(segment_id)},
                )
            )

    if failures:
        return None, failures
    return (
        SequenceRecord(
            seq_id=seq_id,
            observations=tuple(checked_observations),
            timestamp_interval=timestamp_interval,
            segment_ids=tuple(segment_ids),
        ),
        [],
    )


def load_library(path: str | Path, config: Config) -> LibraryLoadResult:
    library = LibraryData()
    failures: list[Failure] = []
    seen: set[str] = set()
    source_index = 0
    try:
        handle = Path(path).open("r", encoding="utf-8")
    except OSError as exc:
        return LibraryLoadResult(
            library, [Failure(E_LOAD, "library", f"failed to open JSONL: {exc}")]
        )

    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                failures.append(
                    Failure(
                        E_LOAD,
                        "library",
                        f"invalid JSON on line {line_number}: {exc.msg}",
                        record_index=source_index,
                    )
                )
                source_index += 1
                continue

            if isinstance(raw, dict) and isinstance(raw.get("id", raw.get("sequence_id")), str):
                candidate_id = raw.get("id", raw.get("sequence_id"))
                if candidate_id in seen:
                    failure = Failure(
                        E_DUPLICATE_ID,
                        "library",
                        "sequence id must be unique",
                        sequence_id=candidate_id,
                        record_index=source_index,
                    )
                    failures.append(failure)
                    library.invalid_entries.append(
                        {"source_index": source_index, "raw": raw, "failure": failure.to_dict()}
                    )
                    library.raw_entries.append({"source_index": source_index, "raw": raw})
                    source_index += 1
                    continue
                seen.add(candidate_id)

            record, record_failures = validate_sequence_record(
                raw, config, record_index=source_index
            )
            library.raw_entries.append({"source_index": source_index, "raw": raw})
            if record_failures:
                failures.extend(record_failures)
                library.invalid_entries.append(
                    {
                        "source_index": source_index,
                        "raw": raw,
                        "failures": [item.to_dict() for item in record_failures],
                    }
                )
            else:
                assert record is not None
                library.records[record.seq_id] = record
            source_index += 1

    return LibraryLoadResult(library, failures)


def validate_edit_shape(edit: Any, *, batch: int, edit_index: int) -> list[Failure]:
    if not isinstance(edit, dict):
        return [
            Failure(
                E_EDIT_TYPE,
                "edit",
                "edit must be an object",
                batch=batch,
                edit_index=edit_index,
            )
        ]
    edit_type = edit.get("type")
    allowed = {
        "sequence_replace",
        "point_update",
        "point_correction",
        "sequence_append",
        "bandwidth_update",
        "step_mode_update",
        "step_mode_switch",
    }
    if edit_type not in allowed:
        return [
            Failure(
                E_EDIT_TYPE,
                "edit",
                f"unknown edit type {edit_type!r}",
                batch=batch,
                edit_index=edit_index,
                detail={"allowed": sorted(allowed)},
            )
        ]
    if edit_type == "bandwidth_update":
        return []
    if edit_type in {"step_mode_update", "step_mode_switch"}:
        return []
    edit_sequence_id = edit.get("sequence_id", edit.get("id"))
    if edit_type == "sequence_replace" and edit_sequence_id is None:
        record = edit.get("record")
        if isinstance(record, dict):
            edit_sequence_id = record.get("id", record.get("sequence_id"))
    if not isinstance(edit_sequence_id, str):
        return [
            Failure(
                E_SEQUENCE_ID,
                "edit",
                "edit sequence_id must be a string",
                batch=batch,
                edit_index=edit_index,
                detail={"value": repr(edit_sequence_id)},
            )
        ]
    if edit_type in {"point_update", "point_correction"} and not isinstance(
        edit.get("index"), int
    ):
        return [
            Failure(
                E_INDEX_RANGE,
                "edit",
                "point update index must be an integer",
                sequence_id=edit_sequence_id,
                batch=batch,
                edit_index=edit_index,
            )
        ]
    return []


def load_edit_plan(path: str | Path) -> EditPlanLoadResult:
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except Exception as exc:
        return EditPlanLoadResult([], [Failure(E_LOAD, "edit_plan", f"failed to load JSON: {exc}")])

    failures: list[Failure] = []
    if isinstance(raw, list):
        raw_batches = raw
    elif isinstance(raw, dict) and isinstance(raw.get("batches"), list):
        raw_batches = raw["batches"]
    else:
        return EditPlanLoadResult(
            [], [Failure(E_LOAD, "edit_plan", "plan must be a list or {batches: [...]}")]
        )

    batches: list[EditBatch] = []
    for fallback_batch, raw_batch in enumerate(raw_batches, start=1):
        if not isinstance(raw_batch, dict):
            failures.append(
                Failure(E_LOAD, "edit_plan", "batch must be an object", batch=fallback_batch)
            )
            continue
        batch_number = raw_batch.get("batch", fallback_batch)
        if not isinstance(batch_number, int) or isinstance(batch_number, bool):
            failures.append(
                Failure(E_LOAD, "edit_plan", "batch number must be an integer", batch=fallback_batch)
            )
            batch_number = fallback_batch
        raw_edits = raw_batch.get("edits", [])
        if not isinstance(raw_edits, list):
            failures.append(
                Failure(E_LOAD, "edit_plan", "edits must be a list", batch=batch_number)
            )
            raw_edits = []
        edits: list[dict[str, Any]] = []
        for edit_index, edit in enumerate(raw_edits):
            edit_failures = validate_edit_shape(
                edit, batch=batch_number, edit_index=edit_index
            )
            if edit_failures:
                failures.extend(edit_failures)
            else:
                edits.append(dict(edit))
        batches.append(EditBatch(batch=batch_number, edits=tuple(edits)))
    return EditPlanLoadResult(batches, failures)


def write_library_jsonl(path: str | Path, records: list[SequenceRecord]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True))
            handle.write("\n")
