"""Incremental DTW diagnostics engine with deterministic scheduling."""

from __future__ import annotations

import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .align import (
    AlignmentResult,
    CellProduct,
    compute_cell,
    feasible_coordinates,
    finalize_alignment,
)
from .band import Band, build_band
from .config import Config
from .errors import (
    E_BANDWIDTH,
    E_CELL_BUDGET_PENDING,
    E_EMPTY_SEQUENCE,
    E_EDIT_TYPE,
    E_INDEX_RANGE,
    E_OBS_VALUE,
    E_SEGMENT_LENGTH,
    E_SEQUENCE_NOT_FOUND,
    E_STEP_MODE,
    Failure,
    sorted_failures,
)
from .fingerprints import fingerprint, task_key
from .invalidation import InvalidationPlan, build_invalidation_plan
from .models import EditBatch, LibraryData, SequenceRecord
from .quantization import QuantizedSequence, quantize_sequence
from .storage import validate_sequence_record


def sequence_target_key(
    query: SequenceRecord,
    reference_record: SequenceRecord,
    query_quantified: QuantizedSequence,
    reference_quantified: QuantizedSequence,
    band: Band,
    config: Config,
) -> str:
    return fingerprint(
        {
            "stage": "sequence_result_target",
            "sequence_id": query.seq_id,
            "reference_id": reference_record.seq_id,
            "query_record_fingerprint": query.fingerprint(),
            "reference_record_fingerprint": reference_record.fingerprint(),
            "query_point_content_key": query_quantified.point_content_key,
            "reference_point_content_key": reference_quantified.point_content_key,
            "band_fingerprint": band.fingerprint,
            "alignment_fingerprint": config.alignment_fingerprint(),
            "max_path_length": config.max_path_length,
            "path_length_policy": config.path_length_policy,
        }
    )


@dataclass
class Artifact:
    query: SequenceRecord
    reference: SequenceRecord
    query_quantified: QuantizedSequence
    reference_quantified: QuantizedSequence
    band: Band
    config: Config
    cells: dict[tuple[int, int], CellProduct]
    target_key: str
    complete: bool
    result: AlignmentResult | None = None
    plan: InvalidationPlan | None = None
    pending_cells: list[tuple[int, int]] = field(default_factory=list)
    recomputed_total: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query.to_dict(),
            "reference": self.reference.to_dict(),
            "query_quantified": self.query_quantified.to_dict(),
            "reference_quantified": self.reference_quantified.to_dict(),
            "band": self.band.to_dict(),
            "config": self.config.to_dict(),
            "cells": [self.cells[coord].to_dict() for coord in sorted(self.cells)],
            "target_key": self.target_key,
            "complete": self.complete,
            "result": self.result.to_dict() if self.result is not None else None,
            "plan": self.plan.to_dict() if self.plan is not None else None,
            "pending_cells": [list(item) for item in self.pending_cells],
            "recomputed_total": self.recomputed_total,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Artifact":
        return cls(
            query=SequenceRecord.from_dict(raw["query"]),
            reference=SequenceRecord.from_dict(raw["reference"]),
            query_quantified=QuantizedSequence.from_dict(raw["query_quantified"]),
            reference_quantified=QuantizedSequence.from_dict(
                raw["reference_quantified"]
            ),
            band=Band.from_dict(raw["band"]),
            config=Config(**raw["config"]),
            cells={
                (item["row"], item["col"]): CellProduct.from_dict(item)
                for item in raw.get("cells", [])
            },
            target_key=raw["target_key"],
            complete=raw["complete"],
            result=AlignmentResult.from_dict(raw["result"])
            if raw.get("result") is not None
            else None,
            plan=InvalidationPlan.from_dict(raw["plan"])
            if raw.get("plan") is not None
            else None,
            pending_cells=[tuple(item) for item in raw.get("pending_cells", [])],
            recomputed_total=raw.get("recomputed_total", 0),
        )


@dataclass
class Slot:
    committed: Artifact | None = None
    working: Artifact | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "committed": self.committed.to_dict() if self.committed is not None else None,
            "working": self.working.to_dict() if self.working is not None else None,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Slot":
        return cls(
            committed=Artifact.from_dict(raw["committed"])
            if raw.get("committed") is not None
            else None,
            working=Artifact.from_dict(raw["working"])
            if raw.get("working") is not None
            else None,
        )


@dataclass
class WorkJob:
    sequence_id: str
    artifact: Artifact
    allocate_cells: int
    priority: bool
    sequence_task_key: str


class DiagnosticEngine:
    """Stateful batch scheduler.

    Budget allocation happens in the main thread in stable sequence order. Only
    independent sequence jobs enter the thread pool; a job always consumes its
    allocated cells in row/column order.
    """

    schema_version = 1

    def __init__(
        self,
        config: Config,
        library: LibraryData,
        validation_failures: list[Failure] | None = None,
        *,
        workdir: str | Path,
        reset: bool = False,
    ) -> None:
        self.config = config
        self.library = library
        self.validation_failures = list(validation_failures or [])
        self.workdir = Path(workdir)
        self.snapshot_dir = self.workdir / "snapshots"
        self.batch_report_dir = self.workdir / "batches"
        if reset:
            self._reset_workdir()
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.batch_report_dir.mkdir(parents=True, exist_ok=True)
        self.slots: dict[str, Slot] = {}
        self.batch_ledger: list[dict[str, Any]] = []
        self.invalidation_ledger: list[dict[str, Any]] = []
        self.budget_ledger: list[dict[str, Any]] = []
        self.edit_failure_ledger: list[dict[str, Any]] = []
        self.reuse_ledger: list[dict[str, Any]] = []
        self.last_batch = 0
        ref_id = self.config.reference_sequence_id
        if isinstance(ref_id, str) and ref_id and ref_id not in self.library.records:
            self.validation_failures.append(
                Failure(
                    E_SEQUENCE_NOT_FOUND,
                    "reference",
                    "configured reference_sequence_id is absent; self-alignment fallback is used",
                    sequence_id=ref_id,
                )
            )

    def _reset_workdir(self) -> None:
        if not self.workdir.exists():
            return
        for path in sorted(self.workdir.rglob("*"), reverse=True):
            if path.is_file() or path.is_symlink():
                path.unlink(missing_ok=True)
            elif path.is_dir():
                path.rmdir()
        if self.workdir.exists():
            self.workdir.rmdir()

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------
    def to_state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "config": self.config.to_dict(),
            "library": {
                "records": {
                    seq_id: record.to_dict()
                    for seq_id, record in sorted(self.library.records.items())
                },
                "invalid_entries": self.library.invalid_entries,
                "raw_entries": self.library.raw_entries,
            },
            "validation_failures": [
                failure.to_dict() for failure in self.validation_failures
            ],
            "slots": {seq_id: slot.to_dict() for seq_id, slot in sorted(self.slots.items())},
            "batch_ledger": self.batch_ledger,
            "invalidation_ledger": self.invalidation_ledger,
            "budget_ledger": self.budget_ledger,
            "edit_failure_ledger": self.edit_failure_ledger,
            "reuse_ledger": self.reuse_ledger,
            "last_batch": self.last_batch,
        }

    @classmethod
    def from_state_dict(
        cls, raw: dict[str, Any], *, workdir: str | Path
    ) -> "DiagnosticEngine":
        library_raw = raw["library"]
        library = LibraryData(
            records={
                seq_id: SequenceRecord.from_dict(record)
                for seq_id, record in library_raw.get("records", {}).items()
            },
            invalid_entries=library_raw.get("invalid_entries", []),
            raw_entries=library_raw.get("raw_entries", []),
        )
        failures = [
            Failure(
                code=item["code"],
                stage=item["stage"],
                message=item["message"],
                sequence_id=item.get("sequence_id"),
                batch=item.get("batch"),
                edit_index=item.get("edit_index"),
                record_index=item.get("record_index"),
                detail=item.get("detail", {}),
            )
            for item in raw.get("validation_failures", [])
        ]
        engine = cls(
            Config(**raw["config"]), library, failures, workdir=workdir, reset=False
        )
        engine.slots = {
            seq_id: Slot.from_dict(slot_raw)
            for seq_id, slot_raw in raw.get("slots", {}).items()
        }
        engine.batch_ledger = raw.get("batch_ledger", [])
        engine.invalidation_ledger = raw.get("invalidation_ledger", [])
        engine.budget_ledger = raw.get("budget_ledger", [])
        engine.edit_failure_ledger = raw.get("edit_failure_ledger", [])
        engine.reuse_ledger = raw.get("reuse_ledger", [])
        engine.last_batch = raw.get("last_batch", 0)
        return engine

    @classmethod
    def load_latest(cls, *, workdir: str | Path) -> "DiagnosticEngine":
        latest = Path(workdir) / "snapshots" / "latest.json"
        with latest.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        return cls.from_state_dict(raw, workdir=workdir)

    def _atomic_json(self, path: Path, payload: Any, *, pretty: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            if pretty:
                json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
            else:
                json.dump(
                    payload,
                    handle,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            handle.write("\n")
        os.replace(temporary, path)

    def persist(self, *, archive: bool = False) -> dict[str, str]:
        latest = self.snapshot_dir / "latest.json"
        self._atomic_json(latest, self.to_state_dict())
        paths = {"latest": str(latest)}
        if archive:
            archive_path = self.snapshot_dir / f"batch_{self.last_batch:06d}.json"
            self._atomic_json(archive_path, self.to_state_dict())
            paths["archive"] = str(archive_path)
        return paths

    # ------------------------------------------------------------------
    # Edit application
    # ------------------------------------------------------------------
    def _edit_failure(
        self,
        code: str,
        message: str,
        *,
        batch: int,
        edit_index: int,
        sequence_id: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> Failure:
        return Failure(
            code,
            "edit",
            message,
            sequence_id=sequence_id,
            batch=batch,
            edit_index=edit_index,
            detail=detail or {},
        )

    def _clear_validation_failure(self, sequence_id: str) -> None:
        self.validation_failures = [
            item for item in self.validation_failures if item.sequence_id != sequence_id
        ]
        self.library.invalid_entries = [
            item
            for item in self.library.invalid_entries
            if not (
                isinstance(item.get("raw"), dict)
                and item["raw"].get("id", item["raw"].get("sequence_id")) == sequence_id
            )
        ]

    def _install_record(
        self, record: SequenceRecord, *, batch: int, edit_index: int
    ) -> list[Failure]:
        checked, failures = validate_sequence_record(
            record.to_dict(), self.config, record_index=edit_index, stage="edit"
        )
        if failures:
            return [
                Failure(
                    code=item.code,
                    stage=item.stage,
                    message=item.message,
                    sequence_id=record.seq_id,
                    batch=batch,
                    edit_index=edit_index,
                    record_index=item.record_index,
                    detail=item.detail,
                )
                for item in failures
            ]
        assert checked is not None
        self.library.records[checked.seq_id] = checked
        self._clear_validation_failure(checked.seq_id)
        return []

    def _validate_values(
        self,
        values: Any,
        *,
        batch: int,
        edit_index: int,
        sequence_id: str,
    ) -> tuple[list[Any] | None, list[Failure]]:
        if not isinstance(values, list) or not values:
            return None, [
                self._edit_failure(
                    E_EMPTY_SEQUENCE,
                    "observations must be a non-empty list",
                    batch=batch,
                    edit_index=edit_index,
                    sequence_id=sequence_id,
                )
            ]
        checked: list[Any] = []
        failures: list[Failure] = []
        marker = self.config.missing_marker
        for index, value in enumerate(values):
            if value is None or (marker is not None and value == marker):
                checked.append(None if value is None else value)
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or (isinstance(value, float) and not math.isfinite(value))
            ):
                failures.append(
                    self._edit_failure(
                        E_OBS_VALUE,
                        "observation must be finite or the configured missing marker",
                        batch=batch,
                        edit_index=edit_index,
                        sequence_id=sequence_id,
                        detail={"index_in_edit": index, "value": repr(value)},
                    )
                )
            else:
                checked.append(value)
        return (None if failures else checked), failures

    def apply_edit(
        self, edit: dict[str, Any], *, batch: int, edit_index: int
    ) -> list[Failure]:
        edit_type = edit.get("type")
        seq_id = edit.get("sequence_id", edit.get("id"))
        if edit_type == "sequence_replace" and not isinstance(seq_id, str):
            raw_record_for_id = edit.get("record")
            if isinstance(raw_record_for_id, dict):
                seq_id = raw_record_for_id.get(
                    "id", raw_record_for_id.get("sequence_id")
                )

        if edit_type == "bandwidth_update":
            value = edit.get("bandwidth_permille")
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or not 0 <= value <= 1000
            ):
                return [
                    self._edit_failure(
                        E_BANDWIDTH,
                        "bandwidth_permille must be an integer in [0, 1000]",
                        batch=batch,
                        edit_index=edit_index,
                        detail={"value": repr(value)},
                    )
                ]
            self.config = self.config.replace_bandwidth(value)
            return []

        if edit_type in {"step_mode_update", "step_mode_switch"}:
            value = edit.get("step_mode")
            if value not in {"symmetric", "asymmetric"}:
                return [
                    self._edit_failure(
                        E_STEP_MODE,
                        f"unknown step_mode {value!r}",
                        batch=batch,
                        edit_index=edit_index,
                        detail={"allowed": ["symmetric", "asymmetric"], "value": value},
                    )
                ]
            self.config = self.config.replace_step_mode(value)
            return []

        if not isinstance(seq_id, str):
            return [
                self._edit_failure(
                    E_SEQUENCE_NOT_FOUND,
                    "edit must identify an existing sequence by string id",
                    batch=batch,
                    edit_index=edit_index,
                )
            ]

        if edit_type == "sequence_replace":
            raw_record = edit.get("record")
            if isinstance(raw_record, dict):
                candidate = dict(raw_record)
                candidate.setdefault("id", seq_id)
            else:
                candidate = {
                    "id": seq_id,
                    "observations": edit.get("observations", edit.get("values")),
                    "timestamp_interval": edit.get("timestamp_interval", 1.0),
                    "segment_ids": edit.get("segment_ids", edit.get("segments")),
                }
                if candidate["segment_ids"] is None:
                    candidate.pop("segment_ids")
            checked, failures = validate_sequence_record(
                candidate, self.config, record_index=edit_index, stage="edit"
            )
            if failures:
                return [
                    Failure(
                        code=item.code,
                        stage=item.stage,
                        message=item.message,
                        sequence_id=seq_id,
                        batch=batch,
                        edit_index=edit_index,
                        record_index=item.record_index,
                        detail=item.detail,
                    )
                    for item in failures
                ]
            assert checked is not None
            self.library.records[seq_id] = checked
            self._clear_validation_failure(seq_id)
            return []

        record = self.library.records.get(seq_id)
        if record is None:
            return [
                self._edit_failure(
                    E_SEQUENCE_NOT_FOUND,
                    "sequence id is not present in the valid library",
                    batch=batch,
                    edit_index=edit_index,
                    sequence_id=seq_id,
                )
            ]

        if edit_type in {"point_update", "point_correction"}:
            index = edit.get("index")
            if not isinstance(index, int) or isinstance(index, bool):
                return [
                    self._edit_failure(
                        E_INDEX_RANGE,
                        "point update index must be an integer",
                        batch=batch,
                        edit_index=edit_index,
                        sequence_id=seq_id,
                    )
                ]
            if index < 0 or index >= len(record.observations):
                return [
                    self._edit_failure(
                        E_INDEX_RANGE,
                        "point update index is outside the sequence",
                        batch=batch,
                        edit_index=edit_index,
                        sequence_id=seq_id,
                        detail={"index": index, "length": len(record.observations)},
                    )
                ]
            checked_values, failures = self._validate_values(
                [edit.get("value")],
                batch=batch,
                edit_index=edit_index,
                sequence_id=seq_id,
            )
            if failures:
                return failures
            assert checked_values is not None
            observations = list(record.observations)
            observations[index] = checked_values[0]
            updated = SequenceRecord(
                seq_id=record.seq_id,
                observations=tuple(observations),
                timestamp_interval=record.timestamp_interval,
                segment_ids=record.segment_ids,
            )
            return self._install_record(updated, batch=batch, edit_index=edit_index)

        if edit_type == "sequence_append":
            checked_values, failures = self._validate_values(
                edit.get("observations", edit.get("values")),
                batch=batch,
                edit_index=edit_index,
                sequence_id=seq_id,
            )
            if failures:
                return failures
            assert checked_values is not None
            segment_ids_raw = edit.get("segment_ids")
            if segment_ids_raw is None and "segment_id" in edit:
                segment_ids_raw = [edit["segment_id"]] * len(checked_values)
            if segment_ids_raw is None:
                last_segment = record.segment_ids[-1] if record.segment_ids else 0
                segment_ids_raw = [last_segment] * len(checked_values)
            if not isinstance(segment_ids_raw, list) or len(segment_ids_raw) != len(
                checked_values
            ):
                return [
                    self._edit_failure(
                        E_SEGMENT_LENGTH,
                        "append segment_ids must match appended observations",
                        batch=batch,
                        edit_index=edit_index,
                        sequence_id=seq_id,
                    )
                ]
            updated = SequenceRecord(
                seq_id=record.seq_id,
                observations=record.observations + tuple(checked_values),
                timestamp_interval=edit.get(
                    "timestamp_interval", record.timestamp_interval
                ),
                segment_ids=record.segment_ids + tuple(segment_ids_raw),
            )
            return self._install_record(updated, batch=batch, edit_index=edit_index)

        return [
            self._edit_failure(
                E_EDIT_TYPE,
                f"unknown edit type {edit_type!r}",
                batch=batch,
                edit_index=edit_index,
                sequence_id=seq_id,
            )
        ]

    # ------------------------------------------------------------------
    # Work planning and execution
    # ------------------------------------------------------------------
    def _reference_for(self, query: SequenceRecord) -> SequenceRecord:
        reference_id = self.config.reference_sequence_id
        if isinstance(reference_id, str) and reference_id in self.library.records:
            return self.library.records[reference_id]
        return query

    def _new_working_artifact(
        self,
        query: SequenceRecord,
        reference: SequenceRecord,
        query_quantified: QuantizedSequence,
        reference_quantified: QuantizedSequence,
        band: Band,
        target_key: str,
        committed: Artifact | None,
    ) -> Artifact:
        plan = build_invalidation_plan(
            query.seq_id,
            committed.cells if committed is not None else None,
            committed.query_quantified if committed is not None else None,
            committed.reference_quantified if committed is not None else None,
            committed.band if committed is not None else None,
            committed.config if committed is not None else None,
            query_quantified,
            reference_quantified,
            band,
            self.config,
        )
        reused_cells: dict[tuple[int, int], CellProduct] = {}
        if committed is not None:
            invalid = set(plan.invalid_cells)
            new_coords = band_coordinates(band)
            for coord, cell in committed.cells.items():
                if coord in new_coords and coord not in invalid:
                    reused_cells[coord] = cell
        return Artifact(
            query=query,
            reference=reference,
            query_quantified=query_quantified,
            reference_quantified=reference_quantified,
            band=band,
            config=self.config,
            cells=reused_cells,
            target_key=target_key,
            complete=False,
            plan=plan,
            pending_cells=list(plan.invalid_cells),
            recomputed_total=0,
        )

    def _finalize_artifact(self, artifact: Artifact) -> Artifact:
        artifact.result = finalize_alignment(
            artifact.query.seq_id,
            artifact.query_quantified,
            artifact.reference_quantified,
            artifact.band,
            artifact.cells,
            artifact.config,
        )
        artifact.complete = True
        artifact.pending_cells = []
        return artifact

    def _compute_prefix(self, artifact: Artifact, count: int) -> Artifact:
        if count < 0 or count > len(artifact.pending_cells):
            raise ValueError("invalid cell allocation")
        coordinates = artifact.pending_cells[:count]
        for row, col in coordinates:
            cell = compute_cell(
                artifact.query.seq_id,
                row,
                col,
                artifact.query_quantified,
                artifact.reference_quantified,
                artifact.band,
                artifact.config,
                artifact.cells,
            )
            artifact.cells[(row, col)] = cell
        artifact.pending_cells = artifact.pending_cells[count:]
        artifact.recomputed_total += count
        if not artifact.pending_cells:
            self._finalize_artifact(artifact)
        return artifact

    def _prepare_work(
        self, *, batch: int, priority_ids: set[str]
    ) -> tuple[dict[str, dict[str, Any]], list[WorkJob]]:
        stats: dict[str, dict[str, Any]] = {}
        jobs: list[WorkJob] = []
        for seq_id, query in sorted(self.library.records.items()):
            reference = self._reference_for(query)
            query_quantified = quantize_sequence(query, self.config)
            reference_quantified = quantize_sequence(reference, self.config)
            band = build_band(
                len(query.observations),
                len(reference.observations),
                self.config.bandwidth_permille,
            )
            target_key = sequence_target_key(
                query,
                reference,
                query_quantified,
                reference_quantified,
                band,
                self.config,
            )
            slot = self.slots.setdefault(seq_id, Slot())
            committed = slot.committed
            if committed is not None and committed.complete and committed.target_key == target_key:
                slot.working = None
                stats[seq_id] = {
                    "sequence_id": seq_id,
                    "reference_id": reference.seq_id,
                    "status": "RESULT_REUSED",
                    "target_changed": False,
                    "result_reused": True,
                    "invalidated_cells": 0,
                    "removed_cells": 0,
                    "reused_cells": len(committed.cells),
                    "recomputed_cells": 0,
                    "pending_remaining": 0,
                    "result_status": committed.result.status if committed.result else None,
                    "sequence_task_key": task_key("align", seq_id),
                }
                self.reuse_ledger.append(
                    {
                        "batch": batch,
                        "sequence_id": seq_id,
                        "result_reused": True,
                        "reused_cells": len(committed.cells),
                        "recomputed_cells": 0,
                        "invalidated_cells": 0,
                    }
                )
                continue

            working = slot.working
            resumed = False
            if working is not None and not working.complete and working.target_key == target_key:
                resumed = True
            else:
                working = self._new_working_artifact(
                    query,
                    reference,
                    query_quantified,
                    reference_quantified,
                    band,
                    target_key,
                    committed,
                )
                slot.working = working

            stats[seq_id] = {
                "sequence_id": seq_id,
                "reference_id": reference.seq_id,
                "status": "PENDING" if working.pending_cells else "READY",
                "target_changed": True,
                "result_reused": False,
                "invalidated_cells": len(working.plan.invalid_cells)
                if working.plan
                else 0,
                "removed_cells": len(working.plan.removed_cells) if working.plan else 0,
                "reused_cells": working.plan.reused_count if working.plan else len(working.cells),
                "recomputed_cells": 0,
                "pending_remaining": len(working.pending_cells),
                "result_status": None,
                "sequence_task_key": task_key("align", seq_id),
                "resumed": resumed,
            }
            if working.plan is not None:
                self.invalidation_ledger.append(
                    {"batch": batch, **working.plan.to_dict(), "resumed": resumed}
                )

            if not working.pending_cells:
                self._finalize_artifact(working)
                slot.committed = working
                slot.working = None
                stats[seq_id]["status"] = "COMPLETE"
                stats[seq_id]["result_status"] = working.result.status if working.result else None
                self.reuse_ledger.append(
                    {
                        "batch": batch,
                        "sequence_id": seq_id,
                        "result_reused": False,
                        "reused_cells": len(working.cells),
                        "recomputed_cells": 0,
                        "invalidated_cells": 0,
                    }
                )
                continue

            jobs.append(
                WorkJob(
                    sequence_id=seq_id,
                    artifact=working,
                    allocate_cells=0,
                    priority=seq_id in priority_ids,
                    sequence_task_key=task_key("align", seq_id),
                )
            )
        return stats, jobs

    def _run_jobs(self, jobs: list[WorkJob]) -> list[tuple[str, Artifact, int]]:
        def execute(job: WorkJob) -> tuple[str, Artifact, int]:
            artifact = self._compute_prefix(job.artifact, job.allocate_cells)
            return job.sequence_id, artifact, job.allocate_cells

        if not jobs:
            return []
        workers = min(self.config.thread_pool_workers, len(jobs))
        if workers <= 1:
            return [execute(job) for job in jobs]
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="dtw-seq") as pool:
            futures = [(job, pool.submit(execute, job)) for job in jobs]
            return [future.result() for _job, future in futures]

    def run_batch(self, batch: int, edits: tuple[dict[str, Any], ...]) -> dict[str, Any]:
        priority_ids = {
            seq_id
            for seq_id, slot in self.slots.items()
            if slot.working is not None and not slot.working.complete
        }
        edit_failures: list[Failure] = []
        edits_applied = 0
        for edit_index, edit in enumerate(edits):
            failures = self.apply_edit(edit, batch=batch, edit_index=edit_index)
            if failures:
                edit_failures.extend(failures)
            else:
                edits_applied += 1

        stats, jobs = self._prepare_work(batch=batch, priority_ids=priority_ids)
        ordered_jobs = sorted(jobs, key=lambda job: (0 if job.priority else 1, job.sequence_id))

        budget_limit = self.config.cell_recompute_budget
        remaining_budget = budget_limit
        allocations: list[dict[str, Any]] = []
        blocked = False
        for job in ordered_jobs:
            need = len(job.artifact.pending_cells)
            if blocked:
                allocation = 0
            elif need <= remaining_budget:
                allocation = need
            else:
                allocation = remaining_budget
                blocked = True
            job.allocate_cells = allocation
            remaining_budget -= allocation
            allocations.append(
                {
                    "sequence_id": job.sequence_id,
                    "requested_cells": need,
                    "allocated_cells": allocation,
                    "priority": job.priority,
                    "sequence_task_key": job.sequence_task_key,
                }
            )

        executable = [job for job in ordered_jobs if job.allocate_cells > 0]
        results = self._run_jobs(executable)
        result_map = {
            seq_id: (artifact, allocation)
            for seq_id, artifact, allocation in results
        }

        for job in ordered_jobs:
            seq_id = job.sequence_id
            artifact, allocation = result_map.get(seq_id, (job.artifact, 0))
            stat = stats[seq_id]
            stat["recomputed_cells"] = allocation
            stat["pending_remaining"] = len(artifact.pending_cells)
            slot = self.slots[seq_id]
            if artifact.complete:
                slot.committed = artifact
                slot.working = None
                stat["status"] = "COMPLETE"
                stat["result_status"] = artifact.result.status if artifact.result else None
            else:
                slot.working = artifact
                stat["status"] = "PENDING"
            self.reuse_ledger.append(
                {
                    "batch": batch,
                    "sequence_id": seq_id,
                    "result_reused": False,
                    "reused_cells": stat["reused_cells"],
                    "recomputed_cells": allocation,
                    "invalidated_cells": stat["invalidated_cells"],
                    "pending_remaining": stat["pending_remaining"],
                }
            )

        consumed = sum(item["allocated_cells"] for item in allocations)
        pending_ids = sorted(
            seq_id
            for seq_id, slot in self.slots.items()
            if slot.working is not None and not slot.working.complete
        )
        pending_cell_count = 0
        for seq_id in pending_ids:
            working = self.slots[seq_id].working
            assert working is not None
            pending_cell_count += len(working.pending_cells)

        batch_failures = list(edit_failures)
        for seq_id in pending_ids:
            working = self.slots[seq_id].working
            assert working is not None
            batch_failures.append(
                Failure(
                    E_CELL_BUDGET_PENDING,
                    "scheduler",
                    "cell recomputation budget exhausted; sequence retained for next batch",
                    sequence_id=seq_id,
                    batch=batch,
                    detail={
                        "pending_cells": len(working.pending_cells),
                        "budget_limit": budget_limit,
                    },
                )
            )

        completed_results = []
        for seq_id in sorted(stats):
            slot = self.slots.get(seq_id)
            if slot and slot.committed and slot.committed.result:
                result = slot.committed.result
                completed_results.append(
                    {
                        "sequence_id": seq_id,
                        "status": result.status,
                        "failure_code": result.failure_code,
                        "total_cost": result.total_cost,
                        "path_length": result.path_length,
                    }
                )

        budget_entry = {
            "batch": batch,
            "frozen_limit": budget_limit,
            "consumed": consumed,
            "remaining": remaining_budget,
            "pending_sequence_count": len(pending_ids),
            "pending_cell_count": pending_cell_count,
            "priority_sequences": sorted(priority_ids),
            "allocations": allocations,
            "blocked": blocked,
        }
        self.budget_ledger.append(budget_entry)
        self.edit_failure_ledger.extend(item.to_dict() for item in edit_failures)

        batch_record = {
            "batch": batch,
            "edits_applied": edits_applied,
            "edit_count": len(edits),
            "edit_failures": [item.to_dict() for item in edit_failures],
            "budget": budget_entry,
            "sequences": [stats[seq_id] for seq_id in sorted(stats)],
            "pending_queue": pending_ids,
            "completed_results": completed_results,
            "failures": [item.to_dict() for item in batch_failures],
            "priority_sequences": sorted(priority_ids),
            "thread_pool_workers_limit": self.config.thread_pool_workers,
        }
        self.batch_ledger.append(batch_record)
        self.last_batch = max(self.last_batch, batch)

        archive = batch % self.config.snapshot_interval_batches == 0
        snapshot_paths = self.persist(archive=archive)
        self._atomic_json(
            self.batch_report_dir / f"batch_{batch:06d}.json",
            batch_record,
            pretty=True,
        )
        batch_record["snapshot_paths"] = snapshot_paths
        return batch_record

    # ------------------------------------------------------------------
    # Outputs and full rebuild reference
    # ------------------------------------------------------------------
    def active_failures(self) -> list[dict[str, Any]]:
        failures: list[Failure] = list(self.validation_failures)
        for seq_id, slot in sorted(self.slots.items()):
            if slot.committed and slot.committed.result:
                result = slot.committed.result
                if result.failure_code:
                    failures.append(
                        Failure(
                            result.failure_code,
                            "alignment",
                            result.failure_message or result.failure_code,
                            sequence_id=seq_id,
                        )
                    )
            if slot.working is not None and not slot.working.complete:
                failures.append(
                    Failure(
                        E_CELL_BUDGET_PENDING,
                        "scheduler",
                        "sequence still has pending cell recomputation",
                        sequence_id=seq_id,
                        batch=self.last_batch,
                        detail={"pending_cells": len(slot.working.pending_cells)},
                    )
                )
        return sorted_failures(failures)

    def export_result(self) -> dict[str, Any]:
        sequences: list[dict[str, Any]] = []
        pending: list[dict[str, Any]] = []
        for seq_id in sorted(self.library.records):
            slot = self.slots.get(seq_id)
            if slot and slot.committed and slot.committed.result:
                sequences.append(slot.committed.result.to_dict())
            else:
                working = slot.working if slot else None
                pending.append(
                    {
                        "sequence_id": seq_id,
                        "pending_cells": len(working.pending_cells) if working else None,
                        "status": "PENDING",
                    }
                )
        return {
            "sequences": sequences,
            "failures": self.active_failures(),
            "pending_sequences": pending,
        }

    def run_plan(
        self, batches: list[EditBatch], *, auto_drain: bool = True
    ) -> list[dict[str, Any]]:
        reports: list[dict[str, Any]] = []
        for item in batches:
            reports.append(self.run_batch(item.batch, item.edits))
        if auto_drain:
            guard = 0
            next_batch = max(
                self.last_batch,
                max((item.batch for item in batches), default=0),
            ) + 1
            while any(
                slot.working is not None and not slot.working.complete
                for slot in self.slots.values()
            ):
                if self.config.cell_recompute_budget <= 0:
                    break
                reports.append(self.run_batch(next_batch, ()))
                next_batch += 1
                guard += 1
                if guard > 10000:
                    raise RuntimeError("auto-drain did not converge")
        return reports

    def sequence_stats(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for seq_id in sorted(self.library.records):
            slot = self.slots.get(seq_id)
            committed = slot.committed if slot else None
            working = slot.working if slot else None
            reuse_rows = [item for item in self.reuse_ledger if item["sequence_id"] == seq_id]
            rows.append(
                {
                    "sequence_id": seq_id,
                    "committed": bool(committed and committed.complete),
                    "pending_cells": len(working.pending_cells) if working else 0,
                    "reuse_batches": sum(
                        1 for item in reuse_rows if item.get("result_reused")
                    ),
                    "recomputed_cells": sum(
                        item.get("recomputed_cells", 0) for item in reuse_rows
                    ),
                    "reused_cells": sum(
                        item.get("reused_cells", 0)
                        for item in reuse_rows
                        if not item.get("result_reused")
                    ),
                    "result_status": committed.result.status
                    if committed and committed.result
                    else None,
                }
            )
        return rows


def band_coordinates(band: Band) -> set[tuple[int, int]]:
    return set(feasible_coordinates(band))


def _resolved_reference(query: SequenceRecord, config: Config, library: LibraryData) -> SequenceRecord:
    reference_id = config.reference_sequence_id
    if isinstance(reference_id, str) and reference_id in library.records:
        return library.records[reference_id]
    return query


def _full_sequence_result(
    query: SequenceRecord, reference: SequenceRecord, config: Config
) -> AlignmentResult:
    query_quantified = quantize_sequence(query, config)
    reference_quantified = quantize_sequence(reference, config)
    band = build_band(
        len(query.observations), len(reference.observations), config.bandwidth_permille
    )
    cells: dict[tuple[int, int], CellProduct] = {}
    for row, col in feasible_coordinates(band):
        cells[(row, col)] = compute_cell(
            query.seq_id,
            row,
            col,
            query_quantified,
            reference_quantified,
            band,
            config,
            cells,
        )
    return finalize_alignment(
        query.seq_id, query_quantified, reference_quantified, band, cells, config
    )


def full_rebuild(
    config: Config,
    library: LibraryData,
    validation_failures: list[Failure] | None = None,
) -> dict[str, Any]:
    """Reference build from an empty cache, parallelized by sequence id."""

    records = library.ordered_records()

    def run(record: SequenceRecord) -> tuple[str, AlignmentResult]:
        reference = _resolved_reference(record, config, library)
        return record.seq_id, _full_sequence_result(record, reference, config)

    if records:
        workers = min(config.thread_pool_workers, len(records))
        if workers <= 1:
            computed = [run(record) for record in records]
        else:
            with ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="dtw-full"
            ) as pool:
                computed = list(pool.map(run, records))
        results = dict(computed)
    else:
        results = {}

    failures: list[Failure] = list(validation_failures or [])
    for seq_id in sorted(results):
        result = results[seq_id]
        if result.failure_code:
            failures.append(
                Failure(
                    result.failure_code,
                    "alignment",
                    result.failure_message or result.failure_code,
                    sequence_id=seq_id,
                )
            )
    return {
        "sequences": [results[seq_id].to_dict() for seq_id in sorted(results)],
        "failures": sorted_failures(failures),
        "pending_sequences": [],
    }


def canonical_equivalence(
    left: dict[str, Any], right: dict[str, Any]
) -> dict[str, Any]:
    fields = ("sequences", "failures", "pending_sequences")
    differing_fields = [field for field in fields if left.get(field) != right.get(field)]
    return {
        "equal": not differing_fields,
        "compared_fields": list(fields),
        "differing_fields": differing_fields,
        "left_fingerprint": fingerprint({field: left.get(field) for field in fields}),
        "right_fingerprint": fingerprint({field: right.get(field) for field in fields}),
    }


def run_plan_reloading_state(
    engine: DiagnosticEngine,
    batches: list[EditBatch],
    *,
    workdir: str | Path,
    auto_drain: bool = True,
) -> tuple[DiagnosticEngine, list[dict[str, Any]]]:
    """Run a plan while reloading the latest snapshot after every batch.

    This intentionally round-trips through disk so unchanged cell products are
    proven to come from persisted artifacts rather than an in-memory shortcut.
    """

    reports: list[dict[str, Any]] = []
    current = engine
    for item in batches:
        reports.append(current.run_batch(item.batch, item.edits))
        current = DiagnosticEngine.load_latest(workdir=workdir)
    if auto_drain:
        next_batch = max(
            current.last_batch,
            max((item.batch for item in batches), default=0),
        ) + 1
        guard = 0
        while any(
            slot.working is not None and not slot.working.complete
            for slot in current.slots.values()
        ):
            if current.config.cell_recompute_budget <= 0:
                break
            reports.append(current.run_batch(next_batch, ()))
            current = DiagnosticEngine.load_latest(workdir=workdir)
            next_batch += 1
            guard += 1
            if guard > 10000:
                raise RuntimeError("state-reloading auto-drain did not converge")
    return current, reports
