"""Deterministic streaming execution engine.

The compiled plan is a single ordered stage chain.  Every stage consumes the
upstream snapshot as bounded blocks (block size comes from the read stage
``chunk_rows`` session parameter), transforms cells using exact integer
fixed-point arithmetic, and materializes a canonical JSON-lines snapshot plus
per-column statistics.  No wall-clock data, host float arithmetic or set
iteration affects outputs.
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Iterator, Optional

from .canonical import canonical_bytes, chained_line_hash, combined_hash
from .errors import WorkbenchError
from .fixedpoint import scale_int, to_fixed
from .spec import PipelineSpec, SourceSpec, StageSpec

COLUMN_ORDER: list[str] | None = None  # set per run from read stage


@dataclass
class StageObservation:
    ordinal: int
    stage_id: str
    stage_type: str
    version: int
    stage_fingerprint: str
    upstream: list[str]
    params: dict[str, Any]
    status: str
    rows_in: int
    rows_out: int
    output_digest: str
    column_stats: dict[str, dict[str, Any]]
    extra: dict[str, Any]
    snapshot_path: str


@dataclass
class ShardObservation:
    stage_id: str
    shard_index: int
    row_start: int
    row_end: int
    row_count: int
    content_digest: str
    shard_fingerprint: str
    source_stage_digest: str
    target_rows: int
    remainder_rule: str
    file_path: str


def _canonical_line(row: dict[str, Any]) -> bytes:
    ordered = {column: row[column] for column in sorted(row)}
    return canonical_bytes(ordered) + b"\n"


def _iter_snapshot(path: str, block_rows: int
                   ) -> Iterator[list[dict[str, Any]]]:
    block: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            block.append(json.loads(line))
            if len(block) >= block_rows:
                yield block
                block = []
    if block:
        yield block


def _blank_numeric_stats() -> dict[str, Any]:
    return {"count": 0, "missing_count": 0, "present_count": 0,
            "zero_count": 0, "sum_fixed": 0, "min_fixed": None,
            "max_fixed": None}


def _accumulate_fixed(stats: dict[str, Any], value: Optional[int]) -> None:
    stats["count"] += 1
    if value is None:
        stats["missing_count"] += 1
        return
    stats["present_count"] += 1
    stats["sum_fixed"] += value
    if value == 0:
        stats["zero_count"] += 1
    if stats["min_fixed"] is None or value < stats["min_fixed"]:
        stats["min_fixed"] = value
    if stats["max_fixed"] is None or value > stats["max_fixed"]:
        stats["max_fixed"] = value


def _finalize_fixed(columns: list[str],
                    partial: dict[str, dict[str, Any]]
                    ) -> dict[str, dict[str, Any]]:
    return {column: dict(sorted(partial[column].items()))
            for column in columns}


class _SnapshotWriter:
    """Writes canonical JSON-lines while accumulating digest and stats."""

    def __init__(self, path: str, columns: list[str],
                 raw_mode: bool = False) -> None:
        self.path = path
        self.columns = columns
        self.raw_mode = raw_mode
        self.tmp_path = path + ".tmp"
        self.handle = open(self.tmp_path, "w", encoding="utf-8", newline="\n")
        self._lines: list[bytes] = []
        self.rows_out = 0
        if raw_mode:
            self.raw_stats = {
                column: {"count": 0, "missing_count": 0, "present_count": 0,
                         "zero_count": 0, "min_raw": None, "max_raw": None}
                for column in columns}
        else:
            self.fixed_stats = {column: _blank_numeric_stats()
                                for column in columns}

    def write(self, row: dict[str, Any]) -> None:
        line = _canonical_line(row)
        self.handle.write(line.decode("utf-8"))
        self._lines.append(line.rstrip(b"\n"))
        self.rows_out += 1
        for column in self.columns:
            value = row[column]
            if self.raw_mode:
                stat = self.raw_stats[column]
                stat["count"] += 1
                if value is None:
                    stat["missing_count"] += 1
                    continue
                stat["present_count"] += 1
                numeric = Decimal(str(value))
                if numeric == 0:
                    stat["zero_count"] += 1
                text = format(numeric, "f")
                if stat["min_raw"] is None or Decimal(stat["min_raw"]) > numeric:
                    stat["min_raw"] = text
                if stat["max_raw"] is None or Decimal(stat["max_raw"]) < numeric:
                    stat["max_raw"] = text
            else:
                _accumulate_fixed(self.fixed_stats[column], value)

    def close(self) -> tuple[str, dict[str, dict[str, Any]]]:
        self.handle.close()
        os.replace(self.tmp_path, self.path)
        digest = chained_line_hash("stage-output-v1", iter(self._lines))
        if self.raw_mode:
            stats = {column: dict(sorted(stat.items()))
                     for column, stat in self.raw_stats.items()}
        else:
            stats = _finalize_fixed(self.columns, self.fixed_stats)
        return digest, stats

    def abort(self) -> None:
        """Release the handle and drop an uncommitted temp snapshot."""
        if not self.handle.closed:
            self.handle.close()
        if os.path.exists(self.tmp_path):
            os.remove(self.tmp_path)


def _read_stage(spec: StageSpec, sources: dict[str, SourceSpec],
                columns: list[str], writer: _SnapshotWriter) -> dict[str, Any]:
    selected = [sources[source_id]
                for source_id in spec.params["source_ids"]]
    source_row_counts: dict[str, int] = {}
    for source in selected:
        source_row_counts[source.id] = source.row_count
        for row in source.rows:
            writer.write({column: _raw_cell(row.get(column))
                          for column in columns})
    return {"source_ids": [source.id for source in selected],
            "source_row_counts": dict(sorted(source_row_counts.items()))}


def _raw_cell(value: Any) -> Any:
    if isinstance(value, float):
        return format(Decimal(str(value)), "f")
    return value


def _types_stage(spec: StageSpec, upstream_path: str, block_rows: int,
                 columns: list[str], writer: _SnapshotWriter) -> dict[str, Any]:
    scale = spec.params["fixed_scale"]
    rounding = spec.params["rounding"]
    for block in _iter_snapshot(upstream_path, block_rows):
        for row in block:
            converted = {
                column: None if row[column] is None
                else to_fixed(row[column], scale, rounding)
                for column in columns}
            writer.write(converted)
    return {"fixed_scale": scale, "rounding": rounding}


def _fill_stage(spec: StageSpec, upstream_path: str, block_rows: int,
                columns: list[str], writer: _SnapshotWriter) -> dict[str, Any]:
    fill_value = spec.params["fill_value"]
    filled_count = 0
    for block in _iter_snapshot(upstream_path, block_rows):
        for row in block:
            converted = dict(row)
            for column in columns:
                if converted[column] is None:
                    converted[column] = fill_value
                    filled_count += 1
            writer.write(converted)
    return {"fill_value": fill_value, "filled_count": filled_count}


def _scale_stage(spec: StageSpec, upstream_path: str, block_rows: int,
                 columns: list[str], writer: _SnapshotWriter) -> dict[str, Any]:
    decimal_scale = spec.params["scale"]
    divisor = spec.params["divisor"]
    rounding = spec.params["rounding"]
    for block in _iter_snapshot(upstream_path, block_rows):
        for row in block:
            writer.write({
                column: scale_int(row[column], decimal_scale, divisor,
                                  rounding)
                for column in columns})
    return {"scale": decimal_scale, "divisor": divisor,
            "rounding": rounding}


def _clip_stage(spec: StageSpec, upstream_path: str, block_rows: int,
                columns: list[str], writer: _SnapshotWriter
                ) -> dict[str, Any]:
    lower = spec.params["lower"]
    upper = spec.params["upper"]
    max_bp = spec.params["max_outlier_bp"]
    clipped_low = clipped_high = outlier_count = total_cells = 0
    for block in _iter_snapshot(upstream_path, block_rows):
        for row in block:
            converted = dict(row)
            for column in columns:
                value = converted[column]
                total_cells += 1
                if value < lower:
                    converted[column] = lower
                    clipped_low += 1
                    outlier_count += 1
                elif value > upper:
                    converted[column] = upper
                    clipped_high += 1
                    outlier_count += 1
            writer.write(converted)
    # Basis points use floor division: bp = floor(10000 * outliers / cells).
    outlier_bp = (10_000 * outlier_count // total_cells) if total_cells else 0
    extra = {"lower": lower, "upper": upper, "max_outlier_bp": max_bp,
             "clipped_low": clipped_low, "clipped_high": clipped_high,
             "outlier_count": outlier_count, "outlier_bp": outlier_bp}
    if outlier_bp > max_bp:
        raise WorkbenchError(
            "E_OUTLIER",
            f"outlier ratio {outlier_bp}bp exceeds threshold {max_bp}bp",
            spec.id, "$.params.max_outlier_bp",
            {"param": "max_outlier_bp", "outlier_bp": outlier_bp,
             "max_outlier_bp": max_bp, "outlier_count": outlier_count,
             "total_cells": total_cells, **extra})
    return extra


def _shard_stage(spec: StageSpec, upstream_path: str, block_rows: int,
                 columns: list[str], stage_dir: str, shard_dir: str,
                 source_digest: str, pack_path: str
                 ) -> tuple[dict[str, Any], list[ShardObservation], str]:
    target_rows = spec.params["target_rows"]
    remainder_rule = spec.params["remainder_rule"]
    # Full-chain output snapshot is identical to upstream; stream-copy keeps
    # the digest equal to the source stage output digest (binding anchor).
    pack_writer = _SnapshotWriter(pack_path, columns)
    shard_records: list[ShardObservation] = []
    shard_rows: list[dict[str, Any]] = []
    row_start = 0
    global_index = 0
    dropped_rows = 0
    shard_index = 0
    for block in _iter_snapshot(upstream_path, block_rows):
        for row in block:
            pack_writer.write(row)
            shard_rows.append(row)
            global_index += 1
            if len(shard_rows) == target_rows:
                record = _write_shard(shard_dir, spec, shard_index, row_start,
                                      shard_rows, source_digest)
                shard_records.append(record)
                shard_index += 1
                shard_rows = []
                row_start = global_index
    if shard_rows:
        if remainder_rule == "drop_partial":
            dropped_rows = len(shard_rows)
        else:  # keep_partial: explicit remainder shard
            record = _write_shard(shard_dir, spec, shard_index, row_start,
                                  shard_rows, source_digest)
            shard_records.append(record)
    pack_digest, pack_stats = pack_writer.close()
    extra = {"target_rows": target_rows,
             "remainder_rule": remainder_rule,
             "shard_count": len(shard_records),
             "shard_row_counts": [record.row_count for record in shard_records],
             "dropped_rows": dropped_rows,
             "source_stage_digest": source_digest}
    return extra, shard_records, pack_digest


def _write_shard(shard_dir: str, spec: StageSpec, shard_index: int,
                 row_start: int, rows: list[dict[str, Any]],
                 source_digest: str) -> ShardObservation:
    file_name = f"pack_{shard_index:04d}.jsonl"
    path = os.path.join(shard_dir, file_name)
    lines: list[bytes] = []
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            line = _canonical_line(row)
            handle.write(line.decode("utf-8"))
            lines.append(line.rstrip(b"\n"))
    content_digest = chained_line_hash("shard-content-v1", iter(lines))
    shard_fingerprint = combined_hash("shard-fingerprint-v1", (
        ("stage_id", spec.id),
        ("shard_index", shard_index),
        ("row_start", row_start),
        ("row_end", row_start + len(rows)),
        ("target_rows", spec.params["target_rows"]),
        ("remainder_rule", spec.params["remainder_rule"]),
        ("source_stage_digest", source_digest),
        ("content_digest", content_digest),
    ))
    return ShardObservation(
        stage_id=spec.id, shard_index=shard_index, row_start=row_start,
        row_end=row_start + len(rows), row_count=len(rows),
        content_digest=content_digest, shard_fingerprint=shard_fingerprint,
        source_stage_digest=source_digest,
        target_rows=spec.params["target_rows"],
        remainder_rule=spec.params["remainder_rule"], file_path=path)


class ExecutionPlan:
    """Compiled deterministic plan: ordered chain plus shared block size."""

    def __init__(self, pipeline: PipelineSpec) -> None:
        self.pipeline = pipeline
        self.stages = list(pipeline.stages)
        self.sources = {source.id: source for source in pipeline.sources}
        read_stage = self.stages[0]
        if read_stage.type != "read":
            raise WorkbenchError("E_SPEC", "first stage must be read",
                                 field_path="$.stages[0]")
        self.block_rows = read_stage.params["chunk_rows"]
        declared = read_stage.params["source_ids"]
        self.columns = self._column_order(declared)

    def _column_order(self, source_ids: list[str]) -> list[str]:
        ordered: list[str] = []
        seen: set[str] = set()
        for source_id in source_ids:
            for column in self.sources[source_id].columns:
                if column not in seen:
                    seen.add(column)
                    ordered.append(column)
        return ordered


class Engine:
    """Executes a compiled plan into a fresh run directory."""

    STAGE_FILES = ("01_read", "02_types", "03_fill", "04_scale",
                   "05_clip", "06_pack")

    def __init__(self, plan: ExecutionPlan, run_dir: str) -> None:
        self.plan = plan
        self.run_dir = run_dir
        self.stage_dir = os.path.join(run_dir, "stages")
        self.shard_dir = os.path.join(run_dir, "shards")
        self.work_dir = os.path.join(run_dir, "work")
        for directory in (self.stage_dir, self.shard_dir, self.work_dir):
            os.makedirs(directory, exist_ok=True)
        self.columns = plan.columns
        self.observations: list[StageObservation] = []
        self.shards: list[ShardObservation] = []
        self.upstream_path: Optional[str] = None
        self.rows_in = 0

    def execute(self, kill_after_ordinal: Optional[int] = None
                ) -> list[StageObservation]:
        for ordinal, stage in enumerate(self.plan.stages):
            if kill_after_ordinal is not None and ordinal > kill_after_ordinal:
                break
            self._run_stage(ordinal, stage)
        return self.observations

    def _stage_path(self, ordinal: int) -> str:
        label = self.STAGE_FILES[ordinal] if ordinal < len(
            self.STAGE_FILES) else f"{ordinal + 1:02d}_{self.plan.stages[ordinal].id}"
        return os.path.join(self.stage_dir, label + ".jsonl")

    def _run_stage(self, ordinal: int, stage: StageSpec) -> None:
        output_path = self._stage_path(ordinal)
        rows_in = 0 if self.upstream_path is None else self._count_rows(
            self.upstream_path)
        writer: Optional[_SnapshotWriter] = None
        try:
            if stage.type == "read":
                writer = _SnapshotWriter(output_path, self.columns,
                                         raw_mode=True)
                extra = _read_stage(stage, self.plan.sources, self.columns,
                                    writer)
                digest, stats = writer.close()
            elif stage.type == "normalize_types":
                writer = _SnapshotWriter(output_path, self.columns)
                extra = _types_stage(stage, self.upstream_path,
                                     self.plan.block_rows, self.columns,
                                     writer)
                digest, stats = writer.close()
            elif stage.type == "fill_missing":
                writer = _SnapshotWriter(output_path, self.columns)
                extra = _fill_stage(stage, self.upstream_path,
                                    self.plan.block_rows, self.columns,
                                    writer)
                digest, stats = writer.close()
            elif stage.type == "normalize":
                writer = _SnapshotWriter(output_path, self.columns)
                extra = _scale_stage(stage, self.upstream_path,
                                     self.plan.block_rows, self.columns,
                                     writer)
                digest, stats = writer.close()
            elif stage.type == "clip":
                # Intermediate pre-bind stream lives under work/ so a forced
                # termination never leaves a partially committed snapshot.
                work_path = os.path.join(self.work_dir,
                                         f"{ordinal + 1:02d}_{stage.id}",
                                         "stage.jsonl")
                os.makedirs(os.path.dirname(work_path), exist_ok=True)
                writer = _SnapshotWriter(work_path, self.columns)
                extra = _clip_stage(stage, self.upstream_path,
                                    self.plan.block_rows, self.columns,
                                    writer)
                digest, stats = writer.close()
                os.replace(work_path, output_path)
            elif stage.type == "shard":
                extra, self.shards, digest = _shard_stage(
                    stage, self.upstream_path, self.plan.block_rows,
                    self.columns, self.stage_dir, self.shard_dir,
                    self._upstream_digest(), output_path)
                stats = self._stats_from_pack(output_path)
            else:  # pragma: no cover - guarded by spec validation
                raise WorkbenchError(
                    "E_SPEC", f"unknown stage type {stage.type}", stage.id,
                    "$.stages[*].type")
        except WorkbenchError:
            if writer is not None:
                writer.abort()
            raise
        rows_out = self._count_rows(output_path)
        observation = StageObservation(
            ordinal=ordinal, stage_id=stage.id, stage_type=stage.type,
            version=stage.version,
            stage_fingerprint=self.plan.pipeline.stage_fingerprints[
                stage.id],
            upstream=list(stage.upstream), params=dict(stage.params),
            status="COMMITTED", rows_in=rows_in, rows_out=rows_out,
            output_digest=digest, column_stats=stats, extra=extra,
            snapshot_path=output_path)
        self.observations.append(observation)
        self.upstream_path = output_path
        self.rows_in = rows_out

    def _upstream_digest(self) -> str:
        return self.observations[-1].output_digest

    def _stats_from_pack(self, path: str) -> dict[str, dict[str, Any]]:
        writer = _SnapshotWriter(path + ".stat", self.columns)
        for block in _iter_snapshot(path, self.plan.block_rows):
            for row in block:
                writer.write(row)
        _, stats = writer.close()
        os.remove(path + ".stat")
        return stats

    @staticmethod
    def _count_rows(path: str) -> int:
        with open(path, "r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
