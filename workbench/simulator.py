"""Deterministic batch-barrier replica replay simulation.

The replay rate is aggregate records/second. Records are completed in batches
whose maximum size is the configured concurrency. A batch of k records whose
global end position is e finishes at base + ceil(e * 1e9 / rate) nanoseconds.
This makes completion timestamps independent of resume boundaries and process
lifetime, while concurrency changes the observable completion clusters.
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import ParameterError
from .models import MasterRecord, ReplayParams
from .util import ceil_div, fixed_from_ns


@dataclass(frozen=True)
class PlannedBatch:
    batch_index: int
    first_seq: int
    last_seq: int
    records: tuple[MasterRecord, ...]
    completed_mono_ns: int


@dataclass(frozen=True)
class PlannedSample:
    commit_seq: int
    replay_completed_ns: int
    delay_ns: int
    fixed_delay: int


def build_replay_plan(
    records: list[MasterRecord], params: ReplayParams, scale_power: int
) -> list[PlannedBatch]:
    if not records:
        raise ParameterError("没有可回放的主库日志", field="records")
    if params.rate_rps <= 0:
        raise ParameterError("rate_rps 必须为正整数", field="replay.rate_rps")
    if params.concurrency <= 0:
        raise ParameterError("concurrency 必须为正整数", field="replay.concurrency")
    if params.clock_base_ms < 0:
        raise ParameterError("clock_base_ms 必须非负", field="replay.clock_base_ms")

    base_ns = params.clock_base_ms * 1_000_000
    batches: list[PlannedBatch] = []
    for start in range(0, len(records), params.concurrency):
        batch_records = records[start : start + params.concurrency]
        end_position = start + len(batch_records)
        completion_offset_ns = ceil_div(end_position * 1_000_000_000, params.rate_rps)
        completed_ns = base_ns + completion_offset_ns
        batch_index = start // params.concurrency
        batches.append(
            PlannedBatch(
                batch_index=batch_index,
                first_seq=batch_records[0].commit_seq,
                last_seq=batch_records[-1].commit_seq,
                records=tuple(batch_records),
                completed_mono_ns=completed_ns,
            )
        )
    return batches


def materialize_samples(
    batch: PlannedBatch, param_version: str, batch_key: int, scale_power: int
) -> list[PlannedSample]:
    result: list[PlannedSample] = []
    for record in batch.records:
        commit_ns = record.commit_ts_ms * 1_000_000
        delay_ns = batch.completed_mono_ns - commit_ns
        if delay_ns < 0:
            raise ParameterError(
                "主库提交时间晚于模拟回放完成时刻，无法形成非负延迟",
                seq=record.commit_seq,
                field="commit_ts_ms",
                details={
                    "commit_ts_ms": record.commit_ts_ms,
                    "completed_mono_ns": batch.completed_mono_ns,
                    "clock_base_ms": batch.completed_mono_ns // 1_000_000,
                },
            )
        result.append(
            PlannedSample(
                commit_seq=record.commit_seq,
                replay_completed_ns=batch.completed_mono_ns,
                delay_ns=delay_ns,
                fixed_delay=fixed_from_ns(delay_ns, scale_power),
            )
        )
    return result


def batch_quantum_ns(rate_rps: int, batch_size: int) -> int:
    if rate_rps <= 0:
        raise ValueError("rate_rps must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return ceil_div(batch_size * 1_000_000_000, rate_rps)
