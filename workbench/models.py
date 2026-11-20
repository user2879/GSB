"""Typed domain records shared by persistence, simulation and reporting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .constants import ESTIMATOR_VERSION, P95_METHOD, REPLAY_MODEL_VERSION, TRUNCATION_DIRECTION
from .util import short_digest


@dataclass(frozen=True)
class MasterRecord:
    commit_seq: int
    txn_id: str
    commit_ts_ms: int
    txn_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "commit_seq": self.commit_seq,
            "txn_id": self.txn_id,
            "commit_ts_ms": self.commit_ts_ms,
            "txn_bytes": self.txn_bytes,
        }


@dataclass(frozen=True)
class ReplayParams:
    rate_rps: int
    concurrency: int
    clock_mode: str
    clock_base_ms: int
    estimator_digest: str

    @property
    def version(self) -> str:
        return short_digest(self.binding_dict(), 24)

    def scheduling_dict(self) -> dict[str, Any]:
        return {
            "model_version": REPLAY_MODEL_VERSION,
            "rate_rps": self.rate_rps,
            "concurrency": self.concurrency,
            "clock_mode": self.clock_mode,
            "clock_base_ms": self.clock_base_ms,
        }

    def binding_dict(self) -> dict[str, Any]:
        value = self.scheduling_dict()
        value["estimator_digest"] = self.estimator_digest
        return value

    def to_dict(self) -> dict[str, Any]:
        value = self.scheduling_dict()
        value.update(
            {
                "version": self.version,
                "estimator_digest": self.estimator_digest,
            }
        )
        return value


@dataclass(frozen=True)
class AggregationSpec:
    window_size: int
    max_samples_per_window: int
    min_samples: int
    latency_ceiling_ms: int
    scale_power: int
    p95_method: str = P95_METHOD
    truncation_direction: str = TRUNCATION_DIRECTION
    dominant_threshold_pct: int = 60
    version: str = ESTIMATOR_VERSION

    @property
    def scale(self) -> int:
        return 10**self.scale_power

    @property
    def digest(self) -> str:
        return short_digest(self.to_core_dict(), 24)

    def to_core_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "window_size": self.window_size,
            "max_samples_per_window": self.max_samples_per_window,
            "min_samples": self.min_samples,
            "latency_ceiling_ms": self.latency_ceiling_ms,
            "scale_power": self.scale_power,
            "p95_method": self.p95_method,
            "truncation_direction": self.truncation_direction,
            "dominant_threshold_pct": self.dominant_threshold_pct,
        }

    def to_dict(self) -> dict[str, Any]:
        value = self.to_core_dict()
        value["spec_digest"] = self.digest
        return value


@dataclass(frozen=True)
class Sample:
    param_version: str
    commit_seq: int
    batch_id: int
    replay_completed_ns: int
    delay_ns: int
    fixed_delay: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "param_version": self.param_version,
            "commit_seq": self.commit_seq,
            "batch_id": self.batch_id,
            "replay_completed_ns": self.replay_completed_ns,
            "delay_ns": self.delay_ns,
            "fixed_delay": self.fixed_delay,
        }


@dataclass(frozen=True)
class Progress:
    param_version: str
    last_replayed_seq: int
    last_completed_batch_key: int | None
    completed_batches: int
    status: str
    updated_utc: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "param_version": self.param_version,
            "last_replayed_seq": self.last_replayed_seq,
            "last_completed_batch_key": self.last_completed_batch_key,
            "completed_batches": self.completed_batches,
            "status": self.status,
            "updated_utc": self.updated_utc,
        }


@dataclass(frozen=True)
class ReplayBatch:
    batch_key: int
    param_version: str
    batch_index: int
    attempt: int
    first_seq: int
    last_seq: int
    status: str
    started_mono_ns: int
    completed_mono_ns: int | None
    started_utc: str
    completed_utc: str | None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_key": self.batch_key,
            "param_version": self.param_version,
            "batch_index": self.batch_index,
            "attempt": self.attempt,
            "first_seq": self.first_seq,
            "last_seq": self.last_seq,
            "status": self.status,
            "started_mono_ns": self.started_mono_ns,
            "completed_mono_ns": self.completed_mono_ns,
            "started_utc": self.started_utc,
            "completed_utc": self.completed_utc,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class ReplayRun:
    param_version: str
    rate_rps: int
    concurrency: int
    clock_mode: str
    clock_base_ms: int
    estimator_digest: str
    estimator_json: str
    params_json: str
    created_utc: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "param_version": self.param_version,
            "rate_rps": self.rate_rps,
            "concurrency": self.concurrency,
            "clock_mode": self.clock_mode,
            "clock_base_ms": self.clock_base_ms,
            "estimator_digest": self.estimator_digest,
            "estimator_json": self.estimator_json,
            "params_json": self.params_json,
            "created_utc": self.created_utc,
        }
