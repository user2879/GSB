"""归一账本使用的不可变内部表示。"""

from __future__ import annotations

from dataclasses import dataclass, field

MISSING = "__MISSING__"


@dataclass(frozen=True, slots=True)
class Worker:
    worker_id: str
    port: int
    model_dim: int
    hidden_dim: int
    param_version: int
    quant: str


@dataclass(frozen=True, slots=True)
class Request:
    seq: int
    request_id: str
    arrival_ms: int
    token_count: int


@dataclass(frozen=True, slots=True)
class MergeRule:
    rule_id: str
    worker_ids: tuple[str, ...]
    batch_size: int


@dataclass(frozen=True, slots=True)
class Receipt:
    """三版本回执的统一内部表示。

    ``_missing`` 记录在来源版本中本应缺失的字段；显式缺失标记
    仅用于升级后补齐，与“缺失位非空”的畸形情形相互区分。
    """

    source_version: str
    source_file: str
    line: int
    batch_id: str
    request_id: str
    output_digest: str
    param_version: int | None
    duration_ms: int | None
    worker_id: str | None
    rank_in_batch: int | None
    supersedes: tuple[str, ...]
    param_summary: str | None
    missing: tuple[str, ...]
    integrity_tag: str
    injection_id: str | None = None

    def as_ledger_dict(self) -> dict:
        return {
            "source_version": self.source_version,
            "source_file": self.source_file,
            "line": self.line,
            "batch_id": self.batch_id,
            "request_id": self.request_id,
            "output_digest": self.output_digest,
            "param_version": self.param_version,
            "duration_ms": self.duration_ms,
            "worker_id": self.worker_id,
            "rank_in_batch": self.rank_in_batch,
            "supersedes": list(self.supersedes),
            "param_summary": self.param_summary,
            "missing": list(self.missing),
            "integrity_tag": self.integrity_tag,
            "injection_id": self.injection_id,
        }


@dataclass(frozen=True, slots=True)
class LedgerEntry:
    request_id: str
    seq: int
    batch_id: str
    worker_id: str | None
    param_version: int | None
    source_version: str
    duration_ms: int | None
    output_digest: str
    rank_in_batch: int | None
    integrity_tag: str
    basis: str
    candidate_count: int
    superseded_refs: tuple[str, ...]
    receipt_ref: str
    invalid: bool
    invalid_reason: str | None
    propagation_path: tuple[str, ...]

    def as_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "seq": self.seq,
            "batch_id": self.batch_id,
            "worker_id": self.worker_id,
            "param_version": self.param_version,
            "source_version": self.source_version,
            "duration_ms": self.duration_ms,
            "output_digest": self.output_digest,
            "rank_in_batch": self.rank_in_batch,
            "integrity_tag": self.integrity_tag,
            "basis": self.basis,
            "candidate_count": self.candidate_count,
            "superseded_refs": list(self.superseded_refs),
            "receipt_ref": self.receipt_ref,
            "invalid": self.invalid,
            "invalid_reason": self.invalid_reason,
            "propagation_path": list(self.propagation_path),
        }


@dataclass(frozen=True, slots=True)
class BatchResource:
    batch_id: str
    worker_id: str
    param_version: int
    request_count: int
    total_duration_ms: int
    max_duration_ms: int
    macs_fixed: int
    utilization_fixed: int
    macs_per_ms_fixed: int
    quant: str
    invalid: bool
    invalid_reason: str | None
    propagation_path: tuple[str, ...]

    def as_dict(self) -> dict:
        return {
            "batch_id": self.batch_id,
            "worker_id": self.worker_id,
            "param_version": self.param_version,
            "request_count": self.request_count,
            "total_duration_ms": self.total_duration_ms,
            "max_duration_ms": self.max_duration_ms,
            "macs_fixed": self.macs_fixed,
            "utilization_fixed": self.utilization_fixed,
            "macs_per_ms_fixed": self.macs_per_ms_fixed,
            "quant": self.quant,
            "invalid": self.invalid,
            "invalid_reason": self.invalid_reason,
            "propagation_path": list(self.propagation_path),
        }


@dataclass(frozen=True, slots=True)
class ReuseRecord:
    stage: str
    key: str
    action: str
    reason: str
    propagation_path: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict:
        return {
            "stage": self.stage,
            "key": self.key,
            "action": self.action,
            "reason": self.reason,
            "propagation_path": list(self.propagation_path),
        }
