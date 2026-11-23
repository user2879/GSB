"""算子配置加载与校验。

配置 JSON 字段（均有默认值）：
- query_block_len / key_block_len / head_dim
- burst_query_blocks            突发查询块清单
- fixed_point_frac_bits         在线 softmax 定点小数位宽
- quantization_level_bits       量化档位（枚举 8/16）
- accumulator_int_bits          累加缓冲整数位宽（枚举 32/64）
- stability_offset              数值稳定偏移
- tolerances                    容差参数（score / softmax_state / output）
- thread_pool_workers           线程池并发数
- point_budget_per_batch        单批打分点数预算
- snapshot_commit_interval_batches  快照提交间隔批次
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace

from .errors import Code

ALLOWED_QUANT_LEVELS = (8, 16)
ALLOWED_ACC_BITS = (32, 64)
FRAC_BITS_RANGE = (0, 24)
WORKERS_RANGE = (1, 16)
TOLERANCE_STAGES = ("score", "softmax_state", "output")


@dataclass(frozen=True)
class Issue:
    code: str
    message: str
    context: dict = field(default_factory=dict)

    def to_dict(self):
        return {"code": self.code, "message": self.message, "context": self.context}


@dataclass(frozen=True)
class OperatorConfig:
    query_block_len: int = 8
    key_block_len: int = 8
    head_dim: int = 8
    burst_query_blocks: tuple = ()
    fixed_point_frac_bits: int = 12
    quantization_level_bits: int = 16
    accumulator_int_bits: int = 64
    stability_offset: float = 0.0
    tol_score: float = 0.05
    tol_softmax_state: float = 0.05
    tol_output: float = 0.05
    thread_pool_workers: int = 4
    point_budget_per_batch: int = 1_000_000
    snapshot_commit_interval_batches: int = 1

    @classmethod
    def from_dict(cls, data: dict) -> "OperatorConfig":
        tolerances = data.get("tolerances", {}) or {}
        return cls(
            query_block_len=int(data.get("query_block_len", 8)),
            key_block_len=int(data.get("key_block_len", 8)),
            head_dim=int(data.get("head_dim", 8)),
            burst_query_blocks=tuple(int(b) for b in data.get("burst_query_blocks", [])),
            fixed_point_frac_bits=int(data.get("fixed_point_frac_bits", 12)),
            quantization_level_bits=int(data.get("quantization_level_bits", 16)),
            accumulator_int_bits=int(data.get("accumulator_int_bits", 64)),
            stability_offset=float(data.get("stability_offset", 0.0)),
            tol_score=float(tolerances.get("score", 0.05)),
            tol_softmax_state=float(tolerances.get("softmax_state", 0.05)),
            tol_output=float(tolerances.get("output", 0.05)),
            thread_pool_workers=int(data.get("thread_pool_workers", 4)),
            point_budget_per_batch=int(data.get("point_budget_per_batch", 1_000_000)),
            snapshot_commit_interval_batches=int(
                data.get("snapshot_commit_interval_batches", 1)),
        )

    @classmethod
    def load(cls, path) -> "OperatorConfig":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def with_overrides(self, **kwargs) -> "OperatorConfig":
        return replace(self, **kwargs)

    def tolerance_for(self, stage: str) -> float:
        return {
            "score": self.tol_score,
            "softmax_state": self.tol_softmax_state,
            "output": self.tol_output,
        }[stage]

    def validate(self) -> list:
        issues = []
        if self.query_block_len <= 0 or self.key_block_len <= 0:
            issues.append(Issue(Code.E_PAD_RULE_VIOLATION,
                                "块长必须为正整数",
                                {"query_block_len": self.query_block_len,
                                 "key_block_len": self.key_block_len}))
        if self.head_dim <= 0 or self.head_dim % 2 != 0:
            issues.append(Issue(Code.E_HEAD_DIM_MISMATCH,
                                "头维度必须为正偶数",
                                {"head_dim": self.head_dim}))
        lo, hi = FRAC_BITS_RANGE
        if not (lo <= self.fixed_point_frac_bits <= hi):
            issues.append(Issue(Code.E_FRAC_BITS_OUT_OF_RANGE,
                                "定点小数位宽越界",
                                {"fixed_point_frac_bits": self.fixed_point_frac_bits,
                                 "allowed": [lo, hi]}))
        if self.quantization_level_bits not in ALLOWED_QUANT_LEVELS:
            issues.append(Issue(Code.E_QUANT_LEVEL_INVALID,
                                "量化档位不在允许枚举内",
                                {"quantization_level_bits": self.quantization_level_bits,
                                 "allowed": list(ALLOWED_QUANT_LEVELS)}))
        if self.accumulator_int_bits not in ALLOWED_ACC_BITS:
            issues.append(Issue(Code.E_QUANT_LEVEL_INVALID,
                                "累加缓冲整数位宽不在允许枚举内",
                                {"accumulator_int_bits": self.accumulator_int_bits,
                                 "allowed": list(ALLOWED_ACC_BITS)}))
        wlo, whi = WORKERS_RANGE
        if not (wlo <= self.thread_pool_workers <= whi):
            issues.append(Issue(Code.E_WORKERS_OUT_OF_RANGE,
                                "线程并发数越界",
                                {"thread_pool_workers": self.thread_pool_workers,
                                 "allowed": [wlo, whi]}))
        if self.point_budget_per_batch <= 0:
            issues.append(Issue(Code.E_POINT_BUDGET_PENDING,
                                "单批打分点数预算必须为正",
                                {"point_budget_per_batch": self.point_budget_per_batch}))
        if self.snapshot_commit_interval_batches <= 0:
            issues.append(Issue(Code.E_SNAPSHOT_VERSION_MISMATCH,
                                "快照提交间隔必须为正",
                                {"snapshot_commit_interval_batches":
                                 self.snapshot_commit_interval_batches}))
        return issues

    def canonical(self) -> dict:
        """参与参数指纹/配置哈希的字段（含容差与量化档位）。"""
        return {
            "query_block_len": self.query_block_len,
            "key_block_len": self.key_block_len,
            "head_dim": self.head_dim,
            "fixed_point_frac_bits": self.fixed_point_frac_bits,
            "quantization_level_bits": self.quantization_level_bits,
            "accumulator_int_bits": self.accumulator_int_bits,
            "stability_offset": self.stability_offset,
            "tol_score": self.tol_score,
            "tol_softmax_state": self.tol_softmax_state,
            "tol_output": self.tol_output,
        }

    def config_hash(self) -> str:
        blob = json.dumps(self.canonical(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]
