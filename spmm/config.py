"""引擎配置：块大小、alpha/beta、精度档位、预算、并发与快照间隔。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from .errors import E_BLOCK_INVALID, E_CONFIG_INVALID, SpmmError
from .kernel import ORDERS, PRECISIONS

# 影响计算结果的参数：变更时触发全量失效，并纳入配置指纹。
RESULT_PARAMS = (
    "row_block_size",
    "nnz_chunk_size",
    "alpha",
    "beta",
    "precision",
    "accumulation_order",
)


@dataclass
class EngineConfig:
    row_block_size: int = 8
    nnz_chunk_size: int = 16
    alpha: float = 1.0
    beta: float = 0.0
    precision: str = "fp32"
    accumulation_order: str = "asc"
    recompute_budget: int = 4
    max_workers: int = 4
    snapshot_commit_interval: int = 2
    report_dir: str = "reports"
    data_dir: str = "data"

    @classmethod
    def from_dict(cls, payload):
        known = {k: v for k, v in payload.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    @classmethod
    def from_json(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def to_dict(self):
        return asdict(self)

    def fingerprint(self):
        payload = {k: getattr(self, k) for k in RESULT_PARAMS}
        blob = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.blake2b(blob, digest_size=16).hexdigest()

    def validate(self, m, k):
        violations = []
        if not (1 <= int(self.row_block_size) <= max(1, m)):
            violations.append({
                "field": "row_block_size",
                "value": self.row_block_size,
                "detail": f"合法范围 [1, {max(1, m)}]",
            })
        if not (1 <= int(self.nnz_chunk_size) <= max(1, k)):
            violations.append({
                "field": "nnz_chunk_size",
                "value": self.nnz_chunk_size,
                "detail": f"合法范围 [1, {max(1, k)}]",
            })
        if violations:
            raise SpmmError(E_BLOCK_INVALID, "块参数越界", violations)
        if self.precision not in PRECISIONS:
            raise SpmmError(E_CONFIG_INVALID, f"未知精度档位: {self.precision}",
                            [{"field": "precision", "value": self.precision}])
        if self.accumulation_order not in ORDERS:
            raise SpmmError(E_CONFIG_INVALID, f"未知累加顺序: {self.accumulation_order}",
                            [{"field": "accumulation_order", "value": self.accumulation_order}])
        for field in ("alpha", "beta"):
            value = float(getattr(self, field))
            if value != value or value in (float("inf"), float("-inf")):
                raise SpmmError(E_CONFIG_INVALID, f"{field} 必须为有限值",
                                [{"field": field, "value": getattr(self, field)}])
        if int(self.recompute_budget) < 0:
            raise SpmmError(E_CONFIG_INVALID, "重算预算不能为负",
                            [{"field": "recompute_budget", "value": self.recompute_budget}])
        if int(self.max_workers) < 1:
            raise SpmmError(E_CONFIG_INVALID, "并发数至少为 1",
                            [{"field": "max_workers", "value": self.max_workers}])
        if int(self.snapshot_commit_interval) < 1:
            raise SpmmError(E_CONFIG_INVALID, "快照提交间隔至少为 1",
                            [{"field": "snapshot_commit_interval",
                              "value": self.snapshot_commit_interval}])
