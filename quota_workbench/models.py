"""公共模型与确定性序列化工具。"""

import hashlib
import json
from dataclasses import dataclass, field

METRICS = ("capacity_limit_bytes", "ops_per_second", "burst_bucket_capacity")

SOURCE_TYPES = ("capacity_claim", "rate_claim", "derived_claim")


def canonical(obj):
    """确定性 JSON 序列化：键排序、无空白，保证逐字节可比较。"""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(obj):
    return hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()


@dataclass
class NormalizedRecord:
    """归一后的来源记录内部表示。"""

    source_id: str
    source_type: str
    format_version: int
    metrics: dict
    defaulted_fields: list = field(default_factory=list)
    quantization_error: int = 0
    extra: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "source_id": self.source_id,
            "source_type": self.source_type,
            "format_version": self.format_version,
            "metrics": dict(sorted(self.metrics.items())),
            "defaulted_fields": sorted(self.defaulted_fields),
            "quantization_error": self.quantization_error,
            "extra": self.extra,
        }
