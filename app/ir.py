"""统一中间表示（IR）。

三个交换版本（v1 定宽文本 / v2 JSON / v3 二进制分区）解析后统一为：
- 实体特征记录：entity_id -> feature_name -> float
- 缺失标记：entity_id -> {feature_name}（v1 缺失位 / v2 缺键 / v3 缺失标记）
- 显式空值：entity_id -> {feature_name}（v2 显式 null，与缺键语义不同）
- 精度标签：entity_id -> feature_name -> {permille, double, float32}
- 有序步骤列表：步骤顺序即执行顺序

缺失传播规则（显式声明）：任何步骤的某个输出若依赖于缺失输入，
则该输出亦为缺失；步骤不会为缺失条目凭空造值；缺失标记沿步骤链
向后传播，直到被替换实体特征值编辑显式覆盖。
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field

SCALE = 1_000_000

PRECISION_PERMILLE = "permille"
PRECISION_DOUBLE = "double"
PRECISION_FLOAT32 = "float32"


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def fingerprint(obj) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


@dataclass
class Step:
    name: str
    params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"name": self.name, "params": self.params}

    def fingerprint(self) -> str:
        return fingerprint(self.to_dict())


@dataclass
class FeatureSet:
    values: dict[str, dict[str, float]] = field(default_factory=dict)
    missing: dict[str, set[str]] = field(default_factory=dict)
    explicit_null: dict[str, set[str]] = field(default_factory=dict)
    precision: dict[str, dict[str, str]] = field(default_factory=dict)

    def canonical(self) -> dict:
        return {
            "values": {e: dict(sorted(fs.items())) for e, fs in sorted(self.values.items())},
            "missing": {e: sorted(m) for e, m in sorted(self.missing.items())},
            "explicit_null": {e: sorted(m) for e, m in sorted(self.explicit_null.items())},
            "precision": {e: dict(sorted(p.items())) for e, p in sorted(self.precision.items())},
        }

    def fingerprint(self) -> str:
        return fingerprint(self.canonical())


@dataclass
class IR:
    features: FeatureSet = field(default_factory=FeatureSet)
    steps: list[Step] = field(default_factory=list)
    source_version: str = ""
    version_diffs: list[str] = field(default_factory=list)

    def clone(self) -> "IR":
        return copy.deepcopy(self)

    def canonical(self) -> dict:
        return {
            "features": self.features.canonical(),
            "steps": [s.to_dict() for s in self.steps],
            "source_version": self.source_version,
            "version_diffs": sorted(self.version_diffs),
        }

    def fingerprint(self) -> str:
        return fingerprint(self.canonical())

    def to_json_dict(self) -> dict:
        return self.canonical()

    @staticmethod
    def from_json_dict(d: dict) -> "IR":
        fs = d["features"]
        return IR(
            features=FeatureSet(
                values={e: dict(v) for e, v in fs["values"].items()},
                missing={e: set(m) for e, m in fs["missing"].items()},
                explicit_null={e: set(m) for e, m in fs["explicit_null"].items()},
                precision={e: dict(p) for e, p in fs["precision"].items()},
            ),
            steps=[Step(name=s["name"], params=dict(s.get("params", {}))) for s in d["steps"]],
            source_version=d.get("source_version", ""),
            version_diffs=list(d.get("version_diffs", [])),
        )
