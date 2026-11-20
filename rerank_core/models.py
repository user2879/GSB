"""领域数据模型（纯数据容器，全部可 JSON 序列化）。"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass(frozen=True)
class Candidate:
    request_id: str
    item_id: str
    recall_scores: dict[str, float]          # 通道名 -> 原始分数
    dense: list[float]                       # 稠密特征
    sparse: list[int]                        # 稀疏计数
    line_no: int = 0                         # JSONL 行号，便于失败清单定位

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FeatureGroup:
    name: str
    feature_indexes: list[int]               # 组内特征在统一特征向量中的下标
    conflict: bool                           # 是否存在量纲冲突


@dataclass
class Failure:
    code: str
    scope: str                               # request / config / snapshot
    target: str                              # 请求 ID、文件或阶段名
    message: str
    line_no: int | None = None

    def to_dict(self) -> dict[str, Any]:
        data = {"code": self.code, "scope": self.scope, "target": self.target,
                "message": self.message}
        if self.line_no is not None:
            data["line_no"] = self.line_no
        return data


@dataclass
class Contribution:
    feature_index: int
    feature_name: str
    group: str
    normalized_fixed: int                    # 归一化并量化后的定点值
    weight_fixed: int                       # 量化后的权重（定点）
    contribution_fixed: int                  # 该特征对得分的整数贡献
    frac_bits: int


@dataclass
class FusionResult:
    request_id: str
    item_id: str
    score_fixed: int
    score_clip_low: int
    score_clip_high: int
    clipped: bool
    contributions: list[Contribution]
    frac_bits: int
    quant_levels: int
    task_key: int
    recall_tie_channel: str                  # 同分决断使用的最高召回通道
    recall_tie_score_exact: str              # 该通道原始精确分数字符串


@dataclass
class DiscountRecord:
    target_item: str
    reference_item: str
    similarity_fixed: int
    threshold_fixed: int
    before_score_fixed: int
    after_score_fixed: int
    penalty_fixed: int
    step: int


@dataclass
class RerankResult:
    request_id: str
    window_size: int
    final_order: list[str]                   # 窗口内最终物品顺序
    outside_order: list[str]                 # 窗口外保持融合序的物品
    discounts: list[DiscountRecord]
    initial_window_order: list[str]


@dataclass
class RequestReport:
    request_id: str
    fusion_order: list[str]
    fusion_scores: dict[str, int]
    rerank: RerankResult
    high_tier_diff: dict[str, int]           # item -> 高档位与当前档位得分差
