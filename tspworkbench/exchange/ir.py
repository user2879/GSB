"""交换文件的统一中间表示（IR）。

v1（定宽文本）与 v2（JSON）解析后都归一化成本模块的数据类；
统一表示负责三类硬校验：序列 ID+窗口序号组合唯一、滞后阶数非负整数、
模型名在允许枚举内。解析期的布局错误在解析器内直接产出 E_FIELD_LAYOUT。
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

from ..constants import MODELS, MISSING_STRATEGIES
from ..errors import WorkbenchError, E_DUPLICATE_KEY, E_LAG_NEGATIVE, E_MODEL_ENUM, E_MISSING_ENUM
from ..util import fingerprint


@dataclass
class WindowWeight:
    series_id: str
    window_index: int
    model: str
    weight: float                 # 统一表示里始终是浮点权重
    lag_order: int
    precision_label: str = "exact"


@dataclass
class WindowPrediction:
    series_id: str
    window_index: int
    model: str
    step_index: int
    pred_q: int
    actual_q: int


@dataclass
class WindowMetric:
    series_id: str
    window_index: int
    mae_q: int
    rmse_q: int
    mape_bp: int
    near_zero: bool


@dataclass
class ExchangeIR:
    exchange_version: str
    config_fingerprint: str
    version_fingerprint: str
    missing_strategy: str
    weights: list[WindowWeight] = field(default_factory=list)
    predictions: list[WindowPrediction] = field(default_factory=list)
    metrics: list[WindowMetric] = field(default_factory=list)

    def validate(self) -> None:
        # 1) (series_id, window_index, model) 唯一
        seen_w: set[tuple[str, int, str]] = set()
        for w in self.weights:
            key = (w.series_id, w.window_index, w.model)
            if key in seen_w:
                raise WorkbenchError(E_DUPLICATE_KEY,
                                     "权重记录的 序列ID+窗口序号+模型 组合不唯一",
                                     context={"key": list(key)})
            seen_w.add(key)
            if not isinstance(w.lag_order, int) or w.lag_order < 0:
                raise WorkbenchError(E_LAG_NEGATIVE,
                                     f"滞后阶数必须为非负整数: {key}",
                                     context={"lag_order": w.lag_order})
            if w.model not in MODELS:
                raise WorkbenchError(E_MODEL_ENUM, f"模型名 {w.model!r} 越界",
                                     context={"model": w.model})

        seen_p: set[tuple[str, int, str, int]] = set()
        for p in self.predictions:
            key = (p.series_id, p.window_index, p.model, p.step_index)
            if key in seen_p:
                raise WorkbenchError(E_DUPLICATE_KEY,
                                     "预测记录的 序列ID+窗口+模型+步序 组合不唯一",
                                     context={"key": list(key)})
            seen_p.add(key)
            if p.model not in MODELS:
                raise WorkbenchError(E_MODEL_ENUM, f"模型名 {p.model!r} 越界")

        seen_m: set[tuple[str, int]] = set()
        for m in self.metrics:
            key = (m.series_id, m.window_index)
            if key in seen_m:
                raise WorkbenchError(E_DUPLICATE_KEY,
                                     "指标记录的 序列ID+窗口序号 组合不唯一",
                                     context={"key": list(key)})
            seen_m.add(key)

        if self.missing_strategy not in MISSING_STRATEGIES:
            raise WorkbenchError(E_MISSING_ENUM,
                                 f"缺失处理方式 {self.missing_strategy!r} 越界",
                                 context={"value": self.missing_strategy})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def content_fingerprint(self) -> str:
        return fingerprint(self.to_dict())
