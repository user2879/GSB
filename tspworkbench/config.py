"""预测配置的加载、归一化、校验与配置指纹。"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

from .constants import MODELS, MISSING_STRATEGIES, EXCHANGE_VERSIONS, QUANT_GEARS
from .errors import WorkbenchError, E_MODEL_ENUM, E_MISSING_ENUM, E_LAG_NEGATIVE, E_QUANT_GEAR_INVALID
from .util import canonical, fingerprint


@dataclass(frozen=True)
class ModelSpec:
    name: str
    weight: float
    lag_order: int
    extra: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class QuantSpec:
    frac_bits: int
    gear: int

    @property
    def effective_frac_bits(self) -> int:
        """档位 gear 与位宽 frac_bits 共同决定真实定点标度。"""
        return self.frac_bits + self.gear

    @property
    def scale(self) -> int:
        return 2 ** self.effective_frac_bits


@dataclass(frozen=True)
class ExecutionSpec:
    max_workers: int
    observation_budget: int
    series_quota: int
    snapshot_every_windows: int
    temp_dir: str


@dataclass(frozen=True)
class Config:
    window_length: int
    window_step: int
    horizon: int
    models: tuple[ModelSpec, ...]
    feature_lags: tuple[int, ...]
    missing_strategy: str
    quant: QuantSpec
    exchange_version: str
    execution: ExecutionSpec

    def normalized(self) -> dict[str, Any]:
        return {
            "window_length": self.window_length,
            "window_step": self.window_step,
            "horizon": self.horizon,
            "models": [asdict(m) for m in sorted(self.models, key=lambda m: m.name)],
            "feature_lags": list(self.feature_lags),
            "missing_strategy": self.missing_strategy,
            "quant": asdict(self.quant),
            "exchange_version": self.exchange_version,
            "execution": asdict(self.execution),
        }

    def config_hash(self) -> str:
        return fingerprint(self.normalized())


def _as_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WorkbenchError("E_PARSE", f"字段 {field_name} 必须是整数", context={"value": value})
    if int(value) != value:
        raise WorkbenchError("E_PARSE", f"字段 {field_name} 必须是整数", context={"value": value})
    return int(value)


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise WorkbenchError("E_PARSE", "配置根节点必须是映射")
    return build_config(raw)


def build_config(raw: dict[str, Any]) -> Config:
    bt = raw.get("backtest", {})
    window_length = _as_int(bt.get("window_length"), "backtest.window_length")
    window_step = _as_int(bt.get("window_step"), "backtest.window_step")
    horizon = _as_int(bt.get("horizon"), "backtest.horizon")
    if min(window_length, window_step, horizon) <= 0:
        raise WorkbenchError("E_PARSE", "回测窗口长度/步长/预测步长必须为正整数")

    model_specs: list[ModelSpec] = []
    for entry in raw.get("models", []):
        name = entry.get("name")
        if name not in MODELS:
            raise WorkbenchError(E_MODEL_ENUM, f"模型名 {name!r} 不在允许枚举 {MODELS} 内",
                                 context={"value": name})
        lag_order = _as_int(entry.get("lag_order", 0), f"models[{name}].lag_order")
        if lag_order < 0:
            raise WorkbenchError(E_LAG_NEGATIVE, f"模型 {name} 滞后阶数必须为非负整数",
                                 context={"lag_order": lag_order})
        weight = float(entry.get("weight", 0.0))
        extra = {k: _as_int(v, f"models[{name}].{k}")
                 for k, v in entry.items()
                 if k not in ("name", "weight", "lag_order")}
        model_specs.append(ModelSpec(name, weight, lag_order, extra))
    if not model_specs:
        raise WorkbenchError("E_PARSE", "至少需要一个基线模型")
    names = [m.name for m in model_specs]
    if len(set(names)) != len(names):
        raise WorkbenchError(E_MODEL_ENUM, "模型集合中存在重复模型名", context={"names": names})

    feature_lags = tuple(sorted(_as_int(v, "features.lag_orders")
                                for v in raw.get("features", {}).get("lag_orders", [])))
    if any(lag < 0 for lag in feature_lags):
        raise WorkbenchError(E_LAG_NEGATIVE, "特征滞后阶数必须为非负整数")

    missing_strategy = raw.get("missing", {}).get("strategy", "forward_fill")
    if missing_strategy not in MISSING_STRATEGIES:
        raise WorkbenchError(E_MISSING_ENUM,
                             f"缺失处理方式 {missing_strategy!r} 不在 {MISSING_STRATEGIES} 内",
                             context={"value": missing_strategy})

    q = raw.get("quantization", {})
    frac_bits = _as_int(q.get("frac_bits", 4), "quantization.frac_bits")
    gear = _as_int(q.get("gear", 0), "quantization.gear")
    if gear not in QUANT_GEARS:
        raise WorkbenchError(E_QUANT_GEAR_INVALID,
                             f"量化档位 {gear} 越界，允许 {QUANT_GEARS}",
                             context={"gear": gear})
    if frac_bits < 0:
        raise WorkbenchError(E_QUANT_GEAR_INVALID, "定点小数位宽必须非负",
                             context={"frac_bits": frac_bits})

    exchange_version = raw.get("exchange", {}).get("version", "v1")
    if exchange_version not in EXCHANGE_VERSIONS:
        raise WorkbenchError("E_PARSE", f"交换版本 {exchange_version!r} 非法")

    ex = raw.get("execution", {})
    execution = ExecutionSpec(
        max_workers=max(1, _as_int(ex.get("max_workers", 1), "execution.max_workers")),
        observation_budget=_as_int(ex.get("observation_budget", 10 ** 9),
                                   "execution.observation_budget"),
        series_quota=_as_int(ex.get("series_quota", 10 ** 9), "execution.series_quota"),
        snapshot_every_windows=max(1, _as_int(ex.get("snapshot_every_windows", 8),
                                              "execution.snapshot_every_windows")),
        temp_dir=str(ex.get("temp_dir", ".tmp")),
    )

    return Config(
        window_length=window_length,
        window_step=window_step,
        horizon=horizon,
        models=tuple(model_specs),
        feature_lags=feature_lags,
        missing_strategy=missing_strategy,
        quant=QuantSpec(frac_bits, gear),
        exchange_version=exchange_version,
        execution=execution,
    )


def config_canonical(cfg: Config) -> str:
    return canonical(cfg.normalized())
