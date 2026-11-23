"""三个确定性基线模型，共用同一 fit/predict 接口。

- naive：朴素持续，预测恒等于训练窗末值；
- moving_average：移动平均，窗长取自统一表示的模型阶数；
- autoregressive：最小二乘自回归，滞后阶数取自统一表示，递归多步预测。

每个模型都返回训练窗口、拟合参数与残差序列，且不读取任何全局随机状态。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .config import ModelSpec
from .errors import WorkbenchError


@dataclass
class FitArtifact:
    model: str
    lag_order: int
    train_window: list[float]
    params: dict
    residuals: list[float]

    def predict(self, horizon: int) -> list[float]:
        if self.model == "naive":
            return [float(self.params["level"])] * horizon
        if self.model == "moving_average":
            return [float(self.params["mean"])] * horizon
        if self.model == "autoregressive":
            return self._predict_ar(horizon)
        raise WorkbenchError("E_MODEL_ENUM", f"未知模型 {self.model}")

    def _predict_ar(self, horizon: int) -> list[float]:
        lag = int(self.params["lag"])
        intercept = float(self.params["intercept"])
        coef = [float(c) for c in self.params["coef"]]
        history = [float(v) for v in self.train_window[-lag:]]
        preds: list[float] = []
        for _ in range(horizon):
            window = history[-lag:]
            value = intercept + sum(coef[j] * window[lag - 1 - j] for j in range(lag))
            preds.append(float(value))
            history.append(value)
        return preds


class BaseModel(Protocol):
    def fit(self, train: list[float], spec: ModelSpec) -> FitArtifact: ...


class NaiveModel:
    def fit(self, train: list[float], spec: ModelSpec) -> FitArtifact:
        if not train:
            raise WorkbenchError("E_PARSE", "naive 训练窗口为空")
        level = float(train[-1])
        residuals = [float(train[i] - train[i - 1]) for i in range(1, len(train))]
        return FitArtifact("naive", spec.lag_order, list(train),
                           {"level": level}, residuals)


class MovingAverageModel:
    def fit(self, train: list[float], spec: ModelSpec) -> FitArtifact:
        window = spec.lag_order if spec.lag_order > 0 else spec.extra.get("window", 0)
        if window <= 0:
            raise WorkbenchError("E_PARSE", "moving_average 需要正的窗长（lag_order）",
                                 context={"model": spec.name})
        if window > len(train):
            raise WorkbenchError("E_PARSE", "moving_average 窗长超过训练窗口",
                                 context={"window": window, "train_len": len(train)})
        tail = [float(v) for v in train[-window:]]
        mean = float(np.mean(np.asarray(tail, dtype=np.float64)))
        residuals = []
        for t in range(window, len(train)):
            rolling = float(np.mean(np.asarray(train[t - window:t], dtype=np.float64)))
            residuals.append(float(train[t] - rolling))
        return FitArtifact("moving_average", window, list(train),
                           {"mean": mean, "window": window}, residuals)


class AutoRegressiveModel:
    def fit(self, train: list[float], spec: ModelSpec) -> FitArtifact:
        lag = spec.lag_order
        if lag <= 0:
            raise WorkbenchError("E_PARSE", "autoregressive 需要正的滞后阶数",
                                 context={"model": spec.name})
        if len(train) <= lag:
            raise WorkbenchError("E_PARSE", "autoregressive 训练样本不足以支撑滞后阶数",
                                 context={"lag": lag, "train_len": len(train)})
        x_rows: list[list[float]] = []
        y_rows: list[float] = []
        for t in range(lag, len(train)):
            x_rows.append([1.0] + [float(train[t - 1 - j]) for j in range(lag)])
            y_rows.append(float(train[t]))
        design = np.asarray(x_rows, dtype=np.float64)
        target = np.asarray(y_rows, dtype=np.float64)
        beta, *_ = np.linalg.lstsq(design, target, rcond=None)
        beta = [float(b) for b in beta]
        fitted = design @ np.asarray(beta, dtype=np.float64)
        residuals = [float(r) for r in (target - fitted)]
        return FitArtifact(
            "autoregressive", lag, list(train),
            {"lag": lag, "intercept": beta[0], "coef": beta[1:]},
            residuals,
        )


_MODEL_CLASSES = {
    "naive": NaiveModel,
    "moving_average": MovingAverageModel,
    "autoregressive": AutoRegressiveModel,
}


def fit_model(spec: ModelSpec, train: list[float]) -> FitArtifact:
    return _MODEL_CLASSES[spec.name]().fit(train, spec)
