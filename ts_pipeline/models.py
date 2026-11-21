"""Three deterministic baseline forecasters sharing one interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .canonical import fingerprint, to_builtin
from .config import ModelConfig
from .errors import E_MODEL_FIT_FAILED


class ModelError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class TrainingContext:
    series_id: str
    window_index: int
    train_start_index: int
    train_end_index: int
    train_start_ms: int
    train_end_ms: int
    values: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "series_id": self.series_id,
            "window_index": self.window_index,
            "start_index": self.train_start_index,
            "end_index_exclusive": self.train_end_index,
            "start_timestamp_ms": self.train_start_ms,
            "end_timestamp_ms": self.train_end_ms,
            "length": len(self.values),
            "values": list(self.values),
            "values_fingerprint": fingerprint(
                {"schema": "training-values-v1", "values": list(self.values)}
            ),
        }


class BaselineModel(ABC):
    """Common fit/predict interface for every baseline."""

    model_name: str

    @abstractmethod
    def fit(
        self,
        values: Sequence[float],
        timestamps_ms: Sequence[int],
        context: TrainingContext,
        config: ModelConfig,
    ) -> dict[str, Any]:
        """Return a serializable training artifact."""

    @abstractmethod
    def predict(self, artifact: dict[str, Any], horizon: int) -> list[float]:
        """Return exactly ``horizon`` deterministic forecasts."""


def _residual_fingerprint(residuals: list[float]) -> str:
    return fingerprint({"schema": "residuals-v1", "values": residuals})


def _finalize_artifact(
    *,
    model_name: str,
    context: TrainingContext,
    fitted_params: dict[str, Any],
    residuals: list[float],
) -> dict[str, Any]:
    return {
        "schema": "training-artifact-v1",
        "model_name": model_name,
        "training_window": context.to_dict(),
        "fitted_params": to_builtin(fitted_params),
        "residual_sequence": [float(value) for value in residuals],
        "residual_fingerprint": _residual_fingerprint([float(value) for value in residuals]),
    }


class NaivePersistenceModel(BaselineModel):
    model_name = "naive_persistence"

    def fit(
        self,
        values: Sequence[float],
        timestamps_ms: Sequence[int],
        context: TrainingContext,
        config: ModelConfig,
    ) -> dict[str, Any]:
        if not values:
            raise ModelError(E_MODEL_FIT_FAILED, "naive persistence requires at least one value")
        residuals = [float(values[index] - values[index - 1]) for index in range(1, len(values))]
        params = {
            "last_value": float(values[-1]),
            "persistence_horizon": "all_steps",
        }
        return _finalize_artifact(
            model_name=self.model_name,
            context=context,
            fitted_params=params,
            residuals=residuals,
        )

    def predict(self, artifact: dict[str, Any], horizon: int) -> list[float]:
        value = float(artifact["fitted_params"]["last_value"])
        return [value for _ in range(horizon)]


class MovingAverageModel(BaselineModel):
    model_name = "moving_average"

    def fit(
        self,
        values: Sequence[float],
        timestamps_ms: Sequence[int],
        context: TrainingContext,
        config: ModelConfig,
    ) -> dict[str, Any]:
        window = config.moving_average_window
        if len(values) < window:
            raise ModelError(
                E_MODEL_FIT_FAILED,
                f"moving average needs {window} training values but received {len(values)}",
            )
        residuals = []
        for index in range(window, len(values)):
            previous_mean = sum(float(value) for value in values[index - window : index]) / window
            residuals.append(float(values[index]) - previous_mean)
        fitted_params = {
            "window": window,
            "last_window_mean": sum(float(value) for value in values[-window:]) / window,
            "last_window": [float(value) for value in values[-window:]],
        }
        return _finalize_artifact(
            model_name=self.model_name,
            context=context,
            fitted_params=fitted_params,
            residuals=residuals,
        )

    def predict(self, artifact: dict[str, Any], horizon: int) -> list[float]:
        value = float(artifact["fitted_params"]["last_window_mean"])
        return [value for _ in range(horizon)]


class AutoregressiveModel(BaselineModel):
    model_name = "autoregressive"

    def fit(
        self,
        values: Sequence[float],
        timestamps_ms: Sequence[int],
        context: TrainingContext,
        config: ModelConfig,
    ) -> dict[str, Any]:
        lags = config.feature_lag_order
        if len(values) <= lags:
            raise ModelError(
                E_MODEL_FIT_FAILED,
                f"autoregressive model needs more than {lags} training values",
            )
        y = np.asarray(values, dtype=np.float64)
        rows = []
        targets = []
        for index in range(lags, len(y)):
            rows.append([1.0, *[float(y[index - lag]) for lag in range(1, lags + 1)]])
            targets.append(float(y[index]))
        design = np.asarray(rows, dtype=np.float64)
        target = np.asarray(targets, dtype=np.float64)
        normal = design.T @ design
        regularization = float(config.autoregressive_regularization)
        penalty = np.zeros((lags + 1, lags + 1), dtype=np.float64)
        if lags:
            penalty[1:, 1:] = np.eye(lags, dtype=np.float64) * regularization
        regularized = normal + penalty
        solver = "normal_equations"
        try:
            coefficients = np.linalg.solve(regularized, design.T @ target)
            if not np.all(np.isfinite(coefficients)):
                raise np.linalg.LinAlgError("non-finite normal-equation solution")
        except np.linalg.LinAlgError:
            solver = "normal_equations_pseudoinverse"
            coefficients = np.linalg.pinv(regularized) @ (design.T @ target)
            if not np.all(np.isfinite(coefficients)):
                raise ModelError(E_MODEL_FIT_FAILED, "autoregressive fit produced non-finite coefficients")
        fitted = design @ coefficients
        residuals = [float(target[index] - fitted[index]) for index in range(len(target))]
        fitted_params = {
            "lags": lags,
            "regularization": regularization,
            "intercept": float(coefficients[0]),
            "coefficients": [float(value) for value in coefficients[1:]],
            "solver": solver,
            "normal_matrix_fingerprint": fingerprint(regularized.tolist()),
        }
        return _finalize_artifact(
            model_name=self.model_name,
            context=context,
            fitted_params=fitted_params,
            residuals=residuals,
        )

    def predict(self, artifact: dict[str, Any], horizon: int) -> list[float]:
        params = artifact["fitted_params"]
        coefficients = [float(value) for value in params["coefficients"]]
        intercept = float(params["intercept"])
        history = [float(value) for value in artifact["training_window"]["values"]]
        predictions: list[float] = []
        for _ in range(horizon):
            value = intercept
            for lag, coefficient in enumerate(coefficients, start=1):
                value += coefficient * history[-lag]
            predictions.append(float(value))
            history.append(float(value))
        return predictions


MODEL_TYPES: dict[str, type[BaselineModel]] = {
    NaivePersistenceModel.model_name: NaivePersistenceModel,
    MovingAverageModel.model_name: MovingAverageModel,
    AutoregressiveModel.model_name: AutoregressiveModel,
}


def model_for_name(name: str) -> BaselineModel:
    try:
        return MODEL_TYPES[name]()
    except KeyError as exc:
        raise ModelError(E_MODEL_FIT_FAILED, f"unknown model: {name}") from exc
