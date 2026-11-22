"""Deterministic regularized batch-gradient training."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .errors import E_MODEL_BUDGET_EXHAUSTED, E_MODEL_EXCEPTION
from .util import coefficient_digest, split_batches, to_jsonable


@dataclass
class TrainingFailure(Exception):
    code: str
    stage: str
    epoch: int
    exception_type: str
    message: str

    def __post_init__(self) -> None:
        Exception.__init__(self, self.message)

    def to_dict(self, model: str) -> dict[str, Any]:
        return {
            "model": model,
            "epoch": int(self.epoch),
            "stage": self.stage,
            "code": self.code,
            "exception_type": self.exception_type,
            "message": self.message,
        }


@dataclass
class TrainingInterrupted(Exception):
    state: dict[str, Any]

    def __post_init__(self) -> None:
        Exception.__init__(self, "training interrupted")


CheckpointCallback = Callable[[dict[str, Any], str], None]


@dataclass(frozen=True)
class TrainingHyperparameters:
    epochs: int
    learning_rate: float
    batch_size: int
    tolerance: float
    l1_zero_subgradient: float
    snapshot_every_batches: int

    def canonical(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "tolerance": self.tolerance,
            "l1_zero_subgradient": self.l1_zero_subgradient,
            "snapshot_every_batches": self.snapshot_every_batches,
            "optimizer": "deterministic-full-batch-cycle",
            "loss": "0.5*MSE + L1/L2 regularization",
            "trainer_schema": 1,
        }


def _matvec(matrix: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    # Explicit reduction avoids BLAS scheduling as a source of nondeterminism.
    return np.sum(matrix * coefficients.reshape(1, -1), axis=1, dtype=np.float64)


def _regularization(
    scheme: str, penalty: float, mix: float | None, coefficients: np.ndarray
) -> tuple[float, np.ndarray, dict[str, float]]:
    if scheme == "l2":
        l2_ratio, l1_ratio = 1.0, 0.0
    elif scheme == "l1":
        l2_ratio, l1_ratio = 0.0, 1.0
    elif scheme == "elastic_net":
        elastic_mix = 0.5 if mix is None else float(mix)
        l2_ratio, l1_ratio = 1.0 - elastic_mix, elastic_mix
    else:
        raise ValueError(f"unsupported scheme: {scheme}")
    l1_norm = float(np.sum(np.abs(coefficients), dtype=np.float64))
    l2_squared = float(np.sum(coefficients * coefficients, dtype=np.float64))
    value = penalty * (0.5 * l2_ratio * l2_squared + l1_ratio * l1_norm)
    return value, np.asarray([l1_ratio, l2_ratio], dtype=np.float64), {
        "l1_norm": l1_norm,
        "l2_norm": math.sqrt(l2_squared),
        "l2_squared": l2_squared,
        "l1_ratio": l1_ratio,
        "l2_ratio": l2_ratio,
    }


def _subgradient_sign(value: float, zero_decision: float) -> float:
    if value > 0.0:
        return 1.0
    if value < 0.0:
        return -1.0
    return float(zero_decision)


def _validation_metrics(
    x_validation: np.ndarray, y_validation: np.ndarray, coefficients: np.ndarray, intercept: float
) -> dict[str, Any]:
    predictions = _matvec(x_validation, coefficients) + intercept
    residual = predictions - y_validation
    squared = residual * residual
    mse = float(np.mean(squared, dtype=np.float64))
    mae = float(np.mean(np.abs(residual), dtype=np.float64))
    positive = predictions >= 0.5
    accuracy = float(np.mean(positive == (y_validation >= 0.5), dtype=np.float64))
    clipped = np.clip(predictions, 1e-12, 1.0 - 1e-12)
    log_loss = -float(
        np.mean(
            y_validation * np.log(clipped) + (1.0 - y_validation) * np.log(1.0 - clipped),
            dtype=np.float64,
        )
    )
    return {
        "rows": int(y_validation.size),
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": mae,
        "accuracy@0.5": accuracy,
        "log_loss@clipped": log_loss,
        "prediction_mean": float(np.mean(predictions, dtype=np.float64)),
        "validation_positive_rate": float(np.mean(y_validation, dtype=np.float64)),
    }


def _make_checkpoint_state(
    coefficients: np.ndarray,
    intercept: float,
    completed_epochs: int,
    next_batch: int,
    trajectory: list[dict[str, Any]],
    terminal_status: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "train-checkpoint/1",
        "coefficients": [float(value) for value in coefficients.tolist()],
        "intercept": float(intercept),
        "completed_epochs": int(completed_epochs),
        "next_batch": int(next_batch),
        "consumed_epochs": int(completed_epochs),
        "loss_trajectory": to_jsonable(trajectory),
        "terminal_status": terminal_status,
    }


def _raise_injected_fault(fault: dict[str, Any], stage: str, epoch: int) -> None:
    if not fault or fault.get("stage") != stage or int(fault.get("epoch", 1)) != int(epoch):
        return
    exception_name = str(fault.get("exception", "RuntimeError"))
    exception_type = {
        "RuntimeError": RuntimeError,
        "ValueError": ValueError,
        "ZeroDivisionError": ZeroDivisionError,
        "ArithmeticError": ArithmeticError,
    }.get(exception_name, RuntimeError)
    message = str(fault.get("message", f"injected {exception_name} in {stage}"))
    try:
        raise exception_type(message)
    except Exception as exc:
        raise TrainingFailure(
            code=E_MODEL_EXCEPTION,
            stage=stage,
            epoch=epoch,
            exception_type=type(exc).__name__,
            message=str(exc),
        ) from exc


def train_model(
    *,
    model_name: str,
    scheme: str,
    penalty: float,
    mix: float | None,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    hyperparameters: TrainingHyperparameters,
    quota: int,
    fault: dict[str, Any] | None = None,
    resume_state: dict[str, Any] | None = None,
    checkpoint_callback: CheckpointCallback | None = None,
    stop_event: threading.Event | None = None,
) -> dict[str, Any]:
    """Train one model and return its complete, JSON-serializable artifact."""

    if x_train.ndim != 2 or x_train.shape[0] != y_train.shape[0]:
        raise ValueError("x_train/y_train shape mismatch")
    if x_validation.ndim != 2 or x_validation.shape[1] != x_train.shape[1]:
        raise ValueError("validation/train feature shape mismatch")
    if x_train.shape[0] == 0 or x_validation.shape[0] == 0:
        raise ValueError("training and validation samples must be non-empty")
    if quota <= 0:
        raise TrainingFailure(
            code=E_MODEL_BUDGET_EXHAUSTED,
            stage="schedule",
            epoch=0,
            exception_type="BudgetExhausted",
            message="model received a zero training quota",
        )

    batches = split_batches(x_train.shape[0], hyperparameters.batch_size)
    batch_count = len(batches)
    coefficients = np.zeros(x_train.shape[1], dtype=np.float64)
    intercept = float(np.mean(y_train, dtype=np.float64))
    trajectory: list[dict[str, Any]] = []
    completed_epochs = 0
    start_batch = 0
    resumed = False
    resumed_terminal_status: str | None = None

    if resume_state:
        try:
            saved_coefficients = np.asarray(resume_state["coefficients"], dtype=np.float64)
            saved_trajectory = list(resume_state.get("loss_trajectory", []))
            saved_completed = int(resume_state.get("completed_epochs", 0))
            saved_next_batch = int(resume_state.get("next_batch", 0))
        except (KeyError, TypeError, ValueError) as exc:
            raise TrainingFailure(
                code=E_MODEL_EXCEPTION,
                stage="train",
                epoch=0,
                exception_type="CheckpointStateError",
                message=f"invalid resume checkpoint: {exc}",
            ) from exc
        if saved_coefficients.shape != coefficients.shape:
            raise TrainingFailure(
                code=E_MODEL_EXCEPTION,
                stage="train",
                epoch=0,
                exception_type="CheckpointStateError",
                message="checkpoint coefficient shape does not match the current features",
            )
        if saved_completed < 0 or saved_completed > hyperparameters.epochs:
            raise TrainingFailure(
                code=E_MODEL_EXCEPTION,
                stage="train",
                epoch=saved_completed,
                exception_type="CheckpointStateError",
                message="checkpoint epoch count is outside the configured range",
            )
        if saved_next_batch < 0 or saved_next_batch >= batch_count:
            if not (saved_next_batch == 0 and saved_completed == hyperparameters.epochs):
                raise TrainingFailure(
                    code=E_MODEL_EXCEPTION,
                    stage="train",
                    epoch=saved_completed,
                    exception_type="CheckpointStateError",
                    message="checkpoint batch position is outside the current epoch",
                )
        if len(saved_trajectory) != saved_completed:
            raise TrainingFailure(
                code=E_MODEL_EXCEPTION,
                stage="train",
                epoch=saved_completed,
                exception_type="CheckpointStateError",
                message="checkpoint trajectory length does not match completed epochs",
            )
        coefficients = saved_coefficients
        intercept = float(resume_state.get("intercept", intercept))
        trajectory = to_jsonable(saved_trajectory)
        completed_epochs = saved_completed
        start_batch = saved_next_batch
        resumed = True
        if resume_state.get("terminal_status") in {"converged", "max_epochs"}:
            resumed_terminal_status = str(resume_state["terminal_status"])

    if completed_epochs >= quota and completed_epochs < hyperparameters.epochs:
        raise TrainingFailure(
            code=E_MODEL_BUDGET_EXHAUSTED,
            stage="train",
            epoch=completed_epochs,
            exception_type="BudgetExhausted",
            message="model quota was already exhausted by the checkpoint",
        )
    def save_checkpoint(next_batch: int, reason: str, terminal_status: str | None = None) -> None:
        if checkpoint_callback is None:
            return
        state = _make_checkpoint_state(
            coefficients,
            intercept,
            completed_epochs,
            next_batch,
            trajectory,
            terminal_status,
        )
        state["reason"] = reason
        state["model"] = model_name
        checkpoint_callback(state, reason)

    if stop_event is not None and stop_event.is_set():
        save_checkpoint(start_batch, "interrupted")
        raise TrainingInterrupted(_make_checkpoint_state(coefficients, intercept, completed_epochs, start_batch, trajectory))

    n_train = float(x_train.shape[0])
    previous_coefficients = coefficients.copy()
    previous_intercept = intercept
    final_status = resumed_terminal_status or "max_epochs"
    regularization_scale = 1.0 / float(batch_count)

    for epoch_index in range(completed_epochs, hyperparameters.epochs if resumed_terminal_status is None else completed_epochs):
        epoch_number = epoch_index + 1
        if epoch_number <= quota:
            _raise_injected_fault(fault or {}, "gradient", epoch_number)

        initial_batch = start_batch if epoch_index == completed_epochs else 0
        for batch_index in range(initial_batch, batch_count):
            if stop_event is not None and stop_event.is_set():
                save_checkpoint(batch_index, "interrupted")
                raise TrainingInterrupted(
                    _make_checkpoint_state(coefficients, intercept, completed_epochs, batch_index, trajectory)
                )
            start, end = batches[batch_index]
            x_batch = x_train[start:end]
            y_batch = y_train[start:end]
            residual = _matvec(x_batch, coefficients) + intercept - y_batch
            data_gradient = np.sum(
                x_batch * residual.reshape(-1, 1), axis=0, dtype=np.float64
            ) / float(end - start)
            data_gradient = data_gradient.astype(np.float64, copy=False)
            _, ratios, _ = _regularization(scheme, penalty, mix, coefficients)
            l1_ratio, l2_ratio = float(ratios[0]), float(ratios[1])
            sign = np.asarray(
                [
                    _subgradient_sign(float(value), hyperparameters.l1_zero_subgradient)
                    for value in coefficients
                ],
                dtype=np.float64,
            )
            regularization_gradient = penalty * (
                l2_ratio * coefficients + l1_ratio * sign
            ) * regularization_scale
            gradient = data_gradient + regularization_gradient
            intercept_gradient = float(np.mean(residual, dtype=np.float64))
            coefficients = coefficients - hyperparameters.learning_rate * gradient
            intercept = intercept - hyperparameters.learning_rate * intercept_gradient
            coefficients[coefficients == 0.0] = 0.0

            if (batch_index + 1) % hyperparameters.snapshot_every_batches == 0 and batch_index + 1 < batch_count:
                save_checkpoint(batch_index + 1, "batch")

        predictions = _matvec(x_train, coefficients) + intercept
        residual = predictions - y_train
        data_loss = 0.5 * float(np.mean(residual * residual, dtype=np.float64))
        regularization_loss, _, norms = _regularization(scheme, penalty, mix, coefficients)
        loss = data_loss + regularization_loss
        digest = coefficient_digest(coefficients.tolist(), intercept)
        trajectory.append(
            {
                "epoch": epoch_number,
                "loss": loss,
                "data_loss": data_loss,
                "regularization_loss": regularization_loss,
                "coefficient_digest": digest,
                "nonzero_coefficients": int(np.count_nonzero(coefficients)),
                **norms,
            }
        )
        completed_epochs = epoch_number
        start_batch = 0
        max_coefficient_delta = float(np.max(np.abs(coefficients - previous_coefficients), initial=0.0))
        intercept_delta = abs(intercept - previous_intercept)
        loss_delta = abs(loss - trajectory[-2]["loss"]) if len(trajectory) > 1 else float("inf")
        previous_coefficients = coefficients.copy()
        previous_intercept = intercept
        terminal_now: str | None = None
        if (
            hyperparameters.tolerance >= 0.0
            and loss_delta <= hyperparameters.tolerance
            and max(max_coefficient_delta, intercept_delta) <= hyperparameters.tolerance
        ):
            final_status = "converged"
            terminal_now = final_status
        if terminal_now is None and completed_epochs >= hyperparameters.epochs:
            final_status = "max_epochs"
            terminal_now = final_status
        save_checkpoint(0, "epoch", terminal_now)
        if terminal_now is not None:
            break
        if completed_epochs >= quota:
            save_checkpoint(0, "budget")
            raise TrainingFailure(
                code=E_MODEL_BUDGET_EXHAUSTED,
                stage="train",
                epoch=completed_epochs,
                exception_type="BudgetExhausted",
                message="model quota exhausted before convergence",
            )
        if stop_event is not None and stop_event.is_set():
            save_checkpoint(0, "interrupted")
            raise TrainingInterrupted(
                _make_checkpoint_state(coefficients, intercept, completed_epochs, 0, trajectory)
            )

    if final_status == "max_epochs" and completed_epochs >= quota and quota < hyperparameters.epochs:
        # This branch is defensive; the in-loop check normally raises first.
        save_checkpoint(0, "budget")
        raise TrainingFailure(
            code=E_MODEL_BUDGET_EXHAUSTED,
            stage="train",
            epoch=completed_epochs,
            exception_type="BudgetExhausted",
            message="model quota exhausted before convergence",
        )

    metrics = _validation_metrics(x_validation, y_validation, coefficients, intercept)
    result: dict[str, Any] = {
        "model": model_name,
        "scheme": scheme,
        "penalty": float(penalty),
        "mix": mix,
        "coefficients": [float(value) for value in coefficients.tolist()],
        "intercept": float(intercept),
        "convergence_status": final_status,
        "precision_label": "float64",
        "loss_trajectory": trajectory,
        "coefficient_snapshots": [
            {
                "epoch": item["epoch"],
                "coefficient_digest": item["coefficient_digest"],
                "nonzero_coefficients": item["nonzero_coefficients"],
            }
            for item in trajectory
        ],
        "validation_metrics": metrics,
        "epochs_consumed": int(completed_epochs),
        "batches_per_epoch": batch_count,
        "batch_size": hyperparameters.batch_size,
        "learning_rate": hyperparameters.learning_rate,
        "tolerance": hyperparameters.tolerance,
        "l1_zero_subgradient": hyperparameters.l1_zero_subgradient,
    }
    return to_jsonable(result)
