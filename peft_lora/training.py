from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np

from .base import BasePackage, FINGERPRINT_ALGORITHM, LayerWeights
from .canonical import atomic_write_json, sha256_object
from .config import PeftConfig
from .errors import E_TRAIN_FAILED, BusinessError
from .samples import InstructionSample, encode_sample
from .tasks import TaskSpec


ADAPTER_FORMAT = "peft-lora-adapter-v1"
TRAINER_VERSION = "deterministic-sgd-v1"


def _adapter_content_hash(payload: dict[str, Any]) -> str:
    content = {key: value for key, value in payload.items() if key != "content_hash"}
    return sha256_object(content)


def _downcast_list(array: np.ndarray[Any, Any]) -> list[list[float]]:
    return [[float(value) for value in row] for row in array.tolist()]


def train_adapter(
    task: TaskSpec,
    layer: LayerWeights,
    samples: list[InstructionSample],
    config: PeftConfig,
    base_fingerprint: str,
    target_layer_fingerprint: str,
) -> dict[str, Any]:
    if not samples:
        raise BusinessError(
            E_TRAIN_FAILED,
            "no instruction samples target this layer",
            task_name=task.task_name,
            target_layer=task.target_layer,
        )
    output_dim, input_dim = (int(layer.weight.shape[0]), int(layer.weight.shape[1]))
    encoded = [
        encode_sample(sample, task.task_name, input_dim, output_dim) for sample in samples
    ]
    residuals = [
        target - (layer.weight @ features + layer.bias) for features, target in encoded
    ]
    init_seed = int.from_bytes(
        hashlib.sha256(
            f"adapter-init-v1|{task.stable_key}|{task.rank}|{input_dim}|{output_dim}".encode(
                "utf-8"
            )
        ).digest()[:8],
        "little",
    )
    rng = np.random.Generator(np.random.PCG64(init_seed))
    matrix_a = rng.normal(0.0, 0.08, size=(task.rank, input_dim)).astype(np.float64)
    matrix_b = np.zeros((output_dim, task.rank), dtype=np.float64)
    loss_trajectory: list[float] = []

    def loss_for_current_state() -> float:
        values: list[float] = []
        for (features, _), residual in zip(encoded, residuals, strict=True):
            prediction = task.scale * (matrix_b @ (matrix_a @ features))
            error = prediction - residual
            values.append(0.5 * float(np.dot(error, error)))
        return float(sum(values) / len(values))

    batch_size = config.batch_size
    sample_pairs = list(zip(encoded, residuals, strict=True))
    for _epoch in range(config.epochs):
        loss_trajectory.append(loss_for_current_state())
        for batch_start in range(0, len(encoded), batch_size):
            batch = sample_pairs[batch_start : batch_start + batch_size]
            gradient_a = np.zeros_like(matrix_a)
            gradient_b = np.zeros_like(matrix_b)
            for (features, _target), residual in batch:
                hidden = matrix_a @ features
                prediction = matrix_b @ hidden
                error = task.scale * prediction - residual
                gradient_prediction = task.scale * error
                gradient_b += np.outer(gradient_prediction, hidden)
                gradient_hidden = matrix_b.T @ gradient_prediction
                gradient_a += np.outer(gradient_hidden, features)
            divisor = float(len(batch))
            matrix_a -= config.learning_rate * gradient_a / divisor
            matrix_b -= config.learning_rate * gradient_b / divisor
            if not np.isfinite(matrix_a).all() or not np.isfinite(matrix_b).all():
                raise BusinessError(
                    E_TRAIN_FAILED,
                    "training diverged to non-finite values",
                    task_name=task.task_name,
                    target_layer=task.target_layer,
                )
    final_loss = loss_for_current_state()
    if not math.isfinite(final_loss):
        raise BusinessError(
            E_TRAIN_FAILED,
            "training produced a non-finite loss",
            task_name=task.task_name,
            target_layer=task.target_layer,
        )

    payload: dict[str, Any] = {
        "format": ADAPTER_FORMAT,
        "trainer_version": TRAINER_VERSION,
        "fingerprint_algorithm": FINGERPRINT_ALGORITHM,
        "task_name": task.task_name,
        "target_layer": task.target_layer,
        "rank": task.rank,
        "scale": task.scale,
        "stable_key": task.stable_key,
        "base_fingerprint": base_fingerprint,
        "target_layer_fingerprint": target_layer_fingerprint,
        "shape": {"input": input_dim, "output": output_dim},
        "learning_rate": config.learning_rate,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "sample_ids": [sample.sample_id for sample in samples],
        "loss_trajectory": loss_trajectory,
        "final_loss": final_loss,
        "A": _downcast_list(matrix_a),
        "B": _downcast_list(matrix_b),
    }
    payload["content_hash"] = _adapter_content_hash(payload)
    return payload


def validate_adapter_payload(
    payload: Any,
    *,
    expected_task: TaskSpec | None = None,
    expected_layer_shape: tuple[int, int] | None = None,
) -> dict[str, Any]:
    def invalid(message: str) -> BusinessError:
        task_name = payload.get("task_name") if isinstance(payload, dict) else None
        target_layer = payload.get("target_layer") if isinstance(payload, dict) else None
        return BusinessError(
            "E_ADAPTER_CONFIG",
            message,
            task_name=task_name,
            target_layer=target_layer,
        )

    if not isinstance(payload, dict) or payload.get("format") != ADAPTER_FORMAT:
        raise invalid("adapter format is invalid")
    if payload.get("trainer_version") != TRAINER_VERSION:
        raise invalid("adapter trainer version is unsupported")
    if not isinstance(payload.get("task_name"), str) or not payload["task_name"]:
        raise invalid("adapter task_name is invalid")
    if not isinstance(payload.get("target_layer"), str) or not payload["target_layer"]:
        raise invalid("adapter target_layer is invalid")
    rank = payload.get("rank")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise invalid("adapter rank is invalid")
    scale = payload.get("scale")
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or float(scale) <= 0:
        raise invalid("adapter scale is invalid")
    if not isinstance(payload.get("base_fingerprint"), str):
        raise invalid("adapter base fingerprint is missing")
    if not isinstance(payload.get("target_layer_fingerprint"), str):
        raise invalid("adapter target layer fingerprint is missing")
    losses = payload.get("loss_trajectory")
    if (
        not isinstance(losses, list)
        or not losses
        or any(
            isinstance(loss, bool)
            or not isinstance(loss, (int, float))
            or not math.isfinite(float(loss))
            for loss in losses
        )
    ):
        raise invalid("adapter loss trajectory is invalid")
    shape = payload.get("shape")
    if (
        not isinstance(shape, dict)
        or any(
            isinstance(shape.get(key), bool)
            or not isinstance(shape.get(key), int)
            or shape.get(key) <= 0
            for key in ("input", "output")
        )
    ):
        raise invalid("adapter shape is invalid")
    input_dim = int(shape["input"])
    output_dim = int(shape["output"])
    if expected_layer_shape is not None and (output_dim, input_dim) != expected_layer_shape:
        raise invalid("adapter shape does not match the target layer")
    if expected_task is not None:
        if (
            payload.get("task_name") != expected_task.task_name
            or payload.get("target_layer") != expected_task.target_layer
            or payload.get("rank") != expected_task.rank
            or float(payload.get("scale")) != expected_task.scale
        ):
            raise invalid("adapter metadata does not match the configured task")
    try:
        matrix_a = np.asarray(payload.get("A"), dtype=np.float64)
        matrix_b = np.asarray(payload.get("B"), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise invalid(f"adapter matrices cannot be parsed: {exc}") from exc
    if matrix_a.shape != (rank, input_dim) or matrix_b.shape != (output_dim, rank):
        raise invalid("adapter matrix shapes are invalid")
    if not np.isfinite(matrix_a).all() or not np.isfinite(matrix_b).all():
        raise invalid("adapter matrices contain non-finite values")
    expected_hash = _adapter_content_hash(payload)
    if payload.get("content_hash") != expected_hash:
        raise invalid("adapter content hash does not match its content")
    return payload


def save_adapter(path: str | Path, payload: dict[str, Any]) -> None:
    validate_adapter_payload(payload)
    atomic_write_json(path, payload)


def adapter_delta(payload: dict[str, Any]) -> np.ndarray[Any, Any]:
    matrix_a = np.asarray(payload["A"], dtype=np.float64)
    matrix_b = np.asarray(payload["B"], dtype=np.float64)
    return float(payload["scale"]) * (matrix_b @ matrix_a)
