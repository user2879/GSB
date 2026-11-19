"""Deterministic local sample and model construction."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .constants import DEFAULT_OUTPUT_DIM, MODEL_ARRAY_ORDER


@dataclass(frozen=True)
class GeneratedAssets:
    sample_ids: tuple[str, ...]
    inputs: np.ndarray
    model: Mapping[str, np.ndarray]
    config: Mapping[str, Any]


def normalize_seed(seed: int) -> int:
    return int(seed) & 0xFFFFFFFF


def _make_sample_ids(sample_count: int) -> tuple[str, ...]:
    width = max(4, len(str(max(sample_count - 1, 0))))
    return tuple(f"S{i:0{width}d}" for i in range(sample_count))


def generate_assets(
    *,
    seed: int,
    sample_count: int,
    input_dim: int,
    hidden_dim: int,
    output_dim: int = DEFAULT_OUTPUT_DIM,
) -> GeneratedAssets:
    """Generate bit-identical assets for an identical argument tuple.

    NumPy's legacy ``RandomState`` stream is used deliberately.  Each tensor has
    its own derived stream so changing one dimension does not reshuffle unrelated
    tensors.  All persisted tensors are float64.
    """

    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if input_dim <= 0 or hidden_dim <= 0 or output_dim <= 0:
        raise ValueError("input_dim, hidden_dim, and output_dim must be positive")

    seed32 = normalize_seed(seed)
    sample_ids = _make_sample_ids(sample_count)

    input_rng = np.random.RandomState(seed32 ^ 0xA5A5A5A5)
    inputs = np.ascontiguousarray(
        input_rng.uniform(-1.0, 1.0, size=(sample_count, input_dim)), dtype="<f8"
    )

    def weights(stream_tag: int, shape: tuple[int, ...]) -> np.ndarray:
        rng = np.random.RandomState((seed32 + stream_tag) & 0xFFFFFFFF)
        fan_in = shape[0]
        limit = 1.0 / math.sqrt(float(fan_in))
        return np.ascontiguousarray(rng.uniform(-limit, limit, size=shape), dtype="<f8")

    def biases(stream_tag: int, shape: tuple[int, ...]) -> np.ndarray:
        rng = np.random.RandomState((seed32 + stream_tag) & 0xFFFFFFFF)
        return np.ascontiguousarray(rng.uniform(-0.05, 0.05, size=shape), dtype="<f8")

    model = {
        "W1": weights(0x101, (input_dim, hidden_dim)),
        "b1": biases(0x102, (hidden_dim,)),
        "W2": weights(0x201, (hidden_dim, hidden_dim)),
        "b2": biases(0x202, (hidden_dim,)),
        "W3": weights(0x301, (hidden_dim, output_dim)),
        "b3": biases(0x302, (output_dim,)),
    }
    config = {
        "activation": "tanh",
        "hidden_dim": int(hidden_dim),
        "hidden_layers": 2,
        "input_dim": int(input_dim),
        "model_dtype": "<f8",
        "output_dim": int(output_dim),
        "sample_count": int(sample_count),
        "seed": int(seed),
        "seed_normalized": seed32,
        "tensor_order": list(MODEL_ARRAY_ORDER),
    }
    return GeneratedAssets(
        sample_ids=sample_ids,
        inputs=inputs,
        model=model,
        config=config,
    )


def save_assets(assets: GeneratedAssets, dataset_path: Path, model_path: Path) -> None:
    dataset_path = Path(dataset_path)
    model_path = Path(model_path)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        dataset_path,
        sample_ids=np.asarray(assets.sample_ids, dtype=np.str_),
        inputs=assets.inputs,
        config_json=np.asarray(json.dumps(assets.config, sort_keys=True, ensure_ascii=False)),
    )
    arrays = {name: assets.model[name] for name in MODEL_ARRAY_ORDER}
    np.savez(
        model_path,
        config_json=np.asarray(json.dumps(assets.config, sort_keys=True, ensure_ascii=False)),
        **arrays,
    )


def load_assets(dataset_path: Path, model_path: Path) -> GeneratedAssets:
    dataset_path = Path(dataset_path)
    model_path = Path(model_path)
    with np.load(dataset_path, allow_pickle=False) as dataset:
        sample_ids = tuple(str(item) for item in dataset["sample_ids"].tolist())
        inputs = np.ascontiguousarray(dataset["inputs"], dtype="<f8")
        dataset_config = json.loads(str(dataset["config_json"].item()))
    with np.load(model_path, allow_pickle=False) as model_file:
        model = {
            name: np.ascontiguousarray(model_file[name], dtype="<f8")
            for name in MODEL_ARRAY_ORDER
        }
        model_config = json.loads(str(model_file["config_json"].item()))

    if dataset_config != model_config:
        raise ValueError("dataset and model config_json differ")
    if len(sample_ids) != inputs.shape[0]:
        raise ValueError("sample id count does not match input matrix rows")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample ids are not unique")
    _validate_shapes(sample_ids, inputs, model, dataset_config)
    return GeneratedAssets(
        sample_ids=sample_ids,
        inputs=inputs,
        model=model,
        config=dataset_config,
    )


def _validate_shapes(
    sample_ids: tuple[str, ...],
    inputs: np.ndarray,
    model: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
) -> None:
    input_dim = int(config["input_dim"])
    hidden_dim = int(config["hidden_dim"])
    output_dim = int(config["output_dim"])
    expected = {
        "W1": (input_dim, hidden_dim),
        "b1": (hidden_dim,),
        "W2": (hidden_dim, hidden_dim),
        "b2": (hidden_dim,),
        "W3": (hidden_dim, output_dim),
        "b3": (output_dim,),
    }
    if inputs.shape != (len(sample_ids), input_dim):
        raise ValueError("input matrix shape does not match config")
    for name in MODEL_ARRAY_ORDER:
        if model[name].shape != expected[name]:
            raise ValueError(f"model tensor {name} has shape {model[name].shape}, expected {expected[name]}")
