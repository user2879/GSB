from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import InteropConfig, step_id


DOMAIN = b"ckptmeta-deterministic-v1\0"


def _rng_for(seed: int, *salts: object) -> np.random.Generator:
    label = "|".join([str(seed), *[str(item) for item in salts]]).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(label).digest()[:16], "little")
    return np.random.Generator(np.random.PCG64(value))


def _update_bytes(handle: Any, data: bytes) -> None:
    handle.update(struct.pack("<Q", len(data)))
    handle.update(data)


def _float64_bytes(array: np.ndarray) -> bytes:
    canonical = np.ascontiguousarray(array, dtype="<f8")
    return canonical.tobytes(order="C")


def _piece_digest(
    kind: str,
    config: InteropConfig,
    step: int,
    shard_index: int,
    pieces: list[tuple[str, np.ndarray]],
) -> str:
    handle = hashlib.sha256()
    handle.update(DOMAIN)
    _update_bytes(handle, kind.encode("ascii"))
    _update_bytes(handle, config.partition_mode.encode("ascii"))
    handle.update(struct.pack("<QQQQ", config.seed, config.shard_count, step, shard_index))
    for name, piece in pieces:
        _update_bytes(handle, name.encode("ascii"))
        handle.update(struct.pack("<Q", piece.size))
        _update_bytes(handle, _float64_bytes(piece))
    return handle.hexdigest()


def _rng_state(seed: int, step: int, rng: np.random.Generator) -> dict[str, Any]:
    state = rng.bit_generator.state
    inner = state["state"]
    return {
        "algorithm": "PCG64",
        "seed": seed,
        "state": f"{int(inner['state']):032x}",
        "inc": f"{int(inner['inc']):032x}",
        "has_uint32": int(state["has_uint32"]),
        "uinteger": int(state["uinteger"]),
        "draw_step": step,
    }


def _dataset(config: InteropConfig) -> tuple[np.ndarray, np.ndarray]:
    rng = _rng_for(config.seed, "classification-data")
    features = rng.random((config.sample_count, config.input_dim), dtype=np.float64)
    labels = rng.integers(0, config.classes, size=config.sample_count, dtype=np.int64)
    return features, labels


def shard_indices(config: InteropConfig, shard_index: int) -> np.ndarray:
    indices = np.arange(config.sample_count, dtype=np.int64)
    if config.partition_mode == "modulo":
        return indices[indices % config.shard_count == shard_index]
    start = (config.sample_count * shard_index) // config.shard_count
    stop = (config.sample_count * (shard_index + 1)) // config.shard_count
    return indices[start:stop]


def _parameters(config: InteropConfig, step: int) -> dict[str, np.ndarray]:
    rng = _rng_for(config.seed, "mlp-parameters", step)
    return {
        "w1": rng.random((config.input_dim, config.hidden_dim), dtype=np.float64) - 0.5,
        "b1": rng.random((config.hidden_dim,), dtype=np.float64) - 0.5,
        "w2": rng.random((config.hidden_dim, config.classes), dtype=np.float64) - 0.5,
        "b2": rng.random((config.classes,), dtype=np.float64) - 0.5,
    }


def _momentum(config: InteropConfig, step: int) -> dict[str, np.ndarray]:
    rng = _rng_for(config.seed, "mlp-momentum", step)
    return {
        "w1": rng.random((config.input_dim, config.hidden_dim), dtype=np.float64) * 0.01,
        "b1": rng.random((config.hidden_dim,), dtype=np.float64) * 0.01,
        "w2": rng.random((config.hidden_dim, config.classes), dtype=np.float64) * 0.01,
        "b2": rng.random((config.classes,), dtype=np.float64) * 0.01,
    }


def _shard_pieces(array: np.ndarray, shard_index: int, shard_count: int) -> np.ndarray:
    flat = np.ascontiguousarray(array, dtype="<f8").reshape(-1)
    start = (flat.size * shard_index) // shard_count
    stop = (flat.size * (shard_index + 1)) // shard_count
    return flat[start:stop]


@dataclass(frozen=True)
class MaterializedContent:
    features: np.ndarray
    labels: np.ndarray
    parameters: dict[str, np.ndarray]
    momentum: dict[str, np.ndarray]


def materialize_content(config: InteropConfig, step: int) -> MaterializedContent:
    features, labels = _dataset(config)
    return MaterializedContent(
        features=features,
        labels=labels,
        parameters=_parameters(config, step),
        momentum=_momentum(config, step),
    )


def build_step_core(config: InteropConfig, step: int) -> dict[str, Any]:
    content = materialize_content(config, step)
    rng = _rng_for(config.seed, "cursor-state", step)
    rng.random(17 * config.shard_count, dtype=np.float64)
    rng_state = _rng_state(config.seed, step, rng)

    data_indices = [shard_indices(config, index) for index in range(config.shard_count)]
    shards: list[dict[str, Any]] = []
    for shard_index in range(config.shard_count):
        parameter_pieces = [
            (name, _shard_pieces(values, shard_index, config.shard_count))
            for name, values in sorted(content.parameters.items())
        ]
        momentum_pieces = [
            (name, _shard_pieces(values, shard_index, config.shard_count))
            for name, values in sorted(content.momentum.items())
        ]
        indices = data_indices[shard_index]
        steps_per_epoch = max(1, (config.sample_count + config.shard_count - 1) // config.shard_count)
        offset = (step * 11 + shard_index * 3) % max(1, indices.size)
        shards.append(
            {
                "shard_index": shard_index,
                "parameter_digest": _piece_digest(
                    "parameter", config, step, shard_index, parameter_pieces
                ),
                "momentum_digest": _piece_digest(
                    "momentum", config, step, shard_index, momentum_pieces
                ),
                "data_cursor": {
                    "epoch": step // steps_per_epoch,
                    "offset": int(offset),
                    "next_sample_index": int(indices[offset]),
                },
            }
        )

    return {
        "manifest_id": step_id(step),
        "global_step": step,
        "shard_count": config.shard_count,
        "partition_mode": config.partition_mode,
        "shards": shards,
        "rng_state": rng_state,
    }


def direct_v1_record(core: dict[str, Any]) -> dict[str, Any]:
    return {
        "format_version": 1,
        "global_step": core["global_step"],
        "shard_count": core["shard_count"],
        "shards": [
            {
                "shard_index": shard["shard_index"],
                "parameter_digest": shard["parameter_digest"],
            }
            for shard in core["shards"]
        ],
    }


def direct_v2_record(core: dict[str, Any]) -> dict[str, Any]:
    return {
        "format_version": 2,
        "global_step": core["global_step"],
        "shard_count": core["shard_count"],
        "rng_state": core["rng_state"],
        "shards": [
            {
                "shard_index": shard["shard_index"],
                "parameter_digest": shard["parameter_digest"],
                "momentum_digest": shard["momentum_digest"],
            }
            for shard in core["shards"]
        ],
    }
