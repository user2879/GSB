from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .errors import E_DATA, DiagnosticError


OUTPUT_DIM = 4


@dataclass(frozen=True)
class ModelParams:
    weights: tuple[np.ndarray, ...]
    biases: tuple[np.ndarray, ...]

    @property
    def layer_sizes(self) -> tuple[int, ...]:
        sizes = [int(self.weights[0].shape[0])]
        sizes.extend(int(weight.shape[1]) for weight in self.weights)
        return tuple(sizes)

    @property
    def parameter_count(self) -> int:
        return sum(int(array.size) for array in (*self.weights, *self.biases))

    def named_arrays(self) -> Iterable[tuple[str, np.ndarray]]:
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            yield f"layers[{index}].weight", weight
            yield f"layers[{index}].bias", bias


@dataclass(frozen=True)
class Dataset:
    inputs: np.ndarray

    @property
    def sample_count(self) -> int:
        return int(self.inputs.shape[0])

    @property
    def input_dim(self) -> int:
        return int(self.inputs.shape[1])


@dataclass(frozen=True)
class ArtifactSet:
    seed: int
    dataset: Dataset
    model: ModelParams
    data_sha256: str
    params_sha256: str
    data_summary: dict[str, Any]
    params_summary: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "dataset": self.data_summary,
            "parameters": self.params_summary,
            "data_sha256": self.data_sha256,
            "params_sha256": self.params_sha256,
        }


def _canonical_array_bytes(array: np.ndarray) -> bytes:
    if np.issubdtype(array.dtype, np.floating):
        normalized = np.ascontiguousarray(array, dtype="<f4")
    elif np.issubdtype(array.dtype, np.signedinteger):
        normalized = np.ascontiguousarray(array, dtype="<i8")
    else:
        raise TypeError(f"unsupported deterministic dtype: {array.dtype}")
    return normalized.tobytes(order="C")


def digest_named_arrays(named_arrays: Iterable[tuple[str, np.ndarray]]) -> str:
    digest = hashlib.sha256()
    for name, array in sorted(named_arrays, key=lambda item: item[0]):
        header = {
            "name": name,
            "dtype": "float32" if np.issubdtype(array.dtype, np.floating) else "int64",
            "shape": list(array.shape),
        }
        digest.update(json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\0")
        digest.update(_canonical_array_bytes(array))
        digest.update(b"\0")
    return digest.hexdigest()


def _array_summary(name: str, array: np.ndarray) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(array)
    return {
        "name": name,
        "dtype": "float32",
        "shape": list(contiguous.shape),
        "items": int(contiguous.size),
        "sha256": hashlib.sha256(_canonical_array_bytes(contiguous)).hexdigest(),
        "min": float(np.min(contiguous)),
        "max": float(np.max(contiguous)),
        "sample_c_order": [float(value) for value in contiguous.reshape(-1)[:8]],
    }


def generate_artifacts(
    seed: int,
    sample_count: int,
    input_dim: int,
    hidden_dim: int,
) -> ArtifactSet:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise DiagnosticError(E_DATA, "seed must be an integer", "seed", {"received": seed})
    if sample_count < 1 or input_dim < 1 or hidden_dim < 1:
        raise DiagnosticError(
            E_DATA,
            "sample_count, input_dim and hidden_dim must be positive integers",
            "generation",
            {"sample_count": sample_count, "input_dim": input_dim, "hidden_dim": hidden_dim},
        )

    rng = np.random.Generator(np.random.PCG64(seed))
    inputs = rng.standard_normal((sample_count, input_dim)).astype(np.float32)

    sizes = (input_dim, hidden_dim, hidden_dim, OUTPUT_DIM)
    weights: list[np.ndarray] = []
    biases: list[np.ndarray] = []
    for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
        limit = np.float64(np.sqrt(6.0 / (fan_in + fan_out)))
        weight = rng.uniform(-limit, limit, size=(fan_in, fan_out)).astype(np.float32)
        bias = rng.uniform(-limit, limit, size=(fan_out,)).astype(np.float32)
        weights.append(weight)
        biases.append(bias)

    model = ModelParams(tuple(weights), tuple(biases))
    dataset = Dataset(inputs)
    data_sha = digest_named_arrays([("inputs", inputs)])
    params_sha = digest_named_arrays(model.named_arrays())
    data_summary = _array_summary("inputs", inputs)
    data_summary["digest_scope"] = "canonical-IEEE754-little-endian-float32+C-order"
    params_summary = {
        "layer_sizes": list(model.layer_sizes),
        "parameter_count": model.parameter_count,
        "dtype": "float32",
        "sha256": params_sha,
        "arrays": [_array_summary(name, array) for name, array in model.named_arrays()],
        "digest_scope": "names+dtype+shape+canonical-IEEE754-little-endian-float32+C-order",
    }
    return ArtifactSet(
        seed=seed,
        dataset=dataset,
        model=model,
        data_sha256=data_sha,
        params_sha256=params_sha,
        data_summary=data_summary,
        params_summary=params_summary,
    )


def load_expected_manifest(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8-sig"))
    except OSError as exc:
        raise DiagnosticError(E_DATA, f"cannot read digest manifest: {exc}", "$", {"source": str(source)}) from exc
    except json.JSONDecodeError as exc:
        raise DiagnosticError(
            E_DATA,
            "digest manifest is not valid JSON",
            f"$[line {exc.lineno}, column {exc.colno}]",
            {"source": str(source)},
        ) from exc
    if not isinstance(payload, dict):
        raise DiagnosticError(E_DATA, "digest manifest root must be an object", "$", {"source": str(source)})
    return payload


def verify_artifacts(artifacts: ArtifactSet, expected: dict[str, Any] | None) -> dict[str, Any]:
    if expected is None:
        return {
            "checked": False,
            "status": "NOT_REQUESTED",
            "data_sha256": artifacts.data_sha256,
            "params_sha256": artifacts.params_sha256,
        }

    expected_data = expected.get("data_sha256")
    expected_params = expected.get("params_sha256")
    mismatches: list[dict[str, Any]] = []
    if expected_data is None:
        mismatches.append(
            {
                "artifact": "dataset",
                "field_path": "dataset.inputs",
                "manifest_path": "$.data_sha256",
                "expected": "required 64-character SHA-256",
                "actual": artifacts.data_sha256,
            }
        )
    elif expected_data != artifacts.data_sha256:
        mismatches.append(
            {
                "artifact": "dataset",
                "field_path": "dataset.inputs",
                "manifest_path": "$.data_sha256",
                "expected": expected_data,
                "actual": artifacts.data_sha256,
            }
        )
    if expected_params is None:
        mismatches.append(
            {
                "artifact": "parameters",
                "field_path": "model.layers[*].weight/model.layers[*].bias",
                "manifest_path": "$.params_sha256",
                "expected": "required 64-character SHA-256",
                "actual": artifacts.params_sha256,
            }
        )
    elif expected_params != artifacts.params_sha256:
        mismatches.append(
            {
                "artifact": "parameters",
                "field_path": "model.layers[*].weight/model.layers[*].bias",
                "manifest_path": "$.params_sha256",
                "expected": expected_params,
                "actual": artifacts.params_sha256,
            }
        )
    if mismatches:
        raise DiagnosticError(
            E_DATA,
            "deterministic artifact digest mismatch; inference was not started",
            mismatches[0]["manifest_path"],
            {"mismatches": mismatches},
        )
    return {
        "checked": True,
        "status": "MATCH",
        "data_sha256": artifacts.data_sha256,
        "params_sha256": artifacts.params_sha256,
    }


def write_manifest(artifacts: ArtifactSet, path: str | Path) -> None:
    payload = {
        "schema_version": 1,
        "seed": artifacts.seed,
        "sample_count": artifacts.dataset.sample_count,
        "input_dim": artifacts.dataset.input_dim,
        "hidden_dim": int(artifacts.model.weights[0].shape[1]),
        "data_sha256": artifacts.data_sha256,
        "params_sha256": artifacts.params_sha256,
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
