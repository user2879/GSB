from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .canonical import sha256_value
from .errors import LabError
from .specs import QUANT_PROFILES, BackendSpec, SampleSpec


MASK64 = (1 << 64) - 1


class SplitMix64:
    """Small, explicitly specified PRNG with stable results across Python versions."""

    def __init__(self, seed: int) -> None:
        self.state = seed & MASK64

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return (z ^ (z >> 31)) & MASK64

    def uniform(self) -> float:
        return ((self.next_u64() >> 40) / float(1 << 24)) * 2.0 - 1.0


def deterministic_vector(seed: int, size: int) -> np.ndarray:
    rng = SplitMix64(seed)
    return np.asarray([rng.uniform() for _ in range(size)], dtype=np.float64)


def deterministic_matrix(seed: int, rows: int, columns: int) -> np.ndarray:
    rng = SplitMix64(seed)
    values = [rng.uniform() for _ in range(rows * columns)]
    return np.asarray(values, dtype=np.float64).reshape(rows, columns)


def _quantize_matrix(matrix: np.ndarray, profile: str) -> tuple[np.ndarray | None, np.ndarray]:
    bits = QUANT_PROFILES[profile]["bits"]
    if bits is None:
        # The fp32 profile rounds storage to float32, then computes in float64.
        stored = matrix.astype(np.float32).astype(np.float64)
        return None, stored
    qmax = (1 << (bits - 1)) - 1
    maximum = float(np.max(np.abs(matrix))) if matrix.size else 0.0
    if maximum == 0.0:
        scale = 1.0
        quantized = np.zeros(matrix.shape, dtype=np.int64)
    else:
        scale = maximum / float(qmax)
        quantized = np.rint(matrix / scale).astype(np.int64)
        quantized = np.clip(quantized, -qmax, qmax)
    dequantized = quantized.astype(np.float64) * scale
    return quantized, dequantized


def _array_digest(array: np.ndarray) -> str:
    little_endian = np.asarray(array, dtype="<f8", order="C")
    return sha256_value(
        {
            "dtype": "float64-le",
            "shape": list(little_endian.shape),
            "hex": little_endian.tobytes().hex(),
        }
    )


@dataclass(frozen=True)
class BackendParameters:
    backend_id: str
    version: int
    model_dim: int
    hidden_dim: int
    quant_profile: str
    seed: int
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    weights_digest: str
    param_digest: str

    def summary(self) -> dict[str, Any]:
        return {
            "backend_id": self.backend_id,
            "version": self.version,
            "model_dim": self.model_dim,
            "hidden_dim": self.hidden_dim,
            "quant_profile": self.quant_profile,
            "quant_profile_version": QUANT_PROFILES[self.quant_profile]["version"],
            "seed": self.seed,
            "weights_digest": self.weights_digest,
            "param_digest": self.param_digest,
        }


def build_backend_parameters(spec: BackendSpec, version: int = 1) -> BackendParameters:
    if version <= 0:
        raise LabError("E_RUNTIME", "parameter version must be positive", field="version")
    w1_raw = deterministic_matrix(spec.seed ^ 0xA0761D6478BD642F, spec.model_dim, spec.hidden_dim)
    w2_raw = deterministic_matrix(spec.seed ^ 0xE7037ED1A0B428DB, spec.hidden_dim, spec.model_dim)
    q_w1, w1 = _quantize_matrix(w1_raw, spec.quant_profile)
    q_w2, w2 = _quantize_matrix(w2_raw, spec.quant_profile)
    rng = SplitMix64(spec.seed ^ 0x8EBC6AF09C88C6E3)
    b1 = np.asarray([rng.uniform() * 0.1 for _ in range(spec.hidden_dim)], dtype=np.float64)
    b2 = np.asarray([rng.uniform() * 0.1 for _ in range(spec.model_dim)], dtype=np.float64)
    weight_digest = sha256_value(
        {
            "w1_quant": q_w1.tolist() if q_w1 is not None else None,
            "w2_quant": q_w2.tolist() if q_w2 is not None else None,
            "w1": _array_digest(w1),
            "w2": _array_digest(w2),
            "b1": _array_digest(b1),
            "b2": _array_digest(b2),
            "quant_profile": spec.quant_profile,
            "quant_profile_version": QUANT_PROFILES[spec.quant_profile]["version"],
        }
    )
    digest_preimage = {
        "algorithm": "gray-merge-lab-params-v1",
        "backend_id": spec.backend_id,
        "version": version,
        "model_dim": spec.model_dim,
        "hidden_dim": spec.hidden_dim,
        "seed": spec.seed,
        "quant_profile": spec.quant_profile,
        "quant_profile_version": QUANT_PROFILES[spec.quant_profile]["version"],
        "weights_digest": weight_digest,
    }
    param_digest = sha256_value(digest_preimage)
    return BackendParameters(
        backend_id=spec.backend_id,
        version=version,
        model_dim=spec.model_dim,
        hidden_dim=spec.hidden_dim,
        quant_profile=spec.quant_profile,
        seed=spec.seed,
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        weights_digest=weight_digest,
        param_digest=param_digest,
    )


@dataclass(frozen=True)
class InputTensor:
    sample_id: str
    values: np.ndarray
    digest: str


def build_input_tensor(sample: SampleSpec, model_dim: int) -> InputTensor:
    values = deterministic_vector(sample.input_seed ^ 0xD1B54A32D192ED03, model_dim)
    digest = sha256_value(
        {
            "algorithm": "gray-merge-lab-input-v1",
            "sample_id": sample.sample_id,
            "input_seed": sample.input_seed,
            "model_dim": model_dim,
            "values": _array_digest(values),
        }
    )
    return InputTensor(sample.sample_id, values, digest)


def forward_fixed(
    params: BackendParameters,
    inputs: list[InputTensor],
) -> tuple[str, list[dict[str, Any]], list[int]]:
    if not inputs:
        raise LabError("E_RUNTIME", "cannot run an empty inference batch")
    for item in inputs:
        if item.values.shape != (params.model_dim,):
            raise LabError(
                "E_RUNTIME",
                f"input width {item.values.shape} does not match model_dim {params.model_dim}",
                field=f"inputs.{item.sample_id}",
            )
    matrix = np.vstack([item.values for item in inputs])
    hidden = np.maximum(matrix @ params.w1 + params.b1, 0.0)
    logits = hidden @ params.w2 + params.b2
    quantized = np.rint(logits * 1_000_000.0).astype("<i8", copy=False)
    output_digest = sha256_value(
        {
            "algorithm": "gray-merge-lab-output-v1",
            "backend_id": params.backend_id,
            "param_digest": params.param_digest,
            "sample_ids": [item.sample_id for item in inputs],
            "input_digests": [item.digest for item in inputs],
            "output_scale": 1_000_000,
            "outputs_i64": quantized.tolist(),
        }
    )
    summaries = [
        {
            "sample_id": item.sample_id,
            "output_digest": sha256_value(
                {
                    "algorithm": "gray-merge-lab-output-row-v1",
                    "sample_id": item.sample_id,
                    "values_i64": row.tolist(),
                }
            ),
        }
        for item, row in zip(inputs, quantized)
    ]
    return output_digest, summaries, [int(row[0]) for row in quantized]


def stable_route_bucket(rule_id: str, client_key: str) -> int:
    digest = sha256_value(
        {
            "algorithm": "gray-merge-lab-route-v1",
            "rule_id": rule_id,
            "client_key": client_key,
        }
    )
    # The domain is the documented 1..10000 slice of 0..10000. This makes
    # 0 bp mean no canary traffic and 10000 bp mean all canary traffic.
    return (int(digest.split(":", 1)[1][:16], 16) % 10000) + 1

