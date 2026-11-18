from __future__ import annotations

import hashlib
import struct
import threading
import time
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .errors import RequestError
from .spec import GatewaySpec, ModelSpec


@dataclass(frozen=True)
class BackendResult:
    output_digest: str
    elapsed_us: int
    output_value: float

    def canonical(self) -> dict[str, Any]:
        return {
            "output_digest": self.output_digest,
            "elapsed_us": self.elapsed_us,
            "output_value": self.output_value,
        }


def _resize_sample(sample: Sequence[float], model_dim: int) -> np.ndarray:
    if isinstance(sample, (str, bytes)) or not isinstance(sample, Sequence):
        raise RequestError("E_REQUEST", "$.request.sample", "must be a JSON array")
    if not sample:
        raise RequestError(
            "E_REQUEST", "$.request.sample", "must contain at least one number"
        )
    values: list[float] = []
    for index, value in enumerate(sample):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RequestError(
                "E_REQUEST",
                f"$.request.sample[{index}]",
                "must be a finite number",
            )
        number = float(value)
        if not np.isfinite(number):
            raise RequestError(
                "E_REQUEST",
                f"$.request.sample[{index}]",
                "must be a finite number",
            )
        values.append(number)
    result = np.empty(model_dim, dtype=np.float64)
    for index in range(model_dim):
        result[index] = values[index % len(values)]
    return result


class ForwardEngine:
    """Deterministic NumPy forward pass used by both live and replay paths."""

    def __init__(self, spec: GatewaySpec):
        self.spec = spec
        self._weights: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._lock = threading.Lock()

    def _weights_for(self, model: ModelSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with self._lock:
            cached = self._weights.get(model.version)
            if cached is not None:
                return cached
            rng = np.random.Generator(np.random.PCG64(model.seed))
            w1 = np.asarray(
                rng.standard_normal((model.model_dim, model.hidden_dim)),
                dtype=np.float64,
            )
            bias = np.asarray(rng.standard_normal(model.hidden_dim), dtype=np.float64)
            w2 = np.asarray(rng.standard_normal(model.hidden_dim), dtype=np.float64)
            self._weights[model.version] = (w1, bias, w2)
            return w1, bias, w2

    def forward_one(self, model: ModelSpec, sample: Sequence[float]) -> tuple[str, float]:
        w1, bias, w2 = self._weights_for(model)
        vector = _resize_sample(sample, model.model_dim)
        hidden = np.tanh(vector @ w1 + bias)
        value = float(hidden @ w2)
        if not np.isfinite(value):
            raise RequestError(
                "E_REQUEST", "$.request.sample", "forward result is not finite"
            )
        digest = "sha256:" + hashlib.sha256(struct.pack("<d", value)).hexdigest()
        return digest, value

    def infer_batch(
        self, model_version: str, samples: Sequence[Sequence[float]]
    ) -> list[BackendResult]:
        if not samples:
            return []
        try:
            model = self.spec.models_by_version[model_version]
        except KeyError as exc:
            raise RequestError(
                "E_REQUEST",
                "$.backend.model_version",
                f"unknown model version {model_version!r}",
            ) from exc
        started = time.monotonic_ns()
        results: list[BackendResult] = []
        for sample in samples:
            digest, value = self.forward_one(model, sample)
            results.append(BackendResult(digest, 1, value))
        elapsed_us = max(1, (time.monotonic_ns() - started) // 1_000)
        per_sample = max(1, elapsed_us // len(results))
        return [
            BackendResult(item.output_digest, per_sample, item.output_value)
            for item in results
        ]


class LocalBackendClient:
    def __init__(self, engine: ForwardEngine):
        self.engine = engine

    def infer(
        self, model_version: str, samples: Sequence[Sequence[float]]
    ) -> list[BackendResult]:
        return self.engine.infer_batch(model_version, samples)

