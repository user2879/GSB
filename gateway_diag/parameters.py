"""Bit-reproducible model construction and NumPy forward inference."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

from .errors import SpecError
from .model import Backend, canonical_json

PARAMETER_FORMAT_VERSION = 1
_QUANTIZATION_VERSION = {
    "fp32": "fp32-v1",
    "int8": "symmetric-int8-v1",
    "int4": "symmetric-int4-v1",
}


def _uniform(material: bytes, count: int) -> np.ndarray:
    words: list[np.ndarray] = []
    produced = 0
    counter = 0
    while produced < count:
        digest = hashlib.sha256(material + counter.to_bytes(8, "big")).digest()
        chunk = np.frombuffer(digest, dtype=">u8").astype(np.uint64)
        words.append(chunk)
        produced += int(chunk.size)
        counter += 1
    combined = np.concatenate(words)[:count]
    return (combined.astype(np.float64) / float(2**64)) * 2.0 - 1.0


def _quantize(array: np.ndarray, quantization: str) -> np.ndarray:
    if quantization == "fp32":
        return array.astype(np.float32, copy=False)
    levels = 127 if quantization == "int8" else 7
    quantized = np.clip(np.rint(array.astype(np.float64) * levels), -levels, levels)
    return (quantized / float(levels)).astype(np.float32)


def _array_digest(array: np.ndarray) -> str:
    canonical = np.ascontiguousarray(array.astype("<f4", copy=False))
    header = f"{canonical.dtype.str}:{canonical.shape}".encode("utf-8")
    return hashlib.sha256(
        b"gateway-param-array-v1\0" + header + b"\0" + canonical.tobytes()
    ).hexdigest()


@dataclass(frozen=True)
class BackendParameters:
    backend_id: str
    model_dim: int
    hidden_dim: int
    seed: int
    quantization: str
    quantization_version: str
    weight_in: np.ndarray
    bias_in: np.ndarray
    weight_out: np.ndarray
    bias_out: np.ndarray
    parameter_count: int

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "format_version": PARAMETER_FORMAT_VERSION,
            "backend_id": self.backend_id,
            "model_dim": self.model_dim,
            "hidden_dim": self.hidden_dim,
            "seed": self.seed,
            "quantization": self.quantization,
            "quantization_version": self.quantization_version,
            "parameter_count": self.parameter_count,
        }

    @property
    def metadata_digest(self) -> str:
        return hashlib.sha256(
            b"gateway-param-metadata-v1\0"
            + canonical_json(self.metadata).encode("utf-8")
        ).hexdigest()

    @property
    def arrays_digest(self) -> str:
        payload = b"\0".join(
            _array_digest(array).encode("ascii")
            for array in (
                self.weight_in,
                self.bias_in,
                self.weight_out,
                self.bias_out,
            )
        )
        return hashlib.sha256(b"gateway-param-arrays-v1\0" + payload).hexdigest()

    @property
    def parameter_digest(self) -> str:
        payload = (
            b"gateway-params-v1\0"
            + self.metadata_digest.encode("ascii")
            + b"\0"
            + self.arrays_digest.encode("ascii")
        )
        return hashlib.sha256(payload).hexdigest()

    def breakdown(self) -> dict[str, object]:
        result = dict(self.metadata)
        result.update(
            {
                "metadata_digest": self.metadata_digest,
                "arrays_digest": self.arrays_digest,
                "parameter_digest": self.parameter_digest,
            }
        )
        return result


def generate_parameters(backend: Backend) -> BackendParameters:
    quantization_version = _QUANTIZATION_VERSION[backend.quantization]
    namespace = (
        f"gateway-params-v1\0{backend.id}\0{backend.model_dim}\0"
        f"{backend.hidden_dim}\0{backend.seed}\0{quantization_version}"
    ).encode("utf-8")
    input_count = backend.model_dim * backend.hidden_dim
    output_count = backend.hidden_dim * backend.model_dim
    weight_in = _quantize(
        _uniform(namespace + b"\0weight_in", input_count).reshape(
            backend.model_dim, backend.hidden_dim
        )
        * 0.25,
        backend.quantization,
    )
    bias_in = (
        _uniform(namespace + b"\0bias_in", backend.hidden_dim) * 0.05
    ).astype(np.float32)
    weight_out = _quantize(
        _uniform(namespace + b"\0weight_out", output_count).reshape(
            backend.hidden_dim, backend.model_dim
        )
        * 0.25,
        backend.quantization,
    )
    bias_out = (
        _uniform(namespace + b"\0bias_out", backend.model_dim) * 0.05
    ).astype(np.float32)
    parameter_count = int(weight_in.size + bias_in.size + weight_out.size + bias_out.size)
    return BackendParameters(
        backend_id=backend.id,
        model_dim=backend.model_dim,
        hidden_dim=backend.hidden_dim,
        seed=backend.seed,
        quantization=backend.quantization,
        quantization_version=quantization_version,
        weight_in=weight_in,
        bias_in=bias_in,
        weight_out=weight_out,
        bias_out=bias_out,
        parameter_count=parameter_count,
    )


def parameter_change_sources(
    previous: BackendParameters | None, current: BackendParameters
) -> list[str]:
    if previous is None:
        return ["backend_added"]
    fields = (
        "backend_id",
        "model_dim",
        "hidden_dim",
        "seed",
        "quantization",
        "quantization_version",
    )
    changes = [
        field
        for field in fields
        if getattr(previous, field) != getattr(current, field)
    ]
    if previous.parameter_digest != current.parameter_digest and not changes:
        changes.append("parameter_arrays")
    return changes


def forward(params: BackendParameters, vector: Iterable[float]) -> np.ndarray:
    values = np.asarray(tuple(vector), dtype=np.float64)
    if values.ndim != 1 or values.shape[0] != params.model_dim:
        raise SpecError(
            f"input vector length {values.shape[0] if values.ndim else 0} does not match model_dim {params.model_dim}",
            field="input",
            backend_id=params.backend_id,
        )
    if not np.all(np.isfinite(values)):
        raise SpecError(
            "input vector contains a non-finite value",
            field="input",
            backend_id=params.backend_id,
        )
    hidden = np.tanh(
        np.sum(values[:, None] * params.weight_in.astype(np.float64), axis=0)
        + params.bias_in.astype(np.float64)
    )
    output = np.sum(
        hidden[:, None] * params.weight_out.astype(np.float64), axis=0
    ) + params.bias_out.astype(np.float64)
    if not np.all(np.isfinite(output)):
        raise SpecError(
            "inference produced a non-finite output",
            field="output",
            backend_id=params.backend_id,
        )
    return output.astype(np.float32)


def output_digest(backend_id: str, request_id: int, output: np.ndarray) -> str:
    payload = (
        b"gateway-output-v1\0"
        + backend_id.encode("utf-8")
        + b"\0"
        + str(request_id).encode("ascii")
        + b"\0"
        + np.ascontiguousarray(output.astype("<f4", copy=False)).tobytes()
    )
    return hashlib.sha256(payload).hexdigest()


def infer_request(
    params: BackendParameters, request_id: int, vector: Iterable[float]
) -> tuple[np.ndarray, str]:
    output = forward(params, vector)
    return output, output_digest(params.backend_id, request_id, output)


def batch_output_digest(
    backend_id: str,
    batch_id: str,
    request_digests: Mapping[int, str],
) -> str:
    ordered = sorted(request_digests.items())
    payload = {
        "backend_id": backend_id,
        "batch_id": batch_id,
        "request_digests": {str(request_id): digest for request_id, digest in ordered},
    }
    return hashlib.sha256(
        b"gateway-batch-output-v1\0"
        + canonical_json(payload).encode("utf-8")
    ).hexdigest()


def rounded_output_vector(output: np.ndarray, length: int = 8) -> list[float]:
    values = [float(value) for value in output[:length]]
    return [round(value, 8) for value in values]
