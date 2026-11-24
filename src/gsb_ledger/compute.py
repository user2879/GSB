"""Deterministic integer forward computation shared by server and fixtures."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .model import RequestSpec, WorkerSpec
from .util import digest_value


def _seed_bytes(seed: int, label: str, length: int) -> bytes:
    material = f"{seed}:{label}".encode("utf-8")
    output = bytearray()
    counter = 0
    while len(output) < length:
        output.extend(hashlib.sha256(material + b":" + str(counter).encode("ascii")).digest())
        counter += 1
    return bytes(output[:length])


def _int_array(seed: int, label: str, shape: tuple[int, ...], low: int, high: int) -> np.ndarray:
    size = math.prod(shape)
    raw = _seed_bytes(seed, label, size * 4)
    values = [
        low + (int.from_bytes(raw[index * 4 : index * 4 + 4], "little") % (high - low + 1))
        for index in range(size)
    ]
    return np.asarray(values, dtype=np.int64).reshape(shape)


@dataclass
class WorkerRuntime:
    worker: WorkerSpec
    param_version: int
    throughput_ops_per_ms: int = 100_000

    def __post_init__(self) -> None:
        seed = self.worker.seed + self.param_version * 1_000_003
        self.weight_input = _int_array(
            seed,
            "weight-input",
            (self.worker.model_dim, self.worker.hidden_dim),
            -3,
            3,
        )
        self.weight_output = _int_array(
            seed,
            "weight-output",
            (self.worker.hidden_dim, self.worker.model_dim),
            -3,
            3,
        )

    @property
    def param_digest(self) -> str:
        return digest_value(
            {
                "worker_id": self.worker.worker_id,
                "param_version": self.param_version,
                "model_dim": self.worker.model_dim,
                "hidden_dim": self.worker.hidden_dim,
                "seed": self.worker.seed,
                "quant_scale": self.worker.quant_scale,
            }
        )

    def forward(self, request: RequestSpec) -> np.ndarray:
        seed = self.worker.seed + self.param_version * 1_000_003
        if request.payload:
            raw = list(request.payload)
            if len(raw) < self.worker.model_dim:
                raw.extend([0] * (self.worker.model_dim - len(raw)))
            input_vector = np.asarray(raw[: self.worker.model_dim], dtype=np.int64)
        else:
            input_vector = _int_array(
                seed,
                f"input:{request.request_id}",
                (self.worker.model_dim,),
                -8,
                8,
            )
        hidden = np.matmul(input_vector, self.weight_input)
        hidden = np.mod(hidden, 65_521).astype(np.int64, copy=False)
        output = np.matmul(hidden, self.weight_output)
        return np.mod(output, 65_521).astype(np.int64, copy=False)

    def output_digest(self, request: RequestSpec) -> str:
        output = self.forward(request)
        return digest_value(
            {
                "dtype": "int64",
                "shape": list(output.shape),
                "values": output.tolist(),
            }
        )

    def macs(self, request_count: int) -> int:
        model_dim = self.worker.model_dim
        hidden_dim = self.worker.hidden_dim
        per_request = 8 * model_dim * hidden_dim + 4 * model_dim * model_dim + 2 * hidden_dim
        return request_count * per_request

    def duration_ms(self, request_count: int) -> int:
        if request_count <= 0:
            return 0
        return max(1, (self.macs(request_count) + self.throughput_ops_per_ms - 1) // self.throughput_ops_per_ms)


def build_receipt_rows(
    runtime: WorkerRuntime,
    *,
    batch_id: str,
    requests: Iterable[RequestSpec],
    version: str,
    duration_ms: int | None = None,
    covered_refs: dict[str, list[str]] | None = None,
) -> list[dict[str, Any]]:
    ordered = sorted(requests, key=lambda item: (item.seq, item.request_id))
    duration = runtime.duration_ms(len(ordered)) if duration_ms is None else int(duration_ms)
    covered_refs = covered_refs or {}
    rows: list[dict[str, Any]] = []
    for rank, request in enumerate(ordered):
        row: dict[str, Any] = {
            "batch_id": batch_id,
            "request_id": request.request_id,
            "output_digest": runtime.output_digest(request),
        }
        if version in {"r2", "r3"}:
            row["param_version"] = runtime.param_version
            row["duration_ms"] = duration
            row["worker_id"] = runtime.worker.worker_id
        if version == "r3":
            row["rank"] = rank
            row["covered_refs"] = list(covered_refs.get(request.request_id, []))
            row["param_digest"] = runtime.param_digest
        rows.append(row)
    return rows

