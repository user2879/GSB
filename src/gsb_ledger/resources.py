"""Integer-only fixed-point resource calculation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .attribution import Batch
from .model import AppConfig, WorkerSpec, is_missing
from .util import digest_value


@dataclass(frozen=True)
class ResourceMetrics:
    batch_id: str
    worker_id: str
    param_version: int | None
    request_count: int
    model_dim: int
    hidden_dim: int
    quant_scale: int
    ops_per_request: int
    raw_ops: int
    fixed_macs: int
    fixed_point_bits: int
    quantized_step: int
    precision_factor: int
    duration_ms: int | None
    utilization_units: int
    utilization_scale: int
    retained_decimal_digits: int
    utilization_state: str
    cache_key: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "worker_id": self.worker_id,
            "param_version": self.param_version,
            "request_count": self.request_count,
            "model_dim": self.model_dim,
            "hidden_dim": self.hidden_dim,
            "quant_scale": self.quant_scale,
            "ops_per_request": self.ops_per_request,
            "raw_ops": self.raw_ops,
            "fixed_macs": self.fixed_macs,
            "fixed_point_bits": self.fixed_point_bits,
            "quantized_step": self.quantized_step,
            "precision_factor": self.precision_factor,
            "duration_ms": self.duration_ms,
            "utilization_units": self.utilization_units,
            "utilization_scale": self.utilization_scale,
            "retained_decimal_digits": self.retained_decimal_digits,
            "utilization_state": self.utilization_state,
            "cache_key": self.cache_key,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResourceMetrics":
        return cls(
            batch_id=str(data["batch_id"]),
            worker_id=str(data["worker_id"]),
            param_version=data.get("param_version"),
            request_count=int(data["request_count"]),
            model_dim=int(data["model_dim"]),
            hidden_dim=int(data["hidden_dim"]),
            quant_scale=int(data["quant_scale"]),
            ops_per_request=int(data["ops_per_request"]),
            raw_ops=int(data["raw_ops"]),
            fixed_macs=int(data["fixed_macs"]),
            fixed_point_bits=int(data["fixed_point_bits"]),
            quantized_step=int(data["quantized_step"]),
            precision_factor=int(data["precision_factor"]),
            duration_ms=data.get("duration_ms"),
            utilization_units=int(data["utilization_units"]),
            utilization_scale=int(data["utilization_scale"]),
            retained_decimal_digits=int(data["retained_decimal_digits"]),
            utilization_state=str(data["utilization_state"]),
            cache_key=str(data["cache_key"]),
        )


def _quantization(bits: int) -> tuple[int, int, int]:
    quantized_step = 1 << max(0, 32 - bits)
    # The factor changes monotonically with the configured width, while the
    # step controls lost low-order bits below 32 bits. Both the MAC and the
    # capacity are expressed in the same fixed-point unit.
    precision_factor = 1 << max(0, bits - 8)
    retained_decimal_digits = max(1, (bits + 3) // 4)
    return quantized_step, precision_factor, retained_decimal_digits


def resource_cache_material(batch: Batch, config: AppConfig) -> dict[str, Any]:
    worker: WorkerSpec | None = config.worker_by_id.get(batch.worker_id)
    durations = [
        int(receipt.duration_ms)
        for receipt in batch.receipts
        if not is_missing(receipt.duration_ms)
    ]
    return {
        "batch_id": batch.batch_id,
        "worker_id": batch.worker_id,
        "param_version": batch.param_version,
        "request_count": max(0, batch.request_count),
        "model_dim": 0 if worker is None else worker.model_dim,
        "hidden_dim": 0 if worker is None else worker.hidden_dim,
        "quant_scale": 1 if worker is None else worker.quant_scale,
        "duration_ms": max(durations) if durations else None,
        "fixed_point_bits": config.fixed_point_bits,
        "throughput_ops_per_ms": config.throughput_ops_per_ms,
    }


def resource_cache_key(batch: Batch, config: AppConfig) -> str:
    return digest_value(resource_cache_material(batch, config))


def compute_resource(batch: Batch, config: AppConfig) -> ResourceMetrics:
    worker: WorkerSpec | None = config.worker_by_id.get(batch.worker_id)
    if worker is None:
        model_dim = 0
        hidden_dim = 0
        quant_scale = 1
    else:
        model_dim = worker.model_dim
        hidden_dim = worker.hidden_dim
        quant_scale = worker.quant_scale
    request_count = max(0, batch.request_count)
    ops_per_request = 8 * model_dim * hidden_dim + 4 * model_dim * model_dim + 2 * hidden_dim
    raw_ops = request_count * ops_per_request
    step, precision_factor, retained_digits = _quantization(config.fixed_point_bits)
    scaled = raw_ops * quant_scale * precision_factor
    fixed_macs = (scaled // step) * step
    duration = None
    if batch.receipts:
        durations = [
            int(receipt.duration_ms)
            for receipt in batch.receipts
            if not is_missing(receipt.duration_ms)
        ]
        if durations:
            duration = max(durations)
    utilization_scale = 10**retained_digits
    if duration is None or duration <= 0 or worker is None:
        utilization_units = 0
        utilization_state = "unavailable" if worker is None else "zero_duration"
    else:
        capacity = (
            duration
            * config.throughput_ops_per_ms
            * quant_scale
            * precision_factor
        )
        utilization_units = (fixed_macs * utilization_scale) // capacity
        utilization_state = "computed"
    material = resource_cache_material(batch, config)
    return ResourceMetrics(
        batch_id=batch.batch_id,
        worker_id=batch.worker_id,
        param_version=batch.param_version,
        request_count=request_count,
        model_dim=model_dim,
        hidden_dim=hidden_dim,
        quant_scale=quant_scale,
        ops_per_request=ops_per_request,
        raw_ops=raw_ops,
        fixed_macs=fixed_macs,
        fixed_point_bits=config.fixed_point_bits,
        quantized_step=step,
        precision_factor=precision_factor,
        duration_ms=duration,
        utilization_units=utilization_units,
        utilization_scale=utilization_scale,
        retained_decimal_digits=retained_digits,
        utilization_state=utilization_state,
        cache_key=digest_value(material),
    )


def compute_resources(
    batches: tuple[Batch, ...], config: AppConfig
) -> tuple[ResourceMetrics, ...]:
    return tuple(
        compute_resource(batch, config)
        for batch in sorted(batches, key=lambda item: (item.batch_id, item.worker_id))
    )
