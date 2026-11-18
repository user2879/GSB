from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import __version__
from .config import InferenceConfig, load_config
from .deterministic import (
    ArtifactSet,
    ModelParams,
    generate_artifacts,
    load_expected_manifest,
    verify_artifacts,
)
from .errors import E_DATA, E_IO, E_LATENCY, E_OOM, E_PRESSURE, DiagnosticError
from .inference import (
    analyze_sample,
    float_forward,
    output_digest,
    quantized_model_weights,
    sample_input_digest,
    summarize_layer_attribution,
)
from .latency import LatencyRecorder
from .memory import BlockPool, PoolOOM, replay_events
from .versioning import ConfigVersionManager


ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class RunRequest:
    seed: int = 184
    sample_count: int = 48
    input_dim: int = 32
    hidden_dim: int = 128

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "sample_count": self.sample_count,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": 4,
        }


def _activation_bytes() -> int:
    return 4


def _scratch_bytes(precision: str) -> int:
    return 8 if precision == "fixed" else 0


def _layer_sizes(model: ModelParams) -> tuple[int, ...]:
    return model.layer_sizes


def per_batch_workspace_bytes(model: ModelParams, precision: str, batch_size: int) -> int:
    sizes = _layer_sizes(model)
    max_width = max(sizes)
    activation = _activation_bytes()
    scratch = _scratch_bytes(precision)
    bytes_per_sample = (
        sizes[0] * activation
        + max_width * activation
        + max_width * activation
        + sizes[-1] * activation
        + max_width * scratch
    )
    return int(batch_size) * bytes_per_sample


def block_pool_peak_bytes(model: ModelParams, precision: str, batch_size: int) -> int:
    sizes = _layer_sizes(model)
    activation = _activation_bytes()
    scratch = _scratch_bytes(precision)
    peak_per_sample = 0
    previous_width = sizes[0]
    for output_width in sizes[1:]:
        stage = previous_width * activation + output_width * activation + output_width * scratch
        peak_per_sample = max(peak_per_sample, stage)
        previous_width = output_width
    return int(batch_size) * peak_per_sample


def request_blocks(
    model: ModelParams,
    precision: str,
    preallocation: str,
    batch_size: int,
    block_size_bytes: int,
) -> int:
    if batch_size <= 0:
        return 0
    if preallocation == "per_batch":
        request_bytes = per_batch_workspace_bytes(model, precision, batch_size)
    else:
        request_bytes = block_pool_peak_bytes(model, precision, batch_size)
    return math.ceil(request_bytes / block_size_bytes)


def model_weight_bytes(model: ModelParams) -> int:
    return int(model.parameter_count * 4)


class InferenceEngine:
    def __init__(self, on_event: ProgressCallback | None = None) -> None:
        self.on_event = on_event

    def _emit(self, event_type: str, **payload: Any) -> None:
        if self.on_event is not None:
            self.on_event({"type": event_type, **payload})

    def run(
        self,
        config: InferenceConfig,
        artifacts: ArtifactSet,
        *,
        config_version: int,
        replacement: dict[str, Any] | None = None,
        expected_manifest: dict[str, Any] | None = None,
        report_path: str | Path | None = None,
    ) -> dict[str, Any]:
        integrity = verify_artifacts(artifacts, expected_manifest)
        run_origin_ns = time.perf_counter_ns()
        started_at = datetime.now(timezone.utc).isoformat()
        codes: list[str] = []
        queue_events: list[dict[str, Any]] = []
        batches: list[dict[str, Any]] = []
        samples: list[dict[str, Any]] = []
        layer_bounds: list[dict[str, Any]] = []
        fatal_error: dict[str, Any] | None = None

        block_pool = BlockPool(
            config.total_blocks,
            config.block_size_bytes,
            origin_ns=run_origin_ns,
            on_event=self._on_memory_event,
        )
        model_blocks = 0

        self._emit(
            "run_start",
            config=config.to_dict(),
            config_version=config_version,
            total_blocks=config.total_blocks,
            block_size_bytes=config.block_size_bytes,
            artifact_summary=artifacts.summary(),
        )

        try:
            weight_request = model_weight_bytes(artifacts.model)
            model_allocation = block_pool.allocate(
                owner_id="model.weights",
                batch_id=None,
                purpose="model_weights",
                request_bytes=weight_request,
                strategy="persistent_contiguous",
                contiguous=True,
                stage="resident",
            )
            model_blocks = len(model_allocation.blocks)
            self._emit(
                "model_resident",
                blocks=model_blocks,
                requested_bytes=weight_request,
                total_blocks=config.total_blocks,
            )
        except PoolOOM as exc:
            codes.extend([E_OOM])
            fatal_error = {
                **exc.event,
                "reason": "model weights do not fit in the configured block budget",
            }
            self._emit("fatal_oom", error=fatal_error)

        cursor = 0
        batch_number = 0
        quantized_weights = (
            quantized_model_weights(artifacts.model, config.quant_scale_power)
            if config.precision == "fixed"
            else None
        )
        total_samples = artifacts.dataset.sample_count

        while fatal_error is None and cursor < total_samples:
            batch_number += 1
            batch_id = f"B{batch_number:04d}"
            remaining = total_samples - cursor
            requested_count = min(config.batch_size_limit, remaining)
            one_sample_blocks = request_blocks(
                artifacts.model,
                config.precision,
                config.preallocation,
                1,
                config.block_size_bytes,
            )
            if config.preallocation == "per_batch":
                free_capacity = block_pool.largest_free_extent()
            else:
                free_capacity = block_pool.free_blocks

            capacity_count = _max_batch_fit(
                artifacts.model,
                config,
                free_capacity,
                hard_limit=requested_count,
            )
            if capacity_count < requested_count:
                queue_events.append(
                    {
                        "batch_id": batch_id,
                        "queue_reason": "CAPACITY_SPLIT",
                        "requested_samples": requested_count,
                        "admitted_samples": capacity_count,
                        "queued_samples": requested_count - capacity_count,
                        "free_blocks": block_pool.free_blocks,
                        "largest_free_extent": block_pool.largest_free_extent(),
                        "one_sample_blocks": one_sample_blocks,
                        "message": "samples remain queued; block budget is never bypassed",
                    }
                )

            if capacity_count < 1:
                try:
                    block_pool.allocate(
                        owner_id=f"{batch_id}.probe",
                        batch_id=batch_id,
                        purpose="batch_workspace_probe",
                        request_bytes=(
                            per_batch_workspace_bytes(artifacts.model, config.precision, 1)
                            if config.preallocation == "per_batch"
                            else block_pool_peak_bytes(artifacts.model, config.precision, 1)
                        ),
                        strategy=config.preallocation,
                        contiguous=config.preallocation == "per_batch",
                        stage="probe",
                        requested_blocks=one_sample_blocks,
                    )
                except PoolOOM as exc:
                    codes.append(E_OOM)
                    fatal_error = {
                        **exc.event,
                        "reason": "not enough free blocks for one sample; batch remains queued",
                    }
                    self._emit("fatal_oom", error=fatal_error)
                break

            actual_count = capacity_count
            allocation_success = False
            while actual_count >= 1 and not allocation_success:
                start_ns = time.perf_counter_ns()
                self._emit(
                    "batch_start",
                    batch_id=batch_id,
                    config_version=config_version,
                    requested_samples=requested_count,
                    admitted_samples=actual_count,
                    sample_start=cursor,
                    sample_end=cursor + actual_count - 1,
                )
                try:
                    batch_result = self._execute_batch(
                        block_pool=block_pool,
                        artifacts=artifacts,
                        config=config,
                        config_version=config_version,
                        batch_id=batch_id,
                        cursor=cursor,
                        count=actual_count,
                        quantized_weights=quantized_weights,
                        model_blocks=model_blocks,
                    )
                    allocation_success = True
                except PoolOOM as exc:
                    codes.append(E_OOM)
                    queue_events.append(
                        {
                            "batch_id": batch_id,
                            "queue_reason": "ALLOCATION_RETRY",
                            "requested_samples": actual_count,
                            "admitted_samples": actual_count - 1,
                            "queued_samples": 1,
                            "required_blocks": exc.required_blocks,
                            "free_blocks": exc.free_blocks,
                            "largest_free_extent": exc.largest_free_extent,
                            "message": "allocation rolled back atomically; retry with one fewer sample",
                        }
                    )
                    actual_count -= 1
                    if actual_count < 1:
                        fatal_error = {
                            **exc.event,
                            "reason": "single-sample batch cannot be allocated",
                        }
                        self._emit("fatal_oom", error=fatal_error)
                    continue

                end_ns = time.perf_counter_ns()
                batch = batch_result["batch"]
                batch["start_mono_ns"] = start_ns
                batch["end_mono_ns"] = end_ns
                batch["start_elapsed_ms"] = (start_ns - run_origin_ns) / 1_000_000.0
                batch["end_elapsed_ms"] = (end_ns - run_origin_ns) / 1_000_000.0
                batch["duration_ms"] = (end_ns - start_ns) / 1_000_000.0
                batch["progress"] = {
                    "completed_samples": cursor + actual_count,
                    "total_samples": total_samples,
                    "fraction_ppm": int(round(1_000_000 * (cursor + actual_count) / total_samples)),
                }
                batches.append(batch)
                samples.extend(batch_result["samples"])
                cursor += actual_count
                self._emit(
                    "batch_complete",
                    batch_id=batch_id,
                    config_version=config_version,
                    batch=batch,
                    progress=batch["progress"],
                )

                if cursor < total_samples and actual_count < remaining:
                    queue_events.append(
                        {
                            "batch_id": f"B{batch_number + 1:04d}",
                            "queue_reason": "PENDING_NEXT_BATCH",
                            "requested_samples": min(config.batch_size_limit, total_samples - cursor),
                            "admitted_samples": 0,
                            "queued_samples": total_samples - cursor,
                            "free_blocks": block_pool.free_blocks,
                            "largest_free_extent": block_pool.largest_free_extent(),
                            "message": "waiting for the next scheduling cycle",
                        }
                    )

        try:
            if model_blocks:
                block_pool.release("model.weights", reason="run complete")
        except RuntimeError:
            pass

        memory_events = list(block_pool.events)
        replay = replay_events(memory_events, config.total_blocks)
        if not replay["verified"]:
            codes.append("E_REPLAY")

        one_sample_blocks = request_blocks(
            artifacts.model,
            config.precision,
            config.preallocation,
            1,
            config.block_size_bytes,
        )
        pressure_intervals = _derive_pressure_intervals(
            memory_events,
            total_blocks=config.total_blocks,
            one_sample_blocks=one_sample_blocks,
            threshold_ms=config.pressure_threshold_ms,
        )
        if pressure_intervals:
            codes.append(E_PRESSURE)

        layer_bounds = summarize_layer_attribution(samples) if config.precision == "fixed" and samples else []
        conclusion = _build_conclusion(config, codes, samples, layer_bounds, fatal_error)
        if fatal_error is not None:
            conclusion["status"] = "FAIL"
            conclusion["message"] = "run stopped at a fatal allocator error"

        curve = [
            {
                "seq": event["seq"],
                "batch_id": event["batch_id"],
                "action": event["action"],
                "elapsed_ms": event["elapsed_ms"],
                "used_blocks": event["used_blocks_after"],
                "free_blocks": event["free_blocks_after"],
                "fragmentation_blocks": event["fragmentation_blocks_after"],
                "peak_used_blocks": event["peak_used_blocks_after"],
            }
            for event in memory_events
        ]
        unique_codes = sorted(set(codes))
        report = {
            "schema": "mixedprec-diagnostic-report/v1",
            "run_id": _run_id(artifacts, config, config_version),
            "generated_at_utc": started_at,
            "application": {"name": "mixedprec_diag", "version": __version__},
            "environment": _environment(),
            "request": {
                "seed": artifacts.seed,
                "sample_count": artifacts.dataset.sample_count,
                "input_dim": artifacts.dataset.input_dim,
                "hidden_dim": int(artifacts.model.weights[0].shape[1]),
                "output_dim": int(artifacts.model.weights[-1].shape[1]),
                "layer_sizes": list(artifacts.model.layer_sizes),
            },
            "artifacts": {
                "data_sha256": artifacts.data_sha256,
                "params_sha256": artifacts.params_sha256,
                "data_summary": artifacts.data_summary,
                "params_summary": artifacts.params_summary,
                "integrity": integrity,
            },
            "config": config.to_dict(),
            "config_version": config_version,
            "replacement": replacement,
            "summary": {
                "status": conclusion["status"],
                "codes": unique_codes,
                "fatal_error": fatal_error,
                "samples_completed": len(samples),
                "samples_total": total_samples,
                "batches_completed": len(batches),
                "conclusion": conclusion,
            },
            "progress": {
                "completed_samples": len(samples),
                "total_samples": total_samples,
                "fraction_ppm": int(round(1_000_000 * len(samples) / total_samples)) if total_samples else 0,
                "batches": [
                    {
                        "batch_id": batch["batch_id"],
                        "completed_samples": batch["sample_count"],
                        "total_samples": total_samples,
                        "fraction_ppm": batch["progress"]["fraction_ppm"],
                    }
                    for batch in batches
                ],
            },
            "batches": batches,
            "samples": samples,
            "layer_error_bounds": layer_bounds,
            "quantization": {
                "mode": config.precision,
                "quant_scale_power": config.quant_scale_power,
                "quantization_scale": config.quantization_scale,
                "rounding": "nearest; exact ties away from zero",
                "weight_storage": "signed int32 with saturation",
                "activation_storage": "signed int32 with saturation",
                "accumulator": "exact Python integer, then saturated to signed int32 after S^2 -> S rounding",
                "rounding_limit_int_milli": 500 if config.precision == "fixed" else None,
                "weight_saturations": (
                    quantized_weights[2]
                    if quantized_weights is not None
                    else {}
                ),
            },
            "queue_events": queue_events,
            "pressure": {
                "threshold_ms": config.pressure_threshold_ms,
                "intervals": pressure_intervals,
                "reported": E_PRESSURE in unique_codes,
            },
            "memory": {
                "clock_basis": "time.perf_counter_ns (monotonic)",
                "budget_mib": config.memory_budget_mib,
                "requested_budget_bytes": config.requested_budget_bytes,
                "effective_budget_bytes": config.effective_budget_bytes,
                "block_size_kb": config.block_size_kb,
                "block_size_bytes": config.block_size_bytes,
                "total_blocks": config.total_blocks,
                "model_weight_bytes": model_weight_bytes(artifacts.model),
                "model_resident_blocks": model_blocks,
                "fragmentation_definition": "free_blocks - largest_contiguous_free_extent",
                "peak_used_blocks": block_pool.peak_used_blocks,
                "peak_used_bytes": block_pool.peak_used_blocks * config.block_size_bytes,
                "peak_fragmentation_blocks": block_pool.peak_fragmentation_blocks,
                "final_state": block_pool.final_state(),
                "events": memory_events,
                "curve": curve,
                "replay": replay,
            },
            "ui_latency": LatencyRecorder(config.ui_query_budget_ms).summary(),
        }
        report["path_output_sha256"] = _report_output_digest(batches, config.precision)
        if report_path is not None:
            write_report(report_path, report)
        self._emit(
            "engine_complete",
            run_id=report["run_id"],
            config_version=report["config_version"],
            status=report["summary"]["status"],
        )
        return report

    def _on_memory_event(self, event: dict[str, Any]) -> None:
        self._emit("memory_event", event=event)

    def _execute_batch(
        self,
        *,
        block_pool: BlockPool,
        artifacts: ArtifactSet,
        config: InferenceConfig,
        config_version: int,
        batch_id: str,
        cursor: int,
        count: int,
        quantized_weights: Any,
        model_blocks: int,
    ) -> dict[str, Any]:
        allocations: list[list[int]] = []
        batch_peak_batch_blocks = 0
        batch_peak_total_blocks = 0
        fragmentation_at_peak = 0

        def observe() -> None:
            nonlocal batch_peak_batch_blocks, batch_peak_total_blocks, fragmentation_at_peak
            batch_used = max(0, block_pool.used_blocks - model_blocks)
            total_used = block_pool.used_blocks
            if total_used >= batch_peak_total_blocks:
                batch_peak_total_blocks = total_used
                batch_peak_batch_blocks = max(batch_peak_batch_blocks, batch_used)
                fragmentation_at_peak = block_pool.fragmentation_blocks()

        try:
            if config.preallocation == "per_batch":
                request_bytes = per_batch_workspace_bytes(artifacts.model, config.precision, count)
                allocation = block_pool.allocate(
                    owner_id=f"{batch_id}.arena",
                    batch_id=batch_id,
                    purpose="batch_arena",
                    request_bytes=request_bytes,
                    strategy="per_batch",
                    contiguous=True,
                    stage="batch",
                )
                allocations.append(list(allocation.blocks))
                observe()
            else:
                input_bytes = count * artifacts.dataset.input_dim * _activation_bytes()
                current_owner = f"{batch_id}.input"
                current_allocation = block_pool.allocate(
                    owner_id=current_owner,
                    batch_id=batch_id,
                    purpose="activation_input",
                    request_bytes=input_bytes,
                    strategy="block_pool",
                    contiguous=False,
                    stage="input",
                )
                allocations.append(list(current_allocation.blocks))
                observe()
                sizes = artifacts.model.layer_sizes
                for layer_index, output_width in enumerate(sizes[1:]):
                    output_owner = f"{batch_id}.layer{layer_index}.output"
                    output_allocation = block_pool.allocate(
                        owner_id=output_owner,
                        batch_id=batch_id,
                        purpose=f"activation_layer_{layer_index}",
                        request_bytes=count * output_width * _activation_bytes(),
                        strategy="block_pool",
                        contiguous=False,
                        stage=f"layer_{layer_index}",
                    )
                    allocations.append(list(output_allocation.blocks))
                    observe()
                    scratch_owner = f"{batch_id}.layer{layer_index}.scratch"
                    if _scratch_bytes(config.precision):
                        scratch_allocation = block_pool.allocate(
                            owner_id=scratch_owner,
                            batch_id=batch_id,
                            purpose=f"accumulator_scratch_layer_{layer_index}",
                            request_bytes=count * output_width * _scratch_bytes(config.precision),
                            strategy="block_pool",
                            contiguous=False,
                            stage=f"layer_{layer_index}",
                        )
                        allocations.append(list(scratch_allocation.blocks))
                        observe()
                    block_pool.release(current_owner, reason="previous activation stage consumed")
                    if _scratch_bytes(config.precision):
                        block_pool.release(scratch_owner, reason="layer accumulation complete")
                    current_owner = output_owner

            float_outputs: list[np.ndarray] = []
            fixed_outputs: list[np.ndarray] = []
            batch_samples: list[dict[str, Any]] = []
            for offset in range(count):
                sample_index = cursor + offset
                sample = artifacts.dataset.inputs[sample_index]
                sample_id = f"S{sample_index:06d}"
                if config.precision == "fixed":
                    diagnosis = analyze_sample(
                        artifacts.model,
                        sample,
                        config.quant_scale_power,
                        quantized_weights=(quantized_weights[0], quantized_weights[1]) if quantized_weights else None,
                    )
                    float_output = np.asarray(diagnosis["float_output"], dtype=np.float32)
                    fixed_output = np.asarray(diagnosis["fixed_output_q"], dtype=np.int64)
                    float_outputs.append(float_output)
                    fixed_outputs.append(fixed_output)
                    sample_record = {
                        "sample_id": sample_id,
                        "sample_index": sample_index,
                        "batch_id": batch_id,
                        "batch_offset": offset,
                        "input_sha256": sample_input_digest(sample),
                        "precision": "fixed",
                        "quant_scale_power": config.quant_scale_power,
                        **diagnosis,
                    }
                else:
                    forward = float_forward(artifacts.model, sample)
                    float_output = np.ascontiguousarray(forward.output, dtype=np.float32)
                    float_outputs.append(float_output)
                    sample_record = {
                        "sample_id": sample_id,
                        "sample_index": sample_index,
                        "batch_id": batch_id,
                        "batch_offset": offset,
                        "input_sha256": sample_input_digest(sample),
                        "precision": "float32",
                        "float_output": [float(value) for value in float_output.tolist()],
                        "dominant_layer": None,
                        "layer_diagnostics": [],
                    }
                dominant = sample_record.get("dominant_layer")
                self._emit(
                    "sample_complete",
                    batch_id=batch_id,
                    sample_id=sample_id,
                    batch_offset=offset,
                    dominant_layer=dominant,
                )
                batch_samples.append(sample_record)
        finally:
            for allocation in block_pool.active_allocations():
                if allocation.get("batch_id") == batch_id:
                    block_pool.release(allocation["owner_id"], reason="batch complete / rollback")

        batch_float_digest = output_digest(float_outputs, "float32")
        batch_fixed_digest = output_digest(fixed_outputs, "fixed") if fixed_outputs else None
        selected_digest = (batch_fixed_digest if config.precision == "fixed" else batch_float_digest) or batch_float_digest
        batch = {
            "batch_id": batch_id,
            "config_version": config_version,
            "precision": config.precision,
            "preallocation": config.preallocation,
            "sample_start": cursor,
            "sample_end": cursor + count - 1,
            "sample_ids": [record["sample_id"] for record in batch_samples],
            "sample_count": count,
            "requested_sample_count": count,
            "status": "COMPLETE",
            "output_sha256": selected_digest,
            "float_output_sha256": batch_float_digest,
            "fixed_output_sha256": batch_fixed_digest,
            "peak_batch_blocks": batch_peak_batch_blocks,
            "peak_total_used_blocks": batch_peak_total_blocks,
            "fragmentation_at_peak_blocks": fragmentation_at_peak,
            "free_blocks_after_release": block_pool.free_blocks,
            "allocated_block_sets": allocations,
            "quant_scale_power": config.quant_scale_power,
        }
        return {"batch": batch, "samples": batch_samples}


def _max_batch_fit(
    model: ModelParams,
    config: InferenceConfig,
    available_blocks: int,
    *,
    hard_limit: int,
) -> int:
    if available_blocks <= 0 or hard_limit <= 0:
        return 0
    low, high = 0, hard_limit
    while low < high:
        middle = (low + high + 1) // 2
        needed = request_blocks(
            model,
            config.precision,
            config.preallocation,
            middle,
            config.block_size_bytes,
        )
        if needed <= available_blocks:
            low = middle
        else:
            high = middle - 1
    return low


def _derive_pressure_intervals(
    events: list[dict[str, Any]],
    *,
    total_blocks: int,
    one_sample_blocks: int,
    threshold_ms: float,
) -> list[dict[str, Any]]:
    """Derive exhausted-capacity episodes from allocator state transitions.

    Exhaustion means the remaining free blocks cannot hold one sample's modeled
    workspace. Episodes are measured between adjacent event timestamps, so the
    reported duration comes from the replayable pool log rather than a separate
    timer estimate.
    """

    intervals: list[dict[str, Any]] = []
    open_event: dict[str, Any] | None = None
    for event in events:
        free_after = int(event.get("free_blocks_after", total_blocks))
        exhausted = free_after < one_sample_blocks
        if exhausted and open_event is None:
            open_event = event
        elif not exhausted and open_event is not None:
            duration_ms = (int(event["t_mono_ns"]) - int(open_event["t_mono_ns"])) / 1_000_000.0
            if duration_ms >= threshold_ms:
                intervals.append(
                    {
                        "code": E_PRESSURE,
                        "batch_id": open_event.get("batch_id"),
                        "start_seq": open_event["seq"],
                        "end_seq": event["seq"],
                        "duration_ms": duration_ms,
                        "threshold_ms": threshold_ms,
                        "free_blocks_at_start": open_event.get("free_blocks_after"),
                        "free_blocks_at_end": free_after,
                        "one_sample_blocks": one_sample_blocks,
                    }
                )
            open_event = None
    if open_event is not None and events:
        duration_ms = (int(events[-1]["t_mono_ns"]) - int(open_event["t_mono_ns"])) / 1_000_000.0
        if duration_ms >= threshold_ms:
            intervals.append(
                {
                    "code": E_PRESSURE,
                    "batch_id": open_event.get("batch_id"),
                    "start_seq": open_event["seq"],
                    "end_seq": events[-1]["seq"],
                    "duration_ms": duration_ms,
                    "threshold_ms": threshold_ms,
                    "free_blocks_at_start": open_event.get("free_blocks_after"),
                    "free_blocks_at_end": events[-1].get("free_blocks_after"),
                    "one_sample_blocks": one_sample_blocks,
                }
            )
    return intervals


def _build_conclusion(
    config: InferenceConfig,
    codes: list[str],
    samples: list[dict[str, Any]],
    layer_bounds: list[dict[str, Any]],
    fatal_error: dict[str, Any] | None,
) -> dict[str, Any]:
    if fatal_error is not None:
        return {
            "status": "FAIL",
            "consistent": False,
            "message": "fatal allocator error; numerical conclusion unavailable",
            "numerical_criterion": "not evaluated",
        }
    if not samples:
        return {
            "status": "FAIL",
            "consistent": False,
            "message": "no samples completed",
            "numerical_criterion": "not evaluated",
        }

    finite = all(
        math.isfinite(value)
        for sample in samples
        for value in sample.get("float_output", [])
    )
    if config.precision == "float32":
        consistent = finite and not any(code in {E_OOM, "E_REPLAY"} for code in codes)
        return {
            "status": "PASS" if consistent else "FAIL",
            "consistent": consistent,
            "message": "float32 reference completed; all outputs are finite" if consistent else "float32 reference failed consistency checks",
            "numerical_criterion": "all float32 outputs finite; allocator replay verified",
            "finite_outputs": finite,
            "max_output_abs_error_int": 0,
        }

    max_output_error = max(int(sample.get("output_max_abs_error_int", 0)) for sample in samples)
    output_bound = layer_bounds[-1]["error_bound_int"] if layer_bounds else 0
    bounds_valid = all(bool(layer["error_bound_valid"]) for layer in layer_bounds)
    saturation_total = sum(int(sample.get("total_saturations", 0)) for sample in samples)
    consistent = (
        finite
        and max_output_error <= output_bound
        and bounds_valid
        and saturation_total == 0
        and not any(code in {E_OOM, "E_REPLAY"} for code in codes)
    )
    return {
        "status": "PASS" if consistent else "FAIL",
        "consistent": consistent,
        "message": (
            "fixed-point outputs stay within the reported integer error bound"
            if consistent
            else "fixed-point consistency gate failed"
        ),
        "numerical_criterion": "max(output error int) <= output-layer error bound int; bounds valid; saturation count == 0",
        "finite_outputs": finite,
        "max_output_abs_error_int": max_output_error,
        "output_error_bound_int": output_bound,
        "saturation_total": saturation_total,
        "quantization_rounding_limit_int_milli": 500,
    }


def _report_output_digest(batches: list[dict[str, Any]], precision: str) -> str | None:
    if not batches:
        return None
    digest = hashlib.sha256()
    digest.update(precision.encode("ascii"))
    for batch in batches:
        digest.update(batch["output_sha256"].encode("ascii"))
    return digest.hexdigest()


def _run_id(artifacts: ArtifactSet, config: InferenceConfig, config_version: int) -> str:
    payload = {
        "data": artifacts.data_sha256,
        "params": artifacts.params_sha256,
        "config_version": config_version,
        "config": config.version_fields(),
        "preallocation": config.preallocation,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:24]


def _environment() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "numpy": np.__version__,
        "tkinter_module": importlib.util.find_spec("tkinter") is not None,
        "determinism_scope": "same Python/NumPy platform and generation arguments",
        "gpu_required": False,
        "network_required": False,
    }


def write_report(path: str | Path, report: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    except OSError as exc:
        raise DiagnosticError(E_IO, f"cannot write report: {exc}", "$", {"path": str(destination)}) from exc
    finally:
        if temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass


def ensure_writable_directory(path: str | Path) -> Path:
    directory = Path(path)
    try:
        directory.mkdir(parents=True, exist_ok=True)
        probe = directory / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        raise DiagnosticError(E_IO, f"directory is not writable: {exc}", "$", {"path": str(directory)}) from exc
    return directory


class DiagnosticSession:
    """Versioned, whole-replacement session used by both CLI demo and Tkinter UI."""

    def __init__(
        self,
        request: RunRequest,
        *,
        report_dir: str | Path = "reports",
        snapshot_dir: str | Path = "snapshots",
        on_event: ProgressCallback | None = None,
    ) -> None:
        self.request = request
        self.artifacts = generate_artifacts(
            request.seed,
            request.sample_count,
            request.input_dim,
            request.hidden_dim,
        )
        self.report_dir = ensure_writable_directory(report_dir)
        self.snapshot_dir = ensure_writable_directory(snapshot_dir)
        self.version_manager = ConfigVersionManager()
        self.current_report: dict[str, Any] | None = None
        self.latency: LatencyRecorder | None = None
        self.on_event = on_event
        self._lock = threading.RLock()
        self.history: list[dict[str, Any]] = []

    def run_config(
        self,
        config: InferenceConfig,
        *,
        expected_manifest_path: str | Path | None = None,
    ) -> dict[str, Any]:
        expected = load_expected_manifest(expected_manifest_path) if expected_manifest_path else None
        with self._lock:
            pending = self.version_manager.begin(config)
            old_report = copy.deepcopy(self.current_report) if self.current_report is not None else None
            if old_report is not None and pending.changed_fields:
                stale = copy.deepcopy(old_report)
                stale["stale"] = {
                    "marker": "STALE",
                    "reason": f"replaced by config version {pending.version}",
                    "stale_batch_ids": sorted(batch["batch_id"] for batch in stale.get("batches", [])),
                    "stale_sections": ["batch_outputs", "memory_statistics", "error_conclusions"],
                    "excluded_from_current_report": True,
                }
                stale_path = self.snapshot_dir / f"version_{pending.previous_version:04d}_stale.json"
                write_report(stale_path, stale)
                stale_report_path = self.report_dir / f"version_{pending.previous_version:04d}.json"
                write_report(stale_report_path, stale)
                stale_current_snapshot = self.snapshot_dir / (
                    f"version_{pending.previous_version:04d}_{old_report['config']['precision']}.json"
                )
                write_report(stale_current_snapshot, stale)
                for history_entry in reversed(self.history):
                    if history_entry.get("version") == pending.previous_version and history_entry.get("status") == "CURRENT":
                        history_entry.update(
                            {
                                "status": "STALE",
                                "path": str(stale_report_path),
                                "snapshot_path": str(stale_path),
                                "stale_batch_ids": stale["stale"]["stale_batch_ids"],
                            }
                        )
                        break

            engine = InferenceEngine(on_event=self.on_event)
            try:
                provisional = engine.run(
                    config,
                    self.artifacts,
                    config_version=pending.version,
                    expected_manifest=expected,
                )
            except DiagnosticError as exc:
                if exc.code == E_DATA:
                    aborted = {
                        "schema": "mixedprec-diagnostic-aborted-report/v1",
                        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "application": {"name": "mixedprec_diag", "version": __version__},
                        "environment": _environment(),
                        "request": self.request.to_dict(),
                        "artifacts": {
                            "data_sha256": self.artifacts.data_sha256,
                            "params_sha256": self.artifacts.params_sha256,
                            "data_summary": self.artifacts.data_summary,
                            "params_summary": self.artifacts.params_summary,
                        },
                        "config": config.to_dict(),
                        "config_version": self.version_manager.current_version or 0,
                        "attempted_version": pending.version,
                        "inference_started": False,
                        "summary": {
                            "status": "FAIL",
                            "codes": [E_DATA],
                            "error": exc.to_dict(),
                        },
                        "ui_latency": LatencyRecorder(config.ui_query_budget_ms).summary(),
                    }
                    aborted_path = self.report_dir / "aborted_E_DATA.json"
                    write_report(aborted_path, aborted)
                    write_report(self.snapshot_dir / "aborted_E_DATA.json", aborted)
                raise
            batch_ids = [batch["batch_id"] for batch in provisional.get("batches", [])]
            replacement = self.version_manager.commit(pending, config, batch_ids)
            provisional["replacement"] = replacement
            self.latency = LatencyRecorder(config.ui_query_budget_ms)
            provisional["ui_latency"] = self.latency.summary()
            self.current_report = provisional
            self._write_current()
            self._write_snapshot(provisional)
            self.history.append(
                {
                    "version": pending.version,
                    "status": "CURRENT",
                    "changed_fields": replacement["changed_fields"],
                    "run_id": provisional["run_id"],
                }
            )
            return copy.deepcopy(provisional)

    def run_config_path(self, path: str | Path, **kwargs: Any) -> dict[str, Any]:
        return self.run_config(load_config(path), **kwargs)

    def query_view(self) -> dict[str, Any]:
        """Return the compact read-only projection consumed by the Tkinter tables."""

        if self.current_report is None or self.latency is None:
            raise DiagnosticError(E_DATA, "no current report available", "session.current_report")
        with self._lock:
            start = self.latency.clock_ns()
            result = self._build_query_view()
            end = self.latency.clock_ns()
            self.latency.record("query_view", None, start, end)
            self._apply_latency()
        return result

    def _build_query_view(self) -> dict[str, Any]:
        report = self.current_report
        if report is None:
            raise DiagnosticError(E_DATA, "no current report available", "session.current_report")
        sample_fields = (
            "sample_id",
            "sample_index",
            "batch_id",
            "batch_offset",
            "input_sha256",
            "precision",
            "dominant_layer",
            "output_max_abs_error_int",
            "total_saturations",
        )
        return {
            "schema": "mixedprec-diagnostic-query-view/v1",
            "run_id": report["run_id"],
            "config_version": report["config_version"],
            "config": copy.deepcopy(report["config"]),
            "replacement": copy.deepcopy(report.get("replacement")),
            "summary": copy.deepcopy(report["summary"]),
            "progress": copy.deepcopy(report["progress"]),
            "batches": copy.deepcopy(report["batches"]),
            "samples": [
                {field: copy.deepcopy(sample.get(field)) for field in sample_fields}
                for sample in report.get("samples", [])
            ],
            "layer_error_bounds": copy.deepcopy(report.get("layer_error_bounds", [])),
            "memory": {
                "curve": copy.deepcopy(report["memory"]["curve"]),
                "events": copy.deepcopy(report["memory"]["events"]),
                "replay": copy.deepcopy(report["memory"]["replay"]),
            },
        }

    def query_batch(self, batch_id: str) -> dict[str, Any]:
        if self.current_report is None or self.latency is None:
            raise DiagnosticError(E_DATA, "no current report available", "session.current_report")
        with self._lock:
            start = self.latency.clock_ns()
            matches = [batch for batch in self.current_report.get("batches", []) if batch["batch_id"] == batch_id]
            result = copy.deepcopy(matches[0]) if matches else {}
            end = self.latency.clock_ns()
            self.latency.record("query_batch", batch_id, start, end)
            self._apply_latency()
        return result

    def query_samples_for_batch(self, batch_id: str) -> list[dict[str, Any]]:
        if self.current_report is None or self.latency is None:
            raise DiagnosticError(E_DATA, "no current report available", "session.current_report")
        with self._lock:
            start = self.latency.clock_ns()
            result = [
                copy.deepcopy(sample)
                for sample in self.current_report.get("samples", [])
                if sample.get("batch_id") == batch_id
            ]
            end = self.latency.clock_ns()
            self.latency.record("query_samples_for_batch", batch_id, start, end)
            self._apply_latency()
        return result

    def _apply_latency(self) -> None:
        if self.current_report is None or self.latency is None:
            return
        summary = self.latency.summary()
        self.current_report["ui_latency"] = summary
        codes = set(self.current_report["summary"]["codes"])
        if summary["codes"]:
            codes.add(E_LATENCY)
        else:
            codes.discard(E_LATENCY)
        self.current_report["summary"]["codes"] = sorted(codes)
        self._write_current()

    def _write_current(self) -> None:
        if self.current_report is None:
            return
        version = self.current_report["config_version"]
        path = self.report_dir / f"version_{version:04d}.json"
        write_report(path, self.current_report)
        write_report(self.report_dir / "current.json", self.current_report)

    def _write_snapshot(self, report: dict[str, Any]) -> None:
        version = report["config_version"]
        path = self.snapshot_dir / f"version_{version:04d}_{report['config']['precision']}.json"
        write_report(path, report)

    def write_session_summary(self) -> dict[str, Any]:
        if self.current_report is None:
            raise DiagnosticError(E_DATA, "no current report available", "session.current_report")
        payload = {
            "schema": "mixedprec-diagnostic-session/v1",
            "request": self.request.to_dict(),
            "artifact_digests": {
                "data_sha256": self.artifacts.data_sha256,
                "params_sha256": self.artifacts.params_sha256,
            },
            "current_version": self.current_report["config_version"],
            "current_run_id": self.current_report["run_id"],
            "current_report": {
                "config_version": self.current_report["config_version"],
                "replacement": self.current_report["replacement"],
                "summary": self.current_report["summary"],
                "ui_latency": self.current_report["ui_latency"],
            },
            "history": self.history,
            "whole_replacement": True,
            "stale_results_excluded_from_current_report": True,
        }
        write_report(self.report_dir / "session.json", payload)
        return payload
