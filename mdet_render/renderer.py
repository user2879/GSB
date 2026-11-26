"""Deterministic multimodal condition renderer with exact incremental caching."""

from __future__ import annotations

import base64
import copy
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from .cache import CacheStore, LedgerWriter
from .conditions import EncodedCondition, encode_image, encode_motion, encode_text
from .errors import BusinessError, E_LEASE_EXPIRED, E_POINT_BUDGET_PENDING
from .fixed import FixedPoint
from .lease import Lease, LocalExecutor
from .png import concat_rows, encode_rgb_png, parse_png
from .util import (
    array_digest,
    atomic_write_bytes,
    atomic_write_text,
    canonical_json,
    clamp_int,
    decode_array,
    digest_obj,
    encode_array,
    quantize_array,
    quantize_int,
    round_div,
    round_div_array,
    sha256_hex,
    stable_task_key,
)
from .workspace import Workspace


@dataclass
class TaskSpec:
    stable_key: str
    stage: str
    frame: int | None
    step: int | None
    block: int | None
    cache_key: str
    params_digest: str
    params: dict[str, Any]
    upstream: list[dict[str, Any]]
    cost: int = 0
    dependency_key: str | None = None
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class RenderResult:
    complete: bool
    code: str | None
    lease_generation: int
    frames: dict[int, dict[str, Any]]
    task_keys: dict[str, str]
    pending_tasks: list[str]
    scheduler_batches: list[dict[str, Any]]
    max_observed_workers: int
    run_id: str
    output_dir: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.complete,
            "code": self.code,
            "complete": self.complete,
            "lease_generation": self.lease_generation,
            "frames": {str(index): value for index, value in sorted(self.frames.items())},
            "task_keys": dict(sorted(self.task_keys.items())),
            "pending_tasks": self.pending_tasks,
            "scheduler_batches": self.scheduler_batches,
            "max_observed_workers": self.max_observed_workers,
            "run_id": self.run_id,
            "output_dir": self.output_dir,
        }


def _make_product(task: TaskSpec, *, content_digest: str, payload: dict[str, Any],
                  quantization: dict[str, Any], lease_generation: int) -> dict[str, Any]:
    product = {
        "schema_version": 1,
        "cache_key": task.cache_key,
        "stable_key": task.stable_key,
        "stage": task.stage,
        "frame": task.frame,
        "step": task.step,
        "block": task.block,
        "content_digest": content_digest,
        "params_digest": task.params_digest,
        "quantization": quantization,
        "lease_generation": lease_generation,
        "upstream": task.upstream,
        "payload": payload,
    }
    product["artifact_digest"] = digest_obj(payload)
    return product


def _upstream(role: str, stage: str, product: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": role,
        "stage": stage,
        "cache_key": product["cache_key"],
        "content_digest": product["content_digest"],
    }


def _split_ranges(total: int, count: int) -> list[tuple[int, int]]:
    base, remainder = divmod(total, count)
    ranges: list[tuple[int, int]] = []
    start = 0
    for index in range(count):
        size = base + (1 if index < remainder else 0)
        ranges.append((start, start + size))
        start += size
    return ranges


class Renderer:
    """A render session preserving one lease generation across incremental edits."""

    def __init__(self, workspace: Workspace, *, cache_dir: str | Path, ledger_path: str | Path,
                 executor: LocalExecutor | None = None) -> None:
        self.workspace = workspace
        self.store = CacheStore(cache_dir)
        self.ledger_path = Path(ledger_path)
        self.executor = executor or LocalExecutor()
        self._owns_executor = executor is None
        self._lease: Lease | None = None
        self._lease_targets: tuple[int, ...] = ()
        self._run_counter = 0
        self._active_workers = 0
        self._max_active_workers = 0
        self._metric_lock = threading.Lock()
        self._previous_manifest: dict[str, Any] = {"format_version": 1, "tasks": {}}
        self._ledger: LedgerWriter | None = None
        self._run_id = ""
        self._products: dict[str, dict[str, Any]] = {}
        self._quantization: dict[str, Any] = {}
        self._lease_generation = 0
        self._manifest_tasks: dict[str, dict[str, Any]] = {}
        self._batch_logs: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Lease integration
    # ------------------------------------------------------------------
    def _ensure_lease(self) -> Lease:
        targets = tuple(range(self.workspace.config.max_denoise_steps))
        ttl_ms = self.workspace.config.lease_ttl_ms
        try:
            if self._lease is None:
                lease = self.executor.acquire(targets, ttl_ms)
            else:
                lease = self.executor.renew(self._lease.lease_id, self._lease.generation, ttl_ms)
        except BusinessError as exc:
            self._lease = None
            if self._ledger is not None and exc.code in {E_LEASE_EXPIRED, "E_LEASE_GENERATION"}:
                self._ledger.event({
                    "code": exc.code,
                    "message": exc.message,
                    "details": exc.details,
                    "executor": self.executor.snapshot(),
                })
            raise
        self._lease = lease
        self._lease_targets = targets
        self._lease_generation = lease.generation
        return lease

    def _renew_for_batch(self) -> None:
        if self._lease is None:
            self._ensure_lease()
            return
        try:
            self._lease = self.executor.renew(
                self._lease.lease_id,
                self._lease.generation,
                self.workspace.config.lease_ttl_ms,
            )
        except BusinessError as exc:
            self._lease = None
            if self._ledger is not None:
                self._ledger.event({
                    "code": exc.code,
                    "message": exc.message,
                    "details": exc.details,
                    "executor": self.executor.snapshot(),
                })
            raise

    # ------------------------------------------------------------------
    # Task/cache helpers
    # ------------------------------------------------------------------
    def _new_task(self, *, stable_key: str, stage: str, frame: int | None, step: int | None,
                  block: int | None, params: dict[str, Any], upstream: list[dict[str, Any]],
                  cost: int = 0, dependency_key: str | None = None,
                  data: dict[str, Any] | None = None) -> TaskSpec:
        params = copy.deepcopy(params)
        params_digest = digest_obj(params)
        upstream = sorted(copy.deepcopy(upstream), key=lambda item: (item["role"], item["cache_key"]))
        key = self.store.make_key(
            stage=stage,
            upstream=upstream,
            params_digest=params_digest,
            quantization=self._quantization,
            lease_generation=self._lease_generation,
        )
        return TaskSpec(stable_key=stable_key, stage=stage, frame=frame, step=step, block=block,
                        cache_key=key, params_digest=params_digest, params=params,
                        upstream=upstream, cost=cost, dependency_key=dependency_key,
                        data=data or {})

    def _invalidation_basis(self, task: TaskSpec) -> list[str]:
        previous = self._previous_manifest.get("tasks", {}).get(task.stable_key)
        if not isinstance(previous, dict):
            return ["new_task"]
        basis: list[str] = []
        if previous.get("quantization") != self._quantization:
            basis.append("quantization_changed")
        if previous.get("lease_generation") != self._lease_generation:
            basis.append("lease_generation_changed")
        if previous.get("params_digest") != task.params_digest:
            basis.append("parameters_changed")
        previous_upstream = {
            item.get("role"): item for item in previous.get("upstream", []) if isinstance(item, dict)
        }
        current_upstream = {item["role"]: item for item in task.upstream}
        for role in sorted(set(previous_upstream) | set(current_upstream)):
            old = previous_upstream.get(role)
            new = current_upstream.get(role)
            if old is None or new is None or old.get("cache_key") != new.get("cache_key"):
                basis.append(f"upstream_changed:{role}")
        if not basis:
            basis.append("cache_artifact_missing")
        return basis

    def _record_task(self, task: TaskSpec, product: dict[str, Any], outcome: str,
                     extra: dict[str, Any] | None = None) -> dict[str, Any]:
        assert self._ledger is not None
        basis = [] if outcome == "reused" else self._invalidation_basis(task)
        entry = {
            "stage": task.stage,
            "stable_key": task.stable_key,
            "cache_key": task.cache_key,
            "outcome": outcome,
            "invalidation_basis": basis,
            "frame": task.frame,
            "step": task.step,
            "block": task.block,
            "quantization": self._quantization,
            "lease_generation": self._lease_generation,
            "upstream": task.upstream,
            "artifact_digest": product.get("artifact_digest"),
            "content_digest": product.get("content_digest"),
        }
        if extra:
            entry.update(extra)
        self._ledger.task(entry)
        manifest_entry = {
            "stable_key": task.stable_key,
            "stage": task.stage,
            "cache_key": task.cache_key,
            "content_digest": product.get("content_digest"),
            "params_digest": task.params_digest,
            "quantization": self._quantization,
            "lease_generation": self._lease_generation,
            "upstream": task.upstream,
            "artifact_digest": product.get("artifact_digest"),
            "frame": task.frame,
            "step": task.step,
            "block": task.block,
        }
        self._manifest_tasks[task.stable_key] = manifest_entry
        return entry

    def _materialize(self, task: TaskSpec, *, producer: Callable[[], tuple[str, dict[str, Any]]]) -> dict[str, Any]:
        if self.store.has(task.cache_key):
            product = self.store.load(task.cache_key)
            if product.get("stage") != task.stage or product.get("stable_key") != task.stable_key:
                raise BusinessError("E_CACHE_ARTIFACT", "cached stage/stable key mismatch",
                                    {"cache_key": task.cache_key, "expected_stage": task.stage,
                                     "actual_stage": product.get("stage"),
                                     "expected_stable_key": task.stable_key,
                                     "actual_stable_key": product.get("stable_key")})
            self._products[task.stable_key] = product
            self._record_task(task, product, "reused")
            return product
        content_digest, payload = producer()
        product = _make_product(task, content_digest=content_digest, payload=payload,
                                quantization=self._quantization,
                                lease_generation=self._lease_generation)
        self.store.save(product)
        self._products[task.stable_key] = product
        self._record_task(task, product, "computed")
        return product

    # ------------------------------------------------------------------
    # Numerical kernels
    # ------------------------------------------------------------------
    def _fuse(self, frame: int, conditions: dict[str, dict[str, Any]]) -> tuple[str, dict[str, Any]]:
        fixed = self.workspace.config.fixed
        dim = self.workspace.config.condition_dim
        vectors = {name: np.asarray(product["payload"]["vector"], dtype=np.int64)
                   for name, product in conditions.items()}
        accumulator = np.zeros(dim, dtype=np.int64)
        trace: list[dict[str, Any]] = []
        for condition_type in sorted(vectors):
            weight = float(self.workspace.config.weights[condition_type])
            weight_int = fixed.from_real(weight)
            for dimension in range(dim):
                value = int(vectors[condition_type][dimension])
                scaled_raw = round_div(value * weight_int, fixed.scale)
                scaled, scaling_residual = quantize_int(scaled_raw, fixed.quant_level)
                before = int(accumulator[dimension])
                summed_raw = before + scaled
                summed, addition_residual = quantize_int(summed_raw, fixed.quant_level)
                clamped = clamp_int(summed, fixed.low, fixed.high)
                saturation_residual = clamped - summed
                accumulator[dimension] = clamped
                trace.append({
                    "condition_type": condition_type,
                    "dimension": dimension,
                    "weight": weight,
                    "weight_fixed": weight_int,
                    "quantization_step": fixed.quant_step,
                    "numeric_quantization_step": fixed.numeric_quant_step,
                    "integer_result": clamped,
                    "residual": int(scaling_residual + addition_residual + saturation_residual),
                })
        digest = array_digest(accumulator)
        return digest, {
            "vector": accumulator.tolist(),
            "condition_types": sorted(vectors),
            "missing_condition_types": sorted({"text", "image", "motion"} - set(vectors)),
            "weights": {name: self.workspace.config.weights[name] for name in sorted(vectors)},
            "trace_order": "condition_type_ascending_then_dimension_ascending",
            "trace": trace,
            "quantization": fixed.as_dict(),
        }

    def _latent_init(self, fusion: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        config = self.workspace.config
        fixed = config.fixed
        params = {
            "algorithm": "latent-pcg64-v1",
            "fusion_content_digest": fusion["content_digest"],
            "seed": config.latent_seed,
            "height": config.height,
            "width": config.width,
            "channels": 3,
            "fixed": fixed.fingerprint(),
        }
        seed_material = sha256_hex(canonical_json(params))
        rng = np.random.Generator(np.random.PCG64(int(seed_material, 16)))
        amplitude = max(1, fixed.scale // 4)
        raw = rng.integers(-amplitude, amplitude + 1, size=(config.height, config.width, 3),
                           dtype=np.int64)
        state, residual = quantize_array(raw, fixed.quant_level)
        state = np.clip(state, fixed.low, fixed.high)
        digest = array_digest(state)
        return digest, {
            "state": encode_array(state),
            "shape": [config.height, config.width, 3],
            "seed_material_digest": seed_material,
            "quantization_residual": int(residual.sum()),
            "initial_state_digest": digest,
            "fixed": fixed.as_dict(),
        }

    def _noise_table(self) -> tuple[np.ndarray, str]:
        config = self.workspace.config
        fixed = config.fixed
        params = {
            "algorithm": "noise-table-pcg64-v1",
            "seed": config.noise_seed,
            "size": config.noise_table_size,
            "shape": [config.height, config.width, 3],
            "fixed": fixed.fingerprint(),
        }
        seed_material = sha256_hex(canonical_json(params))
        rng = np.random.Generator(np.random.PCG64(int(seed_material, 16)))
        amplitude = max(1, fixed.scale // 8)
        raw = rng.integers(-amplitude, amplitude + 1,
                           size=(config.noise_table_size, config.height, config.width, 3),
                           dtype=np.int64)
        table, _ = quantize_array(raw, fixed.quant_level)
        table = np.clip(table, fixed.low, fixed.high)
        return table, array_digest(table)

    def _denoise_step(self, previous_state: np.ndarray, fusion_vector: np.ndarray,
                      noise_entry: np.ndarray, frame: int, step: int) -> tuple[str, dict[str, Any]]:
        config = self.workspace.config
        fixed = config.fixed
        guide_int = fixed.from_real(config.guidance_strength)
        points = config.width * config.height * 3
        indices = (np.arange(points, dtype=np.int64) * 131 + frame * 257 + step * 97) % config.condition_dim
        condition_term = fusion_vector[indices].reshape((config.height, config.width, 3))
        scaled_raw = round_div_array(condition_term * guide_int, fixed.scale)
        scaled, scaling_residual = quantize_array(scaled_raw, fixed.quant_level)
        combined = previous_state.astype(np.int64) + noise_entry.astype(np.int64) + scaled
        quantized, rounding_residual = quantize_array(combined, fixed.quant_level)
        state = np.clip(quantized, fixed.low, fixed.high)
        saturation_residual = state - quantized
        total_residual = int(scaling_residual.sum() + rounding_residual.sum() + saturation_residual.sum())
        below = int(np.count_nonzero(rounding_residual > 0))
        above = int(np.count_nonzero(rounding_residual < 0))
        if below and above:
            direction = "mixed"
        elif below:
            direction = "down"
        elif above:
            direction = "up"
        else:
            direction = "exact"
        digest = array_digest(state)
        summary = {
            "output_digest": digest,
            "rounding_direction": direction,
            "rounding_counts": {"down": below, "up": above,
                                "exact": int(rounding_residual.size - below - above)},
            "residual": total_residual,
            "max_abs_residual": int(np.max(np.abs(rounding_residual))) if rounding_residual.size else 0,
            "guide_strength": config.guidance_strength,
            "guide_fixed": guide_int,
            "fixed": fixed.as_dict(),
        }
        payload = {
            "state": encode_array(state),
            "summary": summary,
        }
        return digest, payload

    def _projection(self) -> tuple[np.ndarray, np.ndarray]:
        config = self.workspace.config
        fixed = config.fixed
        seed = sha256_hex(canonical_json({"seed": config.projection_seed,
                                          "fixed": fixed.fingerprint(),
                                          "algorithm": "rgb-projection-v1"}))
        rng = np.random.Generator(np.random.PCG64(int(seed, 16)))
        offsets = rng.integers(-48, 49, size=(3, 3), dtype=np.int64) / 64.0
        matrix_real = offsets + np.eye(3, dtype=np.float64) * 0.75
        bias_real = rng.integers(-4, 5, size=(3,), dtype=np.int64) / 100.0
        matrix = np.array([[fixed.from_real(float(value)) for value in row] for row in matrix_real],
                          dtype=np.int64)
        bias = np.array([fixed.from_real(float(value)) for value in bias_real], dtype=np.int64)
        return matrix, bias

    def _decode_rows(self, latent_rows: np.ndarray, matrix: np.ndarray, bias: np.ndarray) -> np.ndarray:
        fixed = self.workspace.config.fixed
        if latent_rows.ndim != 3 or latent_rows.shape[2] != 3:
            raise BusinessError("E_DECODE_SHAPE", "latent block row must have shape HxWx3",
                                {"shape": list(latent_rows.shape)})
        values = np.asarray(latent_rows, dtype=np.int64)
        numerator = np.einsum("hwc,dc->hwd", values, matrix, optimize=True)
        numerator = numerator + bias.reshape(1, 1, 3) * fixed.scale
        denominator = fixed.scale * fixed.scale
        pixels = round_div_array(numerator * 255, denominator) + 128
        return np.clip(pixels, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------
    def _encode_conditions(self) -> dict[int, dict[str, dict[str, Any]]]:
        config = self.workspace.config
        fixed = config.fixed
        result: dict[int, dict[str, dict[str, Any]]] = {}
        for frame_index in sorted(self.workspace.frames):
            frame = self.workspace.frames[frame_index]
            conditions: dict[str, dict[str, Any]] = {}
            text = encode_text(frame.prompt, self.workspace.vocabulary, fixed)
            encoded: dict[str, tuple[EncodedCondition, dict[str, Any], str]] = {}
            encoded["text"] = (
                text,
                {"algorithm": "text-bow-v1", "frame": frame_index, "dimension": config.condition_dim,
                 "vocabulary_digest": self.workspace.vocabulary.digest,
                 "prompt_digest": digest_obj(frame.prompt.as_dict()), "fixed": fixed.fingerprint()},
                digest_obj(frame.prompt.as_dict()),
            )
            image = self.workspace.image_for(frame_index)
            if image is None:
                pass
            else:
                encoded_image = encode_image(image, config.image_grid_rows, config.image_grid_cols, fixed)
                encoded["image"] = (
                    encoded_image,
                    {"algorithm": "image-block-stats-v1", "frame": frame_index,
                     "dimension": config.condition_dim, "grid_rows": config.image_grid_rows,
                     "grid_cols": config.image_grid_cols, "image_digest": image.digest,
                     "fixed": fixed.fingerprint()},
                    image.digest,
                )
            motion = encode_motion(self.workspace.motion, fixed, frame=frame_index)
            encoded["motion"] = (
                motion,
                {"algorithm": "motion-fixed-v1", "frame": frame_index,
                 "dimension": config.condition_dim, "motion_table_digest": self.workspace.motion.digest,
                 "fixed": fixed.fingerprint()},
                self.workspace.motion.digest,
            )
            for condition_type in sorted(encoded):
                condition, params, source_digest = encoded[condition_type]
                task = self._new_task(
                    stable_key=stable_task_key("encode", frame=frame_index, kind=condition_type),
                    stage="encode",
                    frame=frame_index,
                    step=None,
                    block=None,
                    params=params,
                    upstream=[{"role": "source", "stage": "source",
                               "cache_key": source_digest, "content_digest": source_digest}],
                    data={"condition_type": condition_type},
                )
                product = self._materialize(task, producer=lambda c=condition: (
                    c.digest,
                    {"condition_type": c.condition_type, "vector": c.values.tolist(),
                     "digest": c.digest, "details": c.details},
                ))
                conditions[condition_type] = product
            result[frame_index] = conditions
        return result

    def _fuse_frames(self, conditions_by_frame: dict[int, dict[str, dict[str, Any]]]
                     ) -> dict[int, dict[str, Any]]:
        config = self.workspace.config
        fused: dict[int, dict[str, Any]] = {}
        for frame_index in sorted(conditions_by_frame):
            conditions = conditions_by_frame[frame_index]
            params = {
                "algorithm": "fixed-fusion-v1",
                "frame": frame_index,
                "dimension": config.condition_dim,
                "weights": {name: config.weights[name] for name in sorted(config.weights)},
                "present_types": sorted(conditions),
                "missing_types": sorted({"text", "image", "motion"} - set(conditions)),
                "fixed": config.fixed.fingerprint(),
                "order": "condition_type_ascending_then_dimension_ascending",
            }
            upstream = [_upstream(f"condition:{name}", "encode", product)
                        for name, product in sorted(conditions.items())]
            task = self._new_task(
                stable_key=stable_task_key("fusion", frame=frame_index),
                stage="fusion",
                frame=frame_index,
                step=None,
                block=None,
                params=params,
                upstream=upstream,
            )
            fused[frame_index] = self._materialize(
                task, producer=lambda frame_index=frame_index, conditions=conditions:
                self._fuse(frame_index, conditions),
            )
        return fused

    def _init_latents(self, fused_by_frame: dict[int, dict[str, Any]]) -> dict[int, dict[str, Any]]:
        config = self.workspace.config
        latents: dict[int, dict[str, Any]] = {}
        for frame_index in sorted(fused_by_frame):
            fusion = fused_by_frame[frame_index]
            params = {
                "algorithm": "latent-init-v1",
                "frame": frame_index,
                "seed": config.latent_seed,
                "height": config.height,
                "width": config.width,
                "channels": 3,
                "fixed": config.fixed.fingerprint(),
            }
            task = self._new_task(
                stable_key=stable_task_key("latent", frame=frame_index),
                stage="latent",
                frame=frame_index,
                step=None,
                block=None,
                params=params,
                upstream=[_upstream("fusion", "fusion", fusion)],
            )
            latents[frame_index] = self._materialize(task, producer=lambda fusion=fusion:
                                                      self._latent_init(fusion))
        return latents

    def _build_denoise_tasks(self, fused_by_frame: dict[int, dict[str, Any]],
                             latent_by_frame: dict[int, dict[str, Any]],
                             noise_entry_digests: list[str]) -> dict[str, TaskSpec]:
        config = self.workspace.config
        fixed = config.fixed
        guide_int = fixed.from_real(config.guidance_strength)
        tasks: dict[str, TaskSpec] = {}
        for frame_index in sorted(latent_by_frame):
            latent = latent_by_frame[frame_index]
            fusion = fused_by_frame[frame_index]
            previous_key = latent["cache_key"]
            previous_digest = latent["content_digest"]
            previous_stage = "latent"
            for step in range(config.denoise_steps):
                stable = stable_task_key("denoise", frame=frame_index, step=step)
                noise_digest = noise_entry_digests[(frame_index + step) % len(noise_entry_digests)]
                params = {
                    "algorithm": "denoise-step-v2",
                    "frame": frame_index,
                    "step": step,
                    "guide_strength": config.guidance_strength,
                    "guide_fixed": guide_int,
                    "noise_entry_digest": noise_digest,
                    "operation_order": "previous+noise+condition",
                    "fixed": fixed.fingerprint(),
                }
                upstream = [
                    {"role": "previous_state", "stage": previous_stage,
                     "cache_key": previous_key,
                     "content_digest": previous_digest},
                    _upstream("condition", "fusion", fusion),
                    {"role": "noise", "stage": "noise", "cache_key": noise_digest,
                     "content_digest": noise_digest},
                ]
                task = self._new_task(
                    stable_key=stable,
                    stage="denoise",
                    frame=frame_index,
                    step=step,
                    block=None,
                    params=params,
                    upstream=upstream,
                    cost=config.denoise_points_per_step,
                    dependency_key=None if step == 0 else
                    stable_task_key("denoise", frame=frame_index, step=step - 1),
                    data={"fusion": fusion, "latent": latent,
                          "noise_index": (frame_index + step) % len(noise_entry_digests)},
                )
                tasks[stable] = task
                previous_key = task.cache_key
                # The previous key is itself a content-addressed fingerprint, so
                # the next step can be keyed before its predecessor executes.
                previous_digest = task.cache_key
                previous_stage = "denoise"
        return tasks

    def _produce_denoise(self, task: TaskSpec, noise_table: np.ndarray) -> tuple[str, dict[str, Any]]:
        frame = int(task.frame)
        step = int(task.step)
        if step == 0:
            previous_state = decode_array(task.data["latent"]["payload"]["state"])
        else:
            dependency = self._products[stable_task_key("denoise", frame=frame, step=step - 1)]
            previous_state = decode_array(dependency["payload"]["state"])
        fusion_vector = np.asarray(task.data["fusion"]["payload"]["vector"], dtype=np.int64)
        noise_entry = noise_table[int(task.data["noise_index"])]
        return self._denoise_step(previous_state, fusion_vector, noise_entry, frame, step)

    def _run_denoise_scheduler(self, tasks: dict[str, TaskSpec], noise_table: np.ndarray,
                               pool: ThreadPoolExecutor, max_batches: int | None) -> tuple[str, bool, list[str]]:
        config = self.workspace.config
        by_key = {task.stable_key: task for task in tasks.values()}
        pending: set[str] = set(by_key)
        completed: set[str] = set()
        deferred_order: list[str] = []
        deferred_rank: dict[str, int] = {}
        batch_number = 0
        scheduler_code = "OK"
        while pending:
            if max_batches is not None and batch_number >= max_batches:
                scheduler_code = scheduler_code if scheduler_code != "OK" else "E_RENDER_INCOMPLETE"
                break
            ready = [key for key in pending
                     if by_key[key].dependency_key is None or by_key[key].dependency_key in completed]
            ready.sort(key=lambda key: (0, deferred_rank[key], key) if key in deferred_rank else (1, 0, key))
            if not ready:
                raise BusinessError("E_SCHEDULER_DEADLOCK", "no ready denoise tasks remain",
                                    {"pending": sorted(pending)})
            budget = config.point_budget
            selected_hits: list[str] = []
            selected_misses: list[str] = []
            deferred: list[str] = []
            for key in ready:
                task = by_key[key]
                if self.store.has(task.cache_key):
                    selected_hits.append(key)
                elif task.cost <= budget:
                    budget -= task.cost
                    selected_misses.append(key)
                else:
                    deferred.append(key)
            selected = selected_hits + selected_misses
            if not selected:
                raise BusinessError("E_POINT_BUDGET", "point budget cannot fit any ready denoise step",
                                    {"point_budget": config.point_budget,
                                     "ready": sorted(ready)})
            self._renew_for_batch()
            miss_results: dict[str, dict[str, Any]] = {}
            if selected_misses:
                futures = {
                    key: pool.submit(self._tracked, self._compute_denoise_task, by_key[key], noise_table)
                    for key in sorted(selected_misses)
                }
                for key in sorted(selected_misses):
                    miss_results[key] = futures[key].result()
            products: dict[str, dict[str, Any]] = {}
            for key in sorted(selected):
                task = by_key[key]
                if key in miss_results:
                    product = miss_results[key]
                    self.store.save(product)
                    self._products[task.stable_key] = product
                    products[key] = product
                    self._record_task(task, product, "computed")
                else:
                    product = self.store.load(task.cache_key)
                    if product.get("stage") != "denoise" or product.get("stable_key") != task.stable_key:
                        raise BusinessError("E_CACHE_ARTIFACT", "cached denoise product is incompatible",
                                            {"cache_key": task.cache_key})
                    self._products[task.stable_key] = product
                    products[key] = product
                    self._record_task(task, product, "reused")
                completed.add(key)
            pending.difference_update(selected)
            selected_set = set(selected)
            deferred_order = [key for key in deferred_order if key not in selected_set and key in pending]
            for key in deferred:
                if key in pending and key not in deferred_rank:
                    deferred_rank[key] = len(deferred_rank)
                    deferred_order.append(key)
            # Keep only live deferred tasks in rank order.  Older pending tasks
            # always outrank tasks that only became ready in a later batch.
            deferred_rank = {key: index for index, key in enumerate(deferred_order)}
            batch_number += 1
            batch = {
                "batch": batch_number,
                "selected": sorted(selected),
                "computed": sorted(selected_misses),
                "reused": sorted(selected_hits),
                "deferred": sorted(deferred),
                "pending": sorted(pending),
                "points_budget": config.point_budget,
                "points_used": config.point_budget - budget,
                "code": E_POINT_BUDGET_PENDING if deferred else "OK",
            }
            self._batch_logs.append(batch)
            assert self._ledger is not None
            self._ledger.scheduler_batch(batch)
            if deferred:
                scheduler_code = E_POINT_BUDGET_PENDING
            elif scheduler_code == "OK":
                scheduler_code = "OK"
        if pending:
            scheduler_code = E_POINT_BUDGET_PENDING if scheduler_code == E_POINT_BUDGET_PENDING else "E_RENDER_INCOMPLETE"
        return scheduler_code, not pending, sorted(pending)

    def _compute_denoise_task(self, task: TaskSpec, noise_table: np.ndarray) -> dict[str, Any]:
        content_digest, payload = self._produce_denoise(task, noise_table)
        return _make_product(task, content_digest=content_digest, payload=payload,
                             quantization=self._quantization,
                             lease_generation=self._lease_generation)

    def _decode_frames(self, fused_by_frame: dict[int, dict[str, Any]],
                       latent_by_frame: dict[int, dict[str, Any]]
                       ) -> dict[int, dict[str, Any]]:
        config = self.workspace.config
        matrix, bias = self._projection()
        matrix_payload = {"matrix": matrix.tolist(), "bias": bias.tolist()}
        decodes: dict[int, dict[str, Any]] = {}
        for frame_index in sorted(latent_by_frame):
            final_key = stable_task_key("denoise", frame=frame_index,
                                        step=config.denoise_steps - 1)
            final_product = self._products[final_key]
            latent = latent_by_frame[frame_index]
            state = decode_array(final_product["payload"]["state"])
            ranges = _split_ranges(config.height, config.decode_block_rows)
            row_digests = [array_digest(state[start:end]) for start, end in ranges]
            params = {
                "algorithm": "frame-decode-plan-v1",
                "frame": frame_index,
                "height": config.height,
                "width": config.width,
                "block_rows": config.decode_block_rows,
                "row_ranges": [list(range_pair) for range_pair in ranges],
                "projection": matrix_payload,
                "fixed": config.fixed.fingerprint(),
            }
            task = self._new_task(
                stable_key=stable_task_key("decode", frame=frame_index),
                stage="decode",
                frame=frame_index,
                step=None,
                block=None,
                params=params,
                upstream=[_upstream("latent", "denoise", final_product)],
            )
            producer = lambda ranges=ranges, row_digests=row_digests, matrix_payload=matrix_payload, frame_index=frame_index: (
                digest_obj({"frame": frame_index, "ranges": [list(r) for r in ranges],
                            "rows": row_digests, "projection": matrix_payload}),
                {
                    "row_ranges": [list(range_pair) for range_pair in ranges],
                    "row_digests": row_digests,
                    "projection": matrix_payload,
                    "completion_order": [stable_task_key("block", frame=frame_index, block=index)
                                         for index in range(len(ranges))],
                    "algorithm": "one-row-latent-slice-per-pixel-row",
                },
            )
            decodes[frame_index] = self._materialize(task, producer=producer)
        return decodes

    def _block_task(self, frame_index: int, block_index: int, decode: dict[str, Any],
                    row_range: tuple[int, int]) -> TaskSpec:
        config = self.workspace.config
        start, end = row_range
        params = {
            "algorithm": "raster-block-decode-v1",
            "frame": frame_index,
            "block": block_index,
            "row_start": start,
            "row_end": end,
            "width": config.width,
            "projection": decode["payload"]["projection"],
            "fixed": config.fixed.fingerprint(),
            "latent_scope": "own_row_range_only",
        }
        return self._new_task(
            stable_key=stable_task_key("block", frame=frame_index, block=block_index),
            stage="block",
            frame=frame_index,
            step=None,
            block=block_index,
            params=params,
            upstream=[_upstream("decode", "decode", decode)],
            data={"row_range": (start, end)},
        )

    def _compute_block(self, task: TaskSpec, latent_rows: np.ndarray,
                       matrix: np.ndarray, bias: np.ndarray) -> dict[str, Any]:
        start, end = task.data["row_range"]
        pixels = self._decode_rows(latent_rows, matrix, bias)
        png = encode_rgb_png(pixels)
        pixel_digest = sha256_hex(b"RGB8|" + pixels.tobytes(order="C"))
        payload = {
            "row_start": start,
            "row_end": end,
            "height": end - start,
            "width": self.workspace.config.width,
            "pixels": encode_array(pixels, dtype="|u1"),
            "pixel_digest": pixel_digest,
            "png_b64": base64.b64encode(png).decode("ascii"),
            "png_digest": sha256_hex(png),
            "latent_scope": [start, end],
        }
        return _make_product(task, content_digest=pixel_digest, payload=payload,
                             quantization=self._quantization,
                             lease_generation=self._lease_generation)

    def _decode_blocks(self, decodes: dict[int, dict[str, Any]], pool: ThreadPoolExecutor,
                       output_dir: Path | None) -> dict[int, dict[str, Any]]:
        config = self.workspace.config
        matrix, bias = self._projection()
        results: dict[int, dict[str, Any]] = {}
        for frame_index in sorted(decodes):
            decode = decodes[frame_index]
            final_product = self._products[stable_task_key(
                "denoise", frame=frame_index, step=config.denoise_steps - 1)]
            state = decode_array(final_product["payload"]["state"])
            ranges = [tuple(item) for item in decode["payload"]["row_ranges"]]
            tasks = [self._block_task(frame_index, index, decode, row_range)
                     for index, row_range in enumerate(ranges)]
            misses: list[TaskSpec] = []
            loaded: dict[str, dict[str, Any]] = {}
            for task in tasks:
                if self.store.has(task.cache_key):
                    product = self.store.load(task.cache_key)
                    if product.get("stage") != "block" or product.get("stable_key") != task.stable_key:
                        raise BusinessError("E_CACHE_ARTIFACT", "cached block product is incompatible",
                                            {"cache_key": task.cache_key})
                    loaded[task.stable_key] = product
                else:
                    misses.append(task)
            computed: dict[str, dict[str, Any]] = {}
            if misses:
                futures = {
                    task.stable_key: pool.submit(
                        self._tracked, self._compute_block, task,
                        state[task.data["row_range"][0]:task.data["row_range"][1]], matrix, bias,
                    )
                    for task in misses
                }
                for task in misses:
                    computed[task.stable_key] = futures[task.stable_key].result()
            block_entries: list[dict[str, Any]] = []
            block_arrays: list[np.ndarray] = []
            current_block_names = {f"row_{task.block:04d}.png" for task in tasks}
            for task in tasks:
                start, end = task.data["row_range"]
                if task.stable_key in computed:
                    product = computed[task.stable_key]
                    self.store.save(product)
                    self._products[task.stable_key] = product
                    self._record_task(task, product, "computed")
                else:
                    product = loaded[task.stable_key]
                    self._products[task.stable_key] = product
                    self._record_task(task, product, "reused")
                pixels = decode_array(product["payload"]["pixels"])
                if pixels.shape != (end - start, config.width, 3):
                    raise BusinessError("E_CACHE_ARTIFACT", "cached block pixel shape is invalid",
                                        {"stable_key": task.stable_key, "shape": list(pixels.shape)})
                if start == 0 and pixels.shape[0] <= 0:
                    raise BusinessError("E_DECODE_EMPTY", "decoded block row is empty")
                block_arrays.append(pixels)
                block_entries.append({
                    "block": task.block,
                    "row_start": start,
                    "row_end": end,
                    "cache_key": task.cache_key,
                    "content_digest": product["content_digest"],
                    "pixel_digest": product["payload"]["pixel_digest"],
                    "png_digest": product["payload"]["png_digest"],
                })
                if output_dir is not None:
                    block_dir = output_dir / "blocks" / f"frame_{frame_index:04d}"
                    png_path = block_dir / f"row_{task.block:04d}.png"
                    atomic_write_bytes(png_path, base64.b64decode(product["payload"]["png_b64"]))
            if output_dir is not None:
                block_dir = output_dir / "blocks" / f"frame_{frame_index:04d}"
                for stale in block_dir.glob("row_*.png"):
                    if stale.name not in current_block_names:
                        stale.unlink()
            assembled = concat_rows(block_arrays)
            frame_png = encode_rgb_png(assembled)
            pixel_digest = sha256_hex(b"RGB8|" + assembled.tobytes(order="C"))
            frame_entry = {
                "frame": frame_index,
                "width": config.width,
                "height": config.height,
                "row_count": len(block_entries),
                "pixel_digest": pixel_digest,
                "png_digest": sha256_hex(frame_png),
                "blocks": block_entries,
                "completion_order": [
                    stable_task_key("block", frame=frame_index, block=entry["block"])
                    for entry in block_entries
                ],
            }
            if output_dir is not None:
                frame_path = output_dir / "frames" / f"frame_{frame_index:04d}.png"
                atomic_write_bytes(frame_path, frame_png)
                frame_entry["png_path"] = str(frame_path)
            results[frame_index] = frame_entry
        return results

    # ------------------------------------------------------------------
    # Run orchestration
    # ------------------------------------------------------------------
    def _tracked(self, function: Callable[..., Any], *args: Any) -> Any:
        with self._metric_lock:
            self._active_workers += 1
            self._max_active_workers = max(self._max_active_workers, self._active_workers)
        try:
            return function(*args)
        finally:
            with self._metric_lock:
                self._active_workers -= 1

    def run(self, *, output_dir: str | Path | None = None, max_batches: int | None = None,
            write_outputs: bool = True) -> RenderResult:
        self.workspace.validate_assets()
        config = self.workspace.config
        self._run_counter += 1
        self._run_id = f"run-{self._run_counter:04d}-{sha256_hex(canonical_json(self.workspace.command_snapshot()))[:12]}"
        self._ledger = LedgerWriter(self.ledger_path, self._run_id)
        self._previous_manifest = self.store.load_manifest()
        self._products = {}
        self._manifest_tasks = {}
        self._batch_logs = []
        self._active_workers = 0
        self._max_active_workers = 0
        lease = self._ensure_lease()
        self._quantization = config.fixed.as_dict()
        self._ledger.run_header({
            "workspace": self.workspace.command_snapshot(),
            "cache_dir": str(self.store.root),
            "lease": lease.as_dict(),
            "max_workers": config.max_workers,
            "point_budget": config.point_budget,
        })
        out_path = Path(output_dir) if output_dir is not None and write_outputs else None
        if out_path is not None:
            out_path.mkdir(parents=True, exist_ok=True)
        noise_table, _noise_digest = self._noise_table()
        noise_entry_digests = [array_digest(noise_table[index]) for index in range(noise_table.shape[0])]
        with ThreadPoolExecutor(max_workers=config.max_workers, thread_name_prefix="mdet") as pool:
            conditions = self._encode_conditions()
            fused = self._fuse_frames(conditions)
            latents = self._init_latents(fused)
            denoise_tasks = self._build_denoise_tasks(fused, latents, noise_entry_digests)
            scheduler_code, complete, pending = self._run_denoise_scheduler(
                denoise_tasks, noise_table, pool, max_batches)
            frames: dict[int, dict[str, Any]] = {}
            if complete:
                decodes = self._decode_frames(fused, latents)
                frames = self._decode_blocks(decodes, pool, out_path)
            else:
                decodes = {}
            task_keys = {stable: product["cache_key"] for stable, product in sorted(self._products.items())}
        result = RenderResult(
            complete=complete,
            code=None if complete else scheduler_code,
            lease_generation=self._lease_generation,
            frames=frames,
            task_keys=task_keys,
            pending_tasks=pending,
            scheduler_batches=self._batch_logs,
            max_observed_workers=self._max_active_workers,
            run_id=self._run_id,
            output_dir=str(out_path) if out_path is not None else None,
        )
        if complete:
            manifest = {
                "format_version": 1,
                "run_id": self._run_id,
                "workspace": self.workspace.command_snapshot(),
                "lease_generation": self._lease_generation,
                "quantization": self._quantization,
                "tasks": dict(sorted(self._manifest_tasks.items())),
                "frames": {str(index): value for index, value in sorted(frames.items())},
            }
            if out_path is not None:
                atomic_write_text(out_path / "render_manifest.json",
                                  canonical_json(manifest) + "\n")
            self.store.save_manifest(manifest)
        assert self._ledger is not None
        self._ledger.run_end({
            "complete": complete,
            "code": result.code,
            "pending_tasks": pending,
            "max_observed_workers": self._max_active_workers,
            "frames": {str(index): {"pixel_digest": item["pixel_digest"],
                                    "png_digest": item["png_digest"]}
                       for index, item in sorted(frames.items())},
            "lease_generation": self._lease_generation,
            "executor": self.executor.snapshot(),
        })
        return result

    def close(self) -> None:
        if self._lease is not None:
            try:
                self.executor.release(self._lease.lease_id, self._lease.generation)
            except BusinessError:
                self.executor.reclaim_expired()
            self._lease = None
        if self._owns_executor:
            self.executor.close()
