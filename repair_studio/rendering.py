from __future__ import annotations

import hashlib
import io
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
from PIL import Image

from .cache import CacheStore
from .config import RenderParams, WEIGHT_SCALE, sha256_json
from .errors import E_RENDER_FAILED, StudioError
from .imaging import LoadedSample, StyleStats, array_sha256


LUT_VERSION = 1
FILL_VERSION = 1
FILL_WEIGHT_SCALE = 1 << 20
HALF_UP = "half_up"


ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class StageTask:
    sort_key: str
    block_row: int
    start: int
    end: int


@dataclass
class TaskResult:
    task: StageTask
    values: np.ndarray
    filled_count: int = 0
    override_count: int = 0


@dataclass
class RenderOutcome:
    params: RenderParams
    params_fingerprint: str
    final_pixels: np.ndarray
    final_png_bytes: bytes
    final_png_sha256: str
    stage_summaries: dict[str, dict[str, Any]]
    reused_stages: list[str]
    invalidated_stages: list[str]
    expected_invalidated_stages: list[str] = field(default_factory=list)
    task_order: list[str] = field(default_factory=list)
    max_workers_configured: int = 1
    max_workers_observed: int = 1
    stage_observed_workers: dict[str, int] = field(default_factory=dict)

    def audit(self) -> dict[str, Any]:
        return {
            "reused_stages": list(self.reused_stages),
            "invalidated_stages": list(self.invalidated_stages),
            "expected_invalidated_stages": list(self.expected_invalidated_stages),
            "reused_count": len(self.reused_stages),
            "invalidated_count": len(self.invalidated_stages),
        }

    def preview_summary(self) -> dict[str, Any]:
        return {
            "params": self.params.as_dict(),
            "params_fingerprint": self.params_fingerprint,
            "stage_summaries": self.stage_summaries,
            "artifact_versions": self.artifact_versions(),
            "audit": self.audit(),
            "final_png_sha256": self.final_png_sha256,
            "max_workers_configured": self.max_workers_configured,
            "max_workers_observed": self.max_workers_observed,
            "stage_observed_workers": dict(self.stage_observed_workers),
        }

    def artifact_versions(self) -> dict[str, dict[str, Any]]:
        return {
            stage: {
                "cache_key": summary.get("cache_key"),
                "content_sha256": summary.get("content_sha256"),
            }
            for stage, summary in self.stage_summaries.items()
        }


class ConcurrencyTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active = 0
        self.maximum = 0

    def enter(self) -> None:
        with self._lock:
            self._active += 1
            self.maximum = max(self.maximum, self._active)

    def leave(self) -> None:
        with self._lock:
            self._active -= 1


def stable_task_key(image_filename: str, block_row: int, stage: str) -> str:
    return f"{image_filename}|{block_row:08d}|{stage}"


def make_tasks(image_filename: str, height: int, chunk_rows: int, stage: str) -> list[StageTask]:
    tasks: list[StageTask] = []
    for block_row, start in enumerate(range(0, height, chunk_rows)):
        end = min(height, start + chunk_rows)
        tasks.append(
            StageTask(
                sort_key=stable_task_key(image_filename, block_row, stage),
                block_row=block_row,
                start=start,
                end=end,
            )
        )
    return tasks


def build_mask(rect: Any, width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.bool_)
    mask[rect.y : rect.y + rect.height, rect.x : rect.x + rect.width] = True
    return mask


def q_array_to_u8(q_values: np.ndarray, fixed_bits: int) -> np.ndarray:
    rounded = (q_values + (1 << (fixed_bits - 1))) >> fixed_bits
    return np.clip(rounded, 0, 255).astype(np.uint8)


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def _png_bytes(pixels: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG", compress_level=6, optimize=False)
    return buffer.getvalue()


def _load_npy(path: Path) -> np.ndarray:
    return np.load(path, allow_pickle=False)


def _mask_artifact_summary(meta: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in meta.items() if key != "_path"}


class Renderer:
    def __init__(self, cache_root: Path) -> None:
        self.cache = CacheStore(cache_root)
        self._executor_lock = threading.Lock()
        self._executors: dict[int, ThreadPoolExecutor] = {}

    def close(self) -> None:
        with self._executor_lock:
            executors = list(self._executors.values())
            self._executors.clear()
        for executor in executors:
            executor.shutdown(wait=True, cancel_futures=False)

    def _executor_for(self, max_workers: int) -> ThreadPoolExecutor:
        with self._executor_lock:
            executor = self._executors.get(max_workers)
            if executor is None:
                executor = ThreadPoolExecutor(
                    max_workers=max_workers,
                    thread_name_prefix=f"repair-block-{max_workers}",
                )
                self._executors[max_workers] = executor
            return executor

    def _cache_hit(self, stage: str, key: str) -> dict[str, Any] | None:
        return self.cache.load(stage, key)

    def render(
        self,
        sample: LoadedSample,
        params: RenderParams,
        command_index: int = 0,
        expected_invalidated_stages: Iterable[str] | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> RenderOutcome:
        config = sample.config
        reused_stages: list[str] = []
        invalidated_stages: list[str] = []
        stage_observed_workers: dict[str, int] = {}
        total_stages = 4

        mask_key = sha256_json(
            {
                "stage": "mask",
                "schema": 1,
                "source_content_sha256": sample.source.content_sha256,
                "mask": params.mask.as_dict(),
            }
        )
        mask_meta = self._cache_hit("mask", mask_key)
        if mask_meta is None:
            mask = build_mask(params.mask, config.width, config.height)
            mask_meta = self.cache.store(
                "mask",
                mask_key,
                {
                    "content_sha256": array_sha256(mask),
                    "mask_rect": params.mask.as_dict(),
                    "pixel_count": int(mask.sum()),
                    "source_content_sha256": sample.source.content_sha256,
                },
                {"mask.npy": _npy_bytes(mask)},
            )
            invalidated_stages.append("mask")
        else:
            reused_stages.append("mask")
        mask = _load_npy(Path(mask_meta["_path"]) / "mask.npy").astype(np.bool_, copy=False)
        if mask.shape != (config.height, config.width):
            raise StudioError(E_RENDER_FAILED, "cached mask shape does not match sample")
        self._progress(
            progress_callback,
            command_index,
            params,
            "mask",
            1,
            total_stages,
            0,
            config.height,
            "mask ready",
        )

        fill_key = sha256_json(
            {
                "stage": "fill",
                "schema": 1,
                "mask_key": mask_key,
                "source_content_sha256": sample.source.content_sha256,
                "radius": params.repair_radius,
                "iterations": params.repair_iterations,
                "fixed_bits": config.fixed_bits,
                "rounding": HALF_UP,
                "algorithm": FILL_VERSION,
            }
        )
        fill_meta = self._cache_hit("fill", fill_key)
        if fill_meta is None:
            filled_q, fill_meta, worker_count = self._execute_fill(
                sample,
                mask,
                params,
                fill_key,
                command_index,
                progress_callback,
                total_stages,
            )
            invalidated_stages.append("fill")
            stage_observed_workers["fill"] = worker_count
        else:
            filled_q = _load_npy(Path(fill_meta["_path"]) / "filled_q.npy")
            reused_stages.append("fill")
            stage_observed_workers["fill"] = 0
        expected_shape = (config.height, config.width, 3)
        if filled_q.shape != expected_shape:
            raise StudioError(E_RENDER_FAILED, "cached fill shape does not match sample")

        style_key = sha256_json(
            {
                "stage": "style",
                "schema": 1,
                "fill_key": fill_key,
                "style_content_sha256": sample.style.content_sha256,
                "style_stats": sample.style_stats.as_dict(),
                "quant_levels": params.quant_levels,
                "channel_weights_q": list(params.channel_weights_q),
                "lut_version": LUT_VERSION,
            }
        )
        style_meta = self._cache_hit("style", style_key)
        if style_meta is None:
            styled, style_meta, worker_count = self._execute_style(
                sample,
                filled_q,
                params,
                style_key,
                command_index,
                progress_callback,
                total_stages,
            )
            invalidated_stages.append("style")
            stage_observed_workers["style"] = worker_count
        else:
            styled = _load_npy(Path(style_meta["_path"]) / "styled.npy")
            reused_stages.append("style")
            stage_observed_workers["style"] = 0
        if styled.shape != expected_shape:
            raise StudioError(E_RENDER_FAILED, "cached style shape does not match sample")

        composite_key = sha256_json(
            {
                "stage": "composite",
                "schema": 1,
                "style_key": style_key,
                "mask_key": mask_key,
                "source_content_sha256": sample.source.content_sha256,
                "preserve_mask": params.preserve_mask,
            }
        )
        composite_meta = self._cache_hit("composite", composite_key)
        if composite_meta is None:
            final_pixels, final_png_bytes, composite_meta, worker_count = self._execute_composite(
                sample,
                styled,
                mask,
                params,
                composite_key,
                command_index,
                progress_callback,
                total_stages,
            )
            invalidated_stages.append("composite")
            stage_observed_workers["composite"] = worker_count
        else:
            artifact_path = Path(composite_meta["_path"])
            final_png_bytes = (artifact_path / "final.png").read_bytes()
            final_pixels = _load_npy(artifact_path / "final.npy")
            reused_stages.append("composite")
            stage_observed_workers["composite"] = 0
        if final_pixels.shape != expected_shape:
            raise StudioError(E_RENDER_FAILED, "cached composite shape does not match sample")

        summaries = {
            "mask": _mask_artifact_summary(mask_meta),
            "fill": _mask_artifact_summary(fill_meta),
            "style": _mask_artifact_summary(style_meta),
            "composite": _mask_artifact_summary(composite_meta),
        }
        self._progress(
            progress_callback,
            command_index,
            params,
            "complete",
            4,
            total_stages,
            config.height,
            config.height,
            "render complete",
        )

        return RenderOutcome(
            params=params,
            params_fingerprint=params.fingerprint(),
            final_pixels=final_pixels,
            final_png_bytes=final_png_bytes,
            final_png_sha256=array_sha256(final_pixels),
            stage_summaries=summaries,
            reused_stages=reused_stages,
            invalidated_stages=invalidated_stages,
            expected_invalidated_stages=list(expected_invalidated_stages or []),
            max_workers_configured=config.thread_workers,
            max_workers_observed=max(stage_observed_workers.values(), default=0),
            stage_observed_workers=stage_observed_workers,
        )

    def _progress(
        self,
        callback: ProgressCallback | None,
        command_index: int,
        params: RenderParams,
        stage: str,
        stage_index: int,
        total_stages: int,
        completed_rows: int,
        total_rows: int,
        message: str,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if callback is None:
            return
        event = {
            "command_index": command_index,
            "params_fingerprint": params.fingerprint(),
            "stage": stage,
            "stage_index": stage_index,
            "total_stages": total_stages,
            "completed_rows": int(completed_rows),
            "total_rows": int(total_rows),
            "progress": round(min(1.0, stage_index / total_stages), 6),
            "message": message,
        }
        if extra:
            event.update(extra)
        callback(event)

    def _run_tasks(
        self,
        tasks: list[StageTask],
        function: Callable[[StageTask], TaskResult],
        max_workers: int,
    ) -> tuple[list[TaskResult], int]:
        tracker = ConcurrencyTracker()
        failures: list[tuple[str, str]] = []

        def guarded(task: StageTask) -> TaskResult:
            tracker.enter()
            try:
                return function(task)
            finally:
                tracker.leave()

        results: list[TaskResult] = []
        executor = self._executor_for(max_workers)
        future_map = {executor.submit(guarded, task): task for task in tasks}
        for future in as_completed(future_map):
            task = future_map[future]
            try:
                results.append(future.result())
            except Exception as exc:
                failures.append((task.sort_key, str(exc)))
        if failures:
            failures.sort(key=lambda item: item[0])
            raise StudioError(
                E_RENDER_FAILED,
                "one or more block tasks failed",
                {"failures": [{"sort_key": key, "message": message} for key, message in failures]},
            )
        results.sort(key=lambda result: result.task.sort_key)
        return results, tracker.maximum

    def _execute_fill(
        self,
        sample: LoadedSample,
        mask: np.ndarray,
        params: RenderParams,
        cache_key: str,
        command_index: int,
        progress_callback: ProgressCallback | None,
        total_stages: int,
    ) -> tuple[np.ndarray, dict[str, Any], int]:
        config = sample.config
        height, width = mask.shape
        fixed_bits = config.fixed_bits
        working = sample.source.pixels.astype(np.int64) << fixed_bits
        known = ~mask
        tasks = make_tasks(Path(config.image_file).name, height, config.chunk_rows, "fill")
        offsets = _distance_offsets(params.repair_radius)
        iterations_used = 0
        total_filled = 0
        maximum_workers = 0

        for iteration in range(1, params.repair_iterations + 1):
            unknown = mask & ~known
            if not np.any(unknown):
                break
            boundary = unknown & _four_neighbor_known(known)
            if not np.any(boundary):
                break
            numerator, denominator = _accumulate_fill_layer(
                working,
                known,
                boundary,
                offsets,
            )
            filled_mask = boundary & (denominator > 0)
            if not np.any(filled_mask):
                break

            def fill_task(task: StageTask) -> TaskResult:
                values, count = _round_fill_block(numerator, denominator, task.start, task.end)
                return TaskResult(task=task, values=values, filled_count=count)

            results, observed = self._run_tasks(tasks, fill_task, config.thread_workers)
            maximum_workers = max(maximum_workers, observed)
            newly_filled = 0
            for result in results:
                task_boundary = boundary[result.task.start : result.task.end]
                working[result.task.start : result.task.end][task_boundary] = result.values[task_boundary]
                newly_filled += result.filled_count
            if newly_filled == 0:
                break
            known = known | filled_mask
            total_filled += int(filled_mask.sum())
            iterations_used = iteration
            self._progress(
                progress_callback,
                command_index,
                params,
                "fill",
                2,
                total_stages,
                0,
                height,
                f"fill layer {iteration}",
                {
                    "completed_fill_layers": iteration,
                    "repair_iterations": params.repair_iterations,
                },
            )

        unresolved = int((mask & ~known).sum())
        content_sha = array_sha256(working)
        meta = self.cache.store(
            "fill",
            cache_key,
            {
                "content_sha256": content_sha,
                "mask_key": sha256_json(
                    {
                        "stage": "mask",
                        "schema": 1,
                        "source_content_sha256": sample.source.content_sha256,
                        "mask": params.mask.as_dict(),
                    }
                ),
                "repair_radius": params.repair_radius,
                "repair_iterations": params.repair_iterations,
                "iterations_used": iterations_used,
                "filled_pixels": total_filled,
                "unresolved_pixels": unresolved,
                "fixed_bits": fixed_bits,
                "rounding": HALF_UP,
                "weight_rule": "round(2^20 / squared_euclidean_distance)",
                "weight_scale": FILL_WEIGHT_SCALE,
                "algorithm_version": FILL_VERSION,
                "chunk_rows": config.chunk_rows,
                "task_keys": [task.sort_key for task in tasks],
                "failures": [],
            },
            {"filled_q.npy": _npy_bytes(working)},
        )
        return working, meta, maximum_workers

    def _execute_style(
        self,
        sample: LoadedSample,
        filled_q: np.ndarray,
        params: RenderParams,
        cache_key: str,
        command_index: int,
        progress_callback: ProgressCallback | None,
        total_stages: int,
    ) -> tuple[np.ndarray, dict[str, Any], int]:
        config = sample.config
        height, width = filled_q.shape[:2]
        lut = build_style_lut(
            sample.style_stats,
            params.channel_weights_q,
            params.quant_levels,
            config.fixed_bits,
        )
        tasks = make_tasks(Path(config.image_file).name, height, config.chunk_rows, "style")
        styled = np.empty_like(filled_q, dtype=np.uint8)

        def style_task(task: StageTask) -> TaskResult:
            block_q = filled_q[task.start : task.end]
            block_u8 = q_array_to_u8(block_q, config.fixed_bits)
            indices = np.minimum(params.quant_levels - 1, (block_u8.astype(np.int64) * params.quant_levels) // 256)
            block_style = np.empty_like(block_u8)
            for channel in range(3):
                block_style[:, :, channel] = lut[channel][indices[:, :, channel]]
            return TaskResult(task=task, values=block_style)

        results, observed = self._run_tasks(tasks, style_task, config.thread_workers)
        for result in results:
            styled[result.task.start : result.task.end] = result.values
        self._progress(
            progress_callback,
            command_index,
            params,
            "style",
            3,
            total_stages,
            height,
            height,
            "style mapping complete",
        )
        content_sha = array_sha256(styled)
        lut_bytes = _npy_bytes(lut)
        meta = self.cache.store(
            "style",
            cache_key,
            {
                "content_sha256": content_sha,
                "filled_content_sha256": array_sha256(filled_q),
                "style_content_sha256": sample.style.content_sha256,
                "style_stats": sample.style_stats.as_dict(),
                "quant_levels": params.quant_levels,
                "channel_weights_q": list(params.channel_weights_q),
                "lut_sha256": array_sha256(lut),
                "lut_entries": lut.tolist(),
                "lut_version": LUT_VERSION,
                "chunk_rows": config.chunk_rows,
                "task_keys": [task.sort_key for task in tasks],
                "failures": [],
            },
            {"styled.npy": _npy_bytes(styled), "lut.npy": lut_bytes},
        )
        return styled, meta, observed

    def _execute_composite(
        self,
        sample: LoadedSample,
        styled: np.ndarray,
        mask: np.ndarray,
        params: RenderParams,
        cache_key: str,
        command_index: int,
        progress_callback: ProgressCallback | None,
        total_stages: int,
    ) -> tuple[np.ndarray, bytes, dict[str, Any], int]:
        config = sample.config
        height, width = mask.shape
        tasks = make_tasks(Path(config.image_file).name, height, config.chunk_rows, "composite")
        final_pixels = np.empty_like(styled, dtype=np.uint8)

        def composite_task(task: StageTask) -> TaskResult:
            style_block = styled[task.start : task.end]
            source_block = sample.source.pixels[task.start : task.end]
            mask_block = mask[task.start : task.end, :, None]
            if params.preserve_mask:
                block = np.where(mask_block, source_block, style_block).astype(np.uint8, copy=False)
                override_count = 0
            else:
                block = style_block.copy()
                override_count = int(mask_block[..., 0].sum())
            return TaskResult(task=task, values=block, override_count=override_count)

        results, observed = self._run_tasks(tasks, composite_task, config.thread_workers)
        override_count = 0
        for result in results:
            final_pixels[result.task.start : result.task.end] = result.values
            override_count += result.override_count
        self._progress(
            progress_callback,
            command_index,
            params,
            "composite",
            4,
            total_stages,
            height,
            height,
            "composite complete",
        )
        final_png_bytes = _png_bytes(final_pixels)
        content_sha = array_sha256(final_pixels)
        meta = self.cache.store(
            "composite",
            cache_key,
            {
                "content_sha256": content_sha,
                "style_content_sha256": array_sha256(styled),
                "source_content_sha256": sample.source.content_sha256,
                "preserve_mask": params.preserve_mask,
                "style_override_pixels": override_count,
                "final_png_sha256": array_sha256(final_pixels),
                "final_png_file_sha256": _sha256_local(final_png_bytes),
                "chunk_rows": config.chunk_rows,
                "task_keys": [task.sort_key for task in tasks],
                "failure_count": 0,
                "failures": [],
            },
            {"final.npy": _npy_bytes(final_pixels), "final.png": final_png_bytes},
        )
        return final_pixels, final_png_bytes, meta, observed


def _sha256_local(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _four_neighbor_known(known: np.ndarray) -> np.ndarray:
    neighbors = np.zeros_like(known, dtype=np.bool_)
    neighbors[1:, :] |= known[:-1, :]
    neighbors[:-1, :] |= known[1:, :]
    neighbors[:, 1:] |= known[:, :-1]
    neighbors[:, :-1] |= known[:, 1:]
    return neighbors


def _distance_offsets(radius: int) -> list[tuple[int, int, int, int]]:
    offsets: list[tuple[int, int, int, int]] = []
    radius_squared = radius * radius
    for delta_y in range(-radius, radius + 1):
        for delta_x in range(-radius, radius + 1):
            distance_squared = delta_y * delta_y + delta_x * delta_x
            if distance_squared == 0 or distance_squared > radius_squared:
                continue
            weight = max(1, FILL_WEIGHT_SCALE // distance_squared)
            offsets.append((delta_y, delta_x, distance_squared, weight))
    offsets.sort(key=lambda item: (item[2], item[0], item[1]))
    return offsets


def _accumulate_fill_layer(
    q_values: np.ndarray,
    known: np.ndarray,
    boundary: np.ndarray,
    offsets: list[tuple[int, int, int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    height, width = boundary.shape
    numerator = np.zeros((height, width, 3), dtype=np.int64)
    denominator = np.zeros((height, width), dtype=np.int64)

    for delta_y, delta_x, _distance_squared, weight in offsets:
        target_y_start = 0
        target_y_end = height
        source_y_start = delta_y
        source_y_end = height + delta_y
        if source_y_start < 0:
            target_y_start -= source_y_start
            source_y_start = 0
        if source_y_end > height:
            target_y_end -= source_y_end - height
            source_y_end = height

        target_x_start = 0
        target_x_end = width
        source_x_start = delta_x
        source_x_end = width + delta_x
        if source_x_start < 0:
            target_x_start -= source_x_start
            source_x_start = 0
        if source_x_end > width:
            target_x_end -= source_x_end - width
            source_x_end = width

        if target_y_start >= target_y_end or target_x_start >= target_x_end:
            continue
        target_y = slice(target_y_start, target_y_end)
        source_y = slice(source_y_start, source_y_end)
        target_x = slice(target_x_start, target_x_end)
        source_x = slice(source_x_start, source_x_end)
        valid = known[source_y, source_x] & boundary[target_y, target_x]
        if not np.any(valid):
            continue
        valid_float = valid[..., None]
        numerator[target_y, target_x, :] += q_values[source_y, source_x, :] * weight * valid_float
        denominator[target_y, target_x] += weight * valid
    return numerator, denominator


def _round_fill_block(
    numerator: np.ndarray,
    denominator: np.ndarray,
    start: int,
    end: int,
) -> tuple[np.ndarray, int]:
    block_numerator = numerator[start:end]
    block_denominator = denominator[start:end]

    width = block_denominator.shape[1]
    values = np.zeros((end - start, width, 3), dtype=np.int64)
    has_values = block_denominator > 0
    for channel in range(3):
        channel_numerator = block_numerator[:, :, channel]
        values[:, :, channel][has_values] = (
            channel_numerator[has_values] + block_denominator[has_values] // 2
        ) // block_denominator[has_values]
    return values, int(has_values.sum())


def build_style_lut(
    stats: StyleStats,
    channel_weights_q: tuple[int, int, int],
    quant_levels: int,
    fixed_bits: int,
) -> np.ndarray:
    lut = np.empty((3, quant_levels), dtype=np.uint8)
    for channel in range(3):
        weight_q = channel_weights_q[channel]
        mean_q = _rescale_q16(stats.mean_q[channel], fixed_bits)
        std_q = _rescale_q16(stats.std_q[channel], fixed_bits)
        for level in range(quant_levels):
            if quant_levels == 1:
                input_value = 0
            else:
                input_value = (level * 255 + (quant_levels - 1) // 2) // (quant_levels - 1)
            input_q = input_value << fixed_bits
            style_value_q = mean_q + ((input_value - 128) * std_q * 2) // 255
            blended_q = (
                (WEIGHT_SCALE - weight_q) * input_q
                + weight_q * style_value_q
                + WEIGHT_SCALE // 2
            ) // WEIGHT_SCALE
            rounded = (blended_q + (1 << (fixed_bits - 1))) >> fixed_bits
            lut[channel, level] = np.uint8(np.clip(rounded, 0, 255))
    return lut


def _rescale_q16(value_q16: int, fixed_bits: int) -> int:
    if fixed_bits < 16:
        return (value_q16 + (1 << (15 - fixed_bits))) >> (16 - fixed_bits)
    if fixed_bits == 16:
        return value_q16
    return value_q16 << (fixed_bits - 16)
