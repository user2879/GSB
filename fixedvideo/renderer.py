"""Deterministic five-stage video renderer with concurrent block decoding."""

from __future__ import annotations

import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .cache import (
    ArtifactCache,
    CompositionArtifact,
    DecodedFrameArtifact,
    LatentArtifact,
    PromptArtifact,
    make_composition_artifact,
    make_decoded_artifact,
    make_latent_artifact,
    make_prompt_artifact,
)
from .config import RenderConfig
from .state import CommandEffect, RenderState, effective_config_fingerprint
from .util import (
    DECODER_VERSION,
    DENOISE_VERSION,
    ENCODER_VERSION,
    PIPELINE_VERSION,
    ROUNDING_VERSION,
    array_sha256,
    encode_png,
    fixed_to_float,
    float_to_fixed,
    hash_object,
    json_safe_float,
    normalize_prompt,
    quantile_summary,
    round_half_away_array,
    round_half_away_scalar,
    sha256_bytes,
    stable_seed,
)


ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class RenderResult:
    state: RenderState
    effect: CommandEffect
    command_id: str
    config_fingerprint: str
    prompt_fp: str
    projection_fp: str
    trace_fingerprint: str
    frame_fps: tuple[str, ...]
    frame_keys: tuple[str, ...]
    composition_fp: str
    composition_key: str
    preview_png: bytes
    step_traces: tuple[dict[str, Any], ...]
    frame_traces: tuple[dict[str, Any], ...]
    audit: tuple[dict[str, Any], ...]
    stage_stats: dict[str, dict[str, int]]
    reused_count: int
    recomputed_count: int
    hit_rate: float
    progress: dict[str, Any]
    max_concurrency: int
    elapsed_ms: float
    seed: int
    steps: int

    def preview_summary(self) -> dict[str, Any]:
        progress = dict(self.progress)
        reuse_keys = [record["key"] for record in self.audit if record["status"] == "reuse"]
        recompute_keys = [record["key"] for record in self.audit if record["status"] == "recompute"]
        progress.update(
            {
                "status": "complete",
                "version": self.state.version,
                "command_cursor": self.state.command_cursor,
                "command_id": self.command_id,
                "config_fingerprint": self.config_fingerprint,
                "seed": self.seed,
                "steps": self.steps,
                "frame_count": self.state.frame_count,
                "chunk_rows": self.state.chunk_rows,
                "quantization_level": self.state.quantization_level,
                "frame_fingerprints": list(self.frame_fps),
                "composition_fingerprint": self.composition_fp,
                "preview_png_sha256": sha256_bytes(self.preview_png),
                "reused_count": self.reused_count,
                "recomputed_count": self.recomputed_count,
                "hit_rate": self.hit_rate,
                "reuse_keys": reuse_keys,
                "recompute_keys": recompute_keys,
            }
        )
        return progress


class ProgressTracker:
    def __init__(
        self,
        state: RenderState,
        config: RenderConfig,
        effect: CommandEffect,
        config_fingerprint: str,
        callback: ProgressCallback | None,
    ) -> None:
        self._lock = threading.Lock()
        self._callback = callback
        self._state = state
        self._config = config
        self._effect = effect
        self._config_fingerprint = config_fingerprint
        self._completed_steps: dict[str, set[int]] = defaultdict(set)
        self._completed_frames: set[int] = set()
        self._completed_blocks: set[tuple[int, int]] = set()
        self._block_rows: dict[tuple[int, int], int] = {}
        self._stage = "encoding"
        self._current_step: int | None = None
        self._current_frame: int | None = None
        self._completion_log: list[str] = []
        self._total_blocks = state.frame_count * self._blocks_per_frame(state.chunk_rows)
        self._total_rows = state.frame_count * config.height
        self._total_denoise_tasks = state.frame_count * state.steps

    def _blocks_per_frame(self, chunk_rows: int) -> int:
        return (self._config.height + chunk_rows - 1) // chunk_rows

    def _snapshot_locked(self) -> dict[str, Any]:
        completed_rows = sum(self._block_rows.values())
        denoise_done = sum(len(frames) for frames in self._completed_steps.values())
        total_work = max(1, self._total_blocks + self._total_denoise_tasks)
        work_done = len(self._completed_blocks) + denoise_done
        ratio = min(1.0, work_done / total_work)
        if self._stage == "complete":
            ratio = 1.0
        return {
            "stage": self._stage,
            "current_step": self._current_step,
            "current_frame": self._current_frame,
            "affected_frames": list(self._effect.affected_frames),
            "affected_frame_range": (
                [min(self._effect.affected_frames), max(self._effect.affected_frames)]
                if self._effect.affected_frames
                else []
            ),
            "affected_stages": list(self._effect.affected_stages),
            "completed_steps": {
                "text_initialization": sorted(self._completed_steps.get("initialization", set())),
                "image_encoding": sorted(self._completed_steps.get("image_encoding", set())),
                "denoise": sorted(self._completed_steps.get("denoise", set())),
            },
            "completed_frames": sorted(self._completed_frames),
            "completed_blocks": len(self._completed_blocks),
            "total_blocks": self._total_blocks,
            "completed_rows": completed_rows,
            "total_rows": self._total_rows,
            "progress": round(ratio, 12),
            "block_completion_order": list(self._completion_log),
            "version": self._state.version,
            "command_cursor": self._state.command_cursor,
            "config_fingerprint": self._config_fingerprint,
            "seed": self._state.seed,
            "steps": self._state.steps,
            "frame_count": self._state.frame_count,
            "chunk_rows": self._state.chunk_rows,
            "quantization_level": self._state.quantization_level,
        }

    def _emit_locked(self) -> None:
        if self._callback is not None:
            self._callback(self._snapshot_locked())

    def stage(self, stage: str, *, step: int | None = None, frame: int | None = None) -> None:
        with self._lock:
            self._stage = stage
            self._current_step = step
            self._current_frame = frame
            self._emit_locked()

    def mark_linear_stage(self, stage: str, index: int) -> None:
        with self._lock:
            self._completed_steps[stage].add(index)
            self._stage = stage
            self._emit_locked()

    def mark_denoise_task(self, step: int, frame: int) -> None:
        with self._lock:
            self._completed_steps["denoise"].add(frame * self._state.steps + step)
            self._stage = "denoise"
            self._current_step = step
            self._current_frame = frame
            self._emit_locked()

    def mark_block(self, frame: int, block_index: int, rows: int, sort_key: str) -> None:
        with self._lock:
            block_id = (frame, block_index)
            if block_id not in self._completed_blocks:
                self._completed_blocks.add(block_id)
                self._block_rows[block_id] = rows
                self._completion_log.append(sort_key)
            self._stage = "decode"
            self._current_frame = frame
            self._emit_locked()

    def mark_frame(self, frame: int) -> None:
        with self._lock:
            self._completed_frames.add(frame)
            self._stage = "decode"
            self._current_frame = frame
            self._emit_locked()

    def complete(self) -> None:
        with self._lock:
            self._stage = "complete"
            self._current_step = self._state.steps - 1
            self._current_frame = self._state.frame_count - 1
            self._completed_frames.update(range(self._state.frame_count))
            for frame in range(self._state.frame_count):
                for block in range(self._blocks_per_frame(self._state.chunk_rows)):
                    block_id = (frame, block)
                    self._completed_blocks.add(block_id)
                    if block_id not in self._block_rows:
                        row_start = block * self._state.chunk_rows
                        row_end = min(self._config.height, row_start + self._state.chunk_rows)
                        self._block_rows[block_id] = max(0, row_end - row_start)
            self._emit_locked()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._snapshot_locked()


class DeterministicRenderer:
    """Single-session renderer. Command execution is serialized by InteractiveEngine."""

    def __init__(self, config: RenderConfig, cache: ArtifactCache | None = None) -> None:
        self.config = config
        self.cache = cache if cache is not None else ArtifactCache()
        self.last_result: RenderResult | None = None
        self._audit: list[dict[str, Any]] = []
        self._stage_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"reuse": 0, "recompute": 0})
        self._progress: ProgressTracker | None = None
        self._active = 0
        self._max_active = 0
        self._active_lock = threading.Lock()

        model_fp = config.static_model_fingerprint()
        projection_seed = stable_seed("projection", config.projection_seed, model_fp)
        rng = np.random.Generator(np.random.PCG64(projection_seed))
        self.projection = rng.integers(-1, 2, size=(3, config.height, config.width, config.latent_size), dtype=np.int16)
        self.pixel_bias = rng.integers(-12, 13, size=(3, config.height, config.width), dtype=np.int16)
        self.projection_fp = hash_object(
            {
                "projection_sha256": array_sha256(self.projection),
                "pixel_bias_sha256": array_sha256(self.pixel_bias),
                "output_gain": config.output_gain,
                "decoder": DECODER_VERSION,
            },
            domain="fixedvideo-projection",
        )

        condition_seed = stable_seed("condition-matrix", config.condition_seed, model_fp)
        condition_rng = np.random.Generator(np.random.PCG64(condition_seed))
        self.condition_matrix = condition_rng.integers(
            -16, 17, size=(config.latent_size, config.embedding_dim), dtype=np.int16
        )
        self.condition_matrix_fp = hash_object(
            {
                "matrix_sha256": array_sha256(self.condition_matrix),
                "condition_seed": config.condition_seed,
            },
            domain="fixedvideo-condition-matrix",
        )

    # ------------------------------------------------------------------
    # Cache/audit helpers
    # ------------------------------------------------------------------
    def _run_task(self, function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        with self._active_lock:
            self._active += 1
            self._max_active = max(self._max_active, self._active)
        try:
            return function(*args, **kwargs)
        finally:
            with self._active_lock:
                self._active -= 1

    def _audit_record(
        self,
        *,
        stage: str,
        status: str,
        key: str,
        content_fp: str,
        params: dict[str, Any],
        deps: tuple[str, ...],
        frame: int | None = None,
        step: int | None = None,
        block_index: int | None = None,
        sort_key: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._stage_stats[stage][status] += 1
        record: dict[str, Any] = {
            "stage": stage,
            "status": status,
            "key": key,
            "content_fp": content_fp,
            "params_fp": hash_object(params, domain="fixedvideo-params"),
            "deps": list(deps),
        }
        if frame is not None:
            record["frame"] = frame
        if step is not None:
            record["step"] = step
        if block_index is not None:
            record["block_index"] = block_index
        if sort_key is not None:
            record["sort_key"] = sort_key
        if extra:
            record["extra"] = extra
        return record

    def _append_audit_batch(self, records: list[dict[str, Any]]) -> None:
        for record in records:
            record["sequence"] = len(self._audit) + 1
            self._audit.append(record)

    # ------------------------------------------------------------------
    # Stage 1: prompt encoding and deterministic branch initialization
    # ------------------------------------------------------------------
    def _encode_prompt(self, state: RenderState) -> tuple[PromptArtifact, str]:
        normalized, tokens = normalize_prompt(state.prompt)
        key = hash_object(
            {
                "stage": "encoding",
                "normalized_prompt": normalized,
                "vocabulary_hash": self.config.vocabulary_hash,
                "embedding_dim": self.config.embedding_dim,
                "encoder": ENCODER_VERSION,
            },
            domain="fixedvideo-cache-key",
        )
        found = self.cache.prompts.get(key)
        if found is not None:
            record = self._audit_record(
                stage="encoding",
                status="reuse",
                key=key,
                content_fp=found.content_fp,
                params=found.params,
                deps=found.deps,
                extra={"normalized_prompt": found.normalized_prompt},
            )
            self._append_audit_batch([record])
            if self._progress:
                self._progress.mark_linear_stage("encoding", 0)
            return found, "reuse"

        counts = Counter(tokens)
        total_known = sum(counts[token] for token in self.config.vocabulary if token in counts)
        embedding_fp = [0] * self.config.embedding_dim
        hits: list[dict[str, Any]] = []
        for vocab_index, token in enumerate(self.config.vocabulary):
            count = counts.get(token, 0)
            if not count:
                continue
            digest = sha256_bytes(f"fixedvideo-token|{token}".encode("utf-8"))
            embedding_index = int(digest[:16], 16) % self.config.embedding_dim
            sign = -1 if int(digest[16:18], 16) & 1 else 1
            value_fp = sign * int(round(1_000_000 * count / max(1, total_known)))
            embedding_fp[embedding_index] += value_fp
            positions = [index for index, item in enumerate(tokens) if item == token]
            hits.append(
                {
                    "token": token,
                    "vocabulary_index": vocab_index,
                    "embedding_index": embedding_index,
                    "count": count,
                    "signed_value_fp": value_fp,
                    "prompt_positions": positions,
                }
            )
        vocabulary_set = set(self.config.vocabulary)
        unknown_terms = tuple(dict.fromkeys(token for token in tokens if token not in vocabulary_set))
        params = {
            "encoder": ENCODER_VERSION,
            "vocabulary_hash": self.config.vocabulary_hash,
            "embedding_dim": self.config.embedding_dim,
            "normalization": "lowercase-ascii-alnum-v1",
            "embedding_scale": 1_000_000,
        }
        deps = (self.config.vocabulary_hash,)
        artifact = make_prompt_artifact(
            key=key,
            normalized_prompt=normalized,
            embedding_fp=tuple(embedding_fp),
            hits=tuple(hits),
            unknown_terms=unknown_terms,
            params=params,
            deps=deps,
        )
        self.cache.put(artifact)
        record = self._audit_record(
            stage="encoding",
            status="recompute",
            key=key,
            content_fp=artifact.content_fp,
            params=params,
            deps=deps,
            extra={"normalized_prompt": normalized, "unknown_terms": list(unknown_terms)},
        )
        self._append_audit_batch([record])
        if self._progress:
            self._progress.mark_linear_stage("encoding", 0)
        return artifact, "recompute"

    def _initialize_frame(
        self,
        state: RenderState,
        prompt: PromptArtifact,
        frame: int,
        latent_config_fp: str,
    ) -> tuple[LatentArtifact, str, dict[str, Any]]:
        key = hash_object(
            {
                "stage": "initialization",
                "prompt_fp": prompt.content_fp,
                "seed": state.seed,
                "latent_config_fp": latent_config_fp,
                "frame": frame,
                "generator": "pcg64-standard-normal-v1",
                "amplitude": 0.32,
            },
            domain="fixedvideo-cache-key",
        )
        found = self.cache.latents.get(key)
        if found is not None:
            record = self._audit_record(
                stage="initialization",
                status="reuse",
                key=key,
                content_fp=found.content_fp,
                params=found.params,
                deps=found.deps,
                frame=frame,
                sort_key=f"initialization:{frame:06d}",
            )
            return found, "reuse", record

        rng_seed = stable_seed(
            "initialization", prompt.content_fp, state.seed, latent_config_fp, frame
        )
        rng = np.random.Generator(np.random.PCG64(rng_seed))
        values = rng.standard_normal(self.config.latent_size).reshape(self.config.latent_shape) * 0.32
        latent = float_to_fixed(values, self.config.fixed_point_bits, state.quantization_step)
        params = {
            "stage": "initialization",
            "prompt_fp": prompt.content_fp,
            "seed": state.seed,
            "rng_seed": rng_seed,
            "latent_config_fp": latent_config_fp,
            "frame": frame,
            "generator": "pcg64-standard-normal-v1",
            "amplitude": 0.32,
            "fixed_point_bits": self.config.fixed_point_bits,
            "quantization_level": state.quantization_level,
            "quantization_step": state.quantization_step,
        }
        metadata = {
            "array_sha256": array_sha256(latent),
            "summary_fp": quantile_summary(latent),
            "summary": quantile_summary(fixed_to_float(latent, self.config.fixed_point_bits)),
            "frame": frame,
        }
        artifact = make_latent_artifact(
            key=key,
            stage="initialization",
            array=latent,
            params=params,
            deps=(prompt.content_fp, latent_config_fp),
            metadata=metadata,
        )
        self.cache.put(artifact)
        record = self._audit_record(
            stage="initialization",
            status="recompute",
            key=key,
            content_fp=artifact.content_fp,
            params=params,
            deps=artifact.deps,
            frame=frame,
            sort_key=f"initialization:{frame:06d}",
        )
        return artifact, "recompute", record

    def _encode_initial_image(self, state: RenderState) -> tuple[LatentArtifact | None, str | None, list[dict[str, Any]]]:
        if state.initial_image_path is None or state.initial_image_raw_fp is None:
            return None, None, []
        key = hash_object(
            {
                "stage": "image_encoding",
                "image_raw_fp": state.initial_image_raw_fp,
                "image_path": state.initial_image_path,
                "latent_shape": list(self.config.latent_shape),
                "fixed_point_bits": self.config.fixed_point_bits,
                "quantization_level": state.quantization_level,
                "quantization_step": state.quantization_step,
                "encoder": "float32-to-fixed-v1",
            },
            domain="fixedvideo-cache-key",
        )
        found = self.cache.latents.get(key)
        if found is not None:
            record = self._audit_record(
                stage="image_encoding",
                status="reuse",
                key=key,
                content_fp=found.content_fp,
                params=found.params,
                deps=found.deps,
                sort_key="image_encoding:000000",
            )
            return found, "reuse", [record]

        # The loader already validated descriptor, dtype, and shape. Re-open to avoid
        # retaining mutable command-owned objects in the cache key path.
        from .config import load_initial_latent

        initial = load_initial_latent(state.initial_image_path, expected_shape=self.config.latent_shape)
        fixed = float_to_fixed(initial.array, self.config.fixed_point_bits, state.quantization_step)
        params = {
            "stage": "image_encoding",
            "image_raw_fp": state.initial_image_raw_fp,
            "image_path": state.initial_image_path,
            "image_shape": list(initial.shape),
            "fixed_point_bits": self.config.fixed_point_bits,
            "quantization_level": state.quantization_level,
            "quantization_step": state.quantization_step,
            "encoder": "float32-to-fixed-v1",
        }
        metadata = {
            "array_sha256": array_sha256(fixed),
            "summary_fp": quantile_summary(fixed),
            "summary": quantile_summary(fixed_to_float(fixed, self.config.fixed_point_bits)),
            "source_raw_fp": state.initial_image_raw_fp,
        }
        artifact = make_latent_artifact(
            key=key,
            stage="image_encoding",
            array=fixed,
            params=params,
            deps=(state.initial_image_raw_fp, self.config.static_model_fingerprint()),
            metadata=metadata,
        )
        self.cache.put(artifact)
        record = self._audit_record(
            stage="image_encoding",
            status="recompute",
            key=key,
            content_fp=artifact.content_fp,
            params=params,
            deps=artifact.deps,
            sort_key="image_encoding:000000",
        )
        return artifact, "recompute", [record]

    # ------------------------------------------------------------------
    # Stage 2: fixed-point denoise and image-to-video replacement steps
    # ------------------------------------------------------------------
    def _condition_base(self, prompt: PromptArtifact, frame: int, state: RenderState) -> np.ndarray:
        embedding = np.asarray(prompt.embedding_fp, dtype=np.int64)
        dot = self.condition_matrix.astype(np.int64) @ embedding
        base = (dot * self.config.scale) // 16_000_000
        rng = np.random.Generator(
            np.random.PCG64(
                stable_seed(
                    "condition-frame",
                    prompt.content_fp,
                    self.config.condition_seed,
                    state.seed,
                    frame,
                )
            )
        )
        offsets = np.rint(rng.standard_normal(self.config.latent_size) * 0.012 * self.config.scale).astype(np.int64)
        return round_half_away_array(base + offsets, state.quantization_step).reshape(self.config.latent_shape)

    def _apply_patches(
        self,
        latent: np.ndarray,
        state: RenderState,
        step: int,
        frame: int,
    ) -> tuple[np.ndarray, list[dict[str, Any]], int]:
        patches = state.patch_for(step, frame)
        if not patches:
            return latent, [], 0
        output = latent.copy()
        records: list[dict[str, Any]] = []
        patch_abs_sum = 0
        for patch in patches:
            delta_fixed = int(
                float_to_fixed(
                    np.asarray([patch.delta], dtype=np.float64),
                    self.config.fixed_point_bits,
                    state.quantization_step,
                )[0]
            )
            output[
                :,
                patch.y : patch.y + patch.height,
                patch.x : patch.x + patch.width,
            ] += delta_fixed
            patch_abs_sum += abs(delta_fixed) * patch.width * patch.height * self.config.latent_shape[0]
            records.append({**patch.as_dict(), "delta_fixed": delta_fixed, "fingerprint": patch.fingerprint()})
        output = round_half_away_array(output, state.quantization_step)
        return output, records, patch_abs_sum

    def _compute_step(
        self,
        *,
        state: RenderState,
        prompt: PromptArtifact,
        frame: int,
        step: int,
        mode: str,
        previous: LatentArtifact | None,
        image_artifact: LatentArtifact | None,
        latent_config_fp: str,
    ) -> tuple[LatentArtifact, str, dict[str, Any]]:
        patch_records_preview = [patch.as_dict() for patch in state.patch_for(step, frame)]
        patch_fp = state.patch_fingerprint(step, frame)
        if mode == "image_replace":
            assert image_artifact is not None
            key = hash_object(
                {
                    "stage": "denoise",
                    "mode": "image_replace",
                    "image_fp": image_artifact.content_fp,
                    "step": step,
                    "frame": frame,
                    "fixed_point_bits": self.config.fixed_point_bits,
                    "quantization_level": state.quantization_level,
                    "quantization_step": state.quantization_step,
                    "patch_fp": patch_fp,
                    "formula": DENOISE_VERSION,
                },
                domain="fixedvideo-cache-key",
            )
            found = self.cache.latents.get(key)
            if found is not None:
                record = self._audit_record(
                    stage="denoise",
                    status="reuse",
                    key=key,
                    content_fp=found.content_fp,
                    params=found.params,
                    deps=found.deps,
                    frame=frame,
                    step=step,
                    sort_key=f"denoise:{step:06d}:{frame:06d}",
                    extra={"mode": mode, "patches": patch_records_preview},
                )
                return found, "reuse", record

            patched, patches, patch_abs_sum = self._apply_patches(
                image_artifact.array, state, step, frame
            )
            params = {
                "stage": "denoise",
                "mode": "image_replace",
                "image_fp": image_artifact.content_fp,
                "step": step,
                "frame": frame,
                "fixed_point_bits": self.config.fixed_point_bits,
                "quantization_level": state.quantization_level,
                "quantization_step": state.quantization_step,
                "patch_fp": patch_fp,
                "patches": patches,
                "formula": DENOISE_VERSION,
            }
            metadata = {
                "mode": mode,
                "step": step,
                "frame": frame,
                "array_sha256": array_sha256(patched),
                "latent_summary_fp": quantile_summary(patched),
                "latent_summary": quantile_summary(
                    fixed_to_float(patched, self.config.fixed_point_bits)
                ),
                "residual": {
                    "kind": "image_replacement",
                    "rounding_abs_sum_fp": 0,
                    "rounding_max_abs_fp": 0,
                    "patch_abs_sum_fp": patch_abs_sum,
                },
                "patches": patches,
            }
            artifact = make_latent_artifact(
                key=key,
                stage="denoise",
                array=patched,
                params=params,
                deps=(image_artifact.content_fp, patch_fp, self.config.static_model_fingerprint()),
                metadata=metadata,
            )
            self.cache.put(artifact)
            record = self._audit_record(
                stage="denoise",
                status="recompute",
                key=key,
                content_fp=artifact.content_fp,
                params=params,
                deps=artifact.deps,
                frame=frame,
                step=step,
                sort_key=f"denoise:{step:06d}:{frame:06d}",
                extra={"mode": mode, "patches": patch_records_preview},
            )
            return artifact, "recompute", record

        assert previous is not None
        noise_table_index = (step + (state.seed % self.config.max_noise_steps)) % self.config.max_noise_steps
        noise_entry = np.asarray(self.config.noise_table[noise_table_index, frame], dtype=np.int64)
        noise_entry_fp = hash_object(
            {
                "step": step,
                "noise_table_index": noise_table_index,
                "frame": frame,
                "seed": state.seed,
                "sha256": array_sha256(noise_entry),
            },
            domain="fixedvideo-noise-entry",
        )
        key = hash_object(
            {
                "stage": "denoise",
                "mode": mode,
                "upstream_fp": previous.content_fp,
                "prompt_fp": prompt.content_fp,
                "frame": frame,
                "step": step,
                "seed": state.seed,
                "guidance": state.guidance,
                "fixed_point_bits": self.config.fixed_point_bits,
                "quantization_level": state.quantization_level,
                "quantization_step": state.quantization_step,
                "noise_version": self.config.noise_table_version,
                "noise_table_index": noise_table_index,
                "noise_entry_fp": noise_entry_fp,
                "patch_fp": patch_fp,
                "formula": DENOISE_VERSION,
            },
            domain="fixedvideo-cache-key",
        )
        found = self.cache.latents.get(key)
        if found is not None:
            record = self._audit_record(
                stage="denoise",
                status="reuse",
                key=key,
                content_fp=found.content_fp,
                params=found.params,
                deps=found.deps,
                frame=frame,
                step=step,
                sort_key=f"denoise:{step:06d}:{frame:06d}",
                extra={"mode": mode, "patches": patch_records_preview},
            )
            return found, "reuse", record

        condition_base = self._condition_base(prompt, frame, state)
        guidance_milli = int(round_half_away_scalar(int(round(state.guidance * 1000)), 1))
        divisor = 1000 * (step + 2)
        condition_term = np.asarray(
            [
                round_half_away_scalar(int(value) * guidance_milli, divisor)
                for value in condition_base.reshape(-1)
            ],
            dtype=np.int64,
        ).reshape(self.config.latent_shape)
        before_quant = previous.array + noise_entry + condition_term
        quantized = round_half_away_array(before_quant, state.quantization_step)
        rounding_residual = before_quant - quantized
        patched, patches, patch_abs_sum = self._apply_patches(quantized, state, step, frame)
        params = {
            "stage": "denoise",
            "mode": mode,
            "upstream_fp": previous.content_fp,
            "prompt_fp": prompt.content_fp,
            "frame": frame,
            "step": step,
            "seed": state.seed,
            "guidance": state.guidance,
            "guidance_milli": guidance_milli,
            "condition_divisor": divisor,
            "condition_matrix_fp": self.condition_matrix_fp,
            "fixed_point_bits": self.config.fixed_point_bits,
            "scale": self.config.scale,
            "quantization_level": state.quantization_level,
            "quantization_step": state.quantization_step,
            "noise_version": self.config.noise_table_version,
            "noise_table_index": noise_table_index,
            "noise_entry_fp": noise_entry_fp,
            "patch_fp": patch_fp,
            "patches": patches,
            "formula": DENOISE_VERSION,
            "operation_order": "previous + noise + condition; quantize; patches; quantize",
        }
        metadata = {
            "mode": mode,
            "step": step,
            "frame": frame,
            "array_sha256": array_sha256(patched),
            "noise_table_index": noise_table_index,
            "latent_summary_fp": quantile_summary(patched),
            "latent_summary": quantile_summary(fixed_to_float(patched, self.config.fixed_point_bits)),
            "residual": {
                "kind": "fixed_point_rounding",
                "rounding_abs_sum_fp": int(np.abs(rounding_residual).sum()),
                "rounding_max_abs_fp": int(np.abs(rounding_residual).max(initial=0)),
                "rounding_summary_fp": quantile_summary(rounding_residual),
                "patch_abs_sum_fp": patch_abs_sum,
                "condition_abs_sum_fp": int(np.abs(condition_term).sum()),
                "noise_abs_sum_fp": int(np.abs(noise_entry).sum()),
            },
            "patches": patches,
        }
        artifact = make_latent_artifact(
            key=key,
            stage="denoise",
            array=patched,
            params=params,
            deps=(
                previous.content_fp,
                prompt.content_fp,
                noise_entry_fp,
                patch_fp,
                self.config.static_model_fingerprint(),
            ),
            metadata=metadata,
        )
        self.cache.put(artifact)
        record = self._audit_record(
            stage="denoise",
            status="recompute",
            key=key,
            content_fp=artifact.content_fp,
            params=params,
            deps=artifact.deps,
            frame=frame,
            step=step,
            sort_key=f"denoise:{step:06d}:{frame:06d}",
            extra={"mode": mode, "patches": patch_records_preview},
        )
        return artifact, "recompute", record

    # ------------------------------------------------------------------
    # Stage 3: row-block linear decode
    # ------------------------------------------------------------------
    def _decode_block(
        self,
        *,
        state: RenderState,
        frame: int,
        block_index: int,
        row_start: int,
        row_end: int,
        latent: np.ndarray,
        latent_fp: str,
        decode_key: str,
    ) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
        rows = row_end - row_start
        block_projection = self.projection[:, row_start:row_end, :, :].reshape(3, rows * self.config.width, self.config.latent_size)
        latent_flat = latent.reshape(-1)
        dots = np.tensordot(block_projection.astype(np.int64), latent_flat, axes=([2], [0]))
        bias = self.pixel_bias[:, row_start:row_end, :].reshape(3, rows * self.config.width).astype(np.int64)
        numerator = (
            dots * self.config.output_gain
            + bias * self.config.scale
            + 128 * self.config.scale
        )
        rounded = np.where(
            numerator >= 0,
            (numerator + self.config.scale // 2) // self.config.scale,
            -((-numerator + self.config.scale // 2) // self.config.scale),
        )
        clipped = np.clip(rounded, 0, 255).astype(np.uint8)
        pixels = clipped.reshape(3, rows, self.config.width).transpose(1, 2, 0)
        block_png = encode_png(pixels)
        sort_key = f"decode:{frame:06d}:block:{block_index:06d}"
        summary = {
            "block_index": block_index,
            "row_start": row_start,
            "row_end": row_end,
            "rows": rows,
            "sort_key": sort_key,
            "pixel_sha256": array_sha256(pixels),
            "png_sha256": sha256_bytes(block_png),
            "mean_rgb": [json_safe_float(float(value)) for value in pixels.reshape(-1, 3).mean(axis=0)],
        }
        params = {
            "decode_key": decode_key,
            "latent_fp": latent_fp,
            "projection_fp": self.projection_fp,
            "chunk_rows": state.chunk_rows,
            "quantization_level": state.quantization_level,
        }
        audit = self._audit_record(
            stage="decode_block",
            status="recompute",
            key=f"{decode_key}:block:{block_index:06d}",
            content_fp=summary["pixel_sha256"],
            params=params,
            deps=(latent_fp, self.projection_fp),
            frame=frame,
            block_index=block_index,
            sort_key=sort_key,
        )
        return pixels, summary, audit

    def _decode_frame(
        self,
        *,
        state: RenderState,
        frame: int,
        latent: LatentArtifact,
        executor: ThreadPoolExecutor,
    ) -> tuple[DecodedFrameArtifact, str, list[dict[str, Any]]]:
        block_count = (self.config.height + state.chunk_rows - 1) // state.chunk_rows
        decode_key = hash_object(
            {
                "stage": "decode",
                "latent_fp": latent.content_fp,
                "frame": frame,
                "projection_fp": self.projection_fp,
                "height": self.config.height,
                "width": self.config.width,
                "pixel_shape": [self.config.height, self.config.width, 3],
                "fixed_point_bits": self.config.fixed_point_bits,
                "quantization_level": state.quantization_level,
                "quantization_step": state.quantization_step,
                "chunk_rows": state.chunk_rows,
                "block_count": block_count,
                "decoder": DECODER_VERSION,
                "rounding": ROUNDING_VERSION,
                "output_gain": self.config.output_gain,
            },
            domain="fixedvideo-cache-key",
        )
        found = self.cache.decodes.get(decode_key)
        if found is not None:
            records: list[dict[str, Any]] = [
                self._audit_record(
                    stage="decode",
                    status="reuse",
                    key=decode_key,
                    content_fp=found.content_fp,
                    params=found.params,
                    deps=found.deps,
                    frame=frame,
                    sort_key=f"decode:{frame:06d}",
                )
            ]
            for block in found.blocks:
                records.append(
                    self._audit_record(
                        stage="decode_block",
                        status="reuse",
                        key=f"{decode_key}:block:{int(block['block_index']):06d}",
                        content_fp=str(block["pixel_sha256"]),
                        params={
                            "decode_key": decode_key,
                            "latent_fp": latent.content_fp,
                            "projection_fp": self.projection_fp,
                            "chunk_rows": state.chunk_rows,
                            "quantization_level": state.quantization_level,
                        },
                        deps=(latent.content_fp, self.projection_fp),
                        frame=frame,
                        block_index=int(block["block_index"]),
                        sort_key=str(block["sort_key"]),
                    )
                )
                if self._progress:
                    self._progress.mark_block(
                        frame,
                        int(block["block_index"]),
                        int(block["rows"]),
                        str(block["sort_key"]),
                    )
            if self._progress:
                self._progress.mark_frame(frame)
            return found, "reuse", records

        partition = []
        for block_index in range(block_count):
            row_start = block_index * state.chunk_rows
            row_end = min(self.config.height, row_start + state.chunk_rows)
            partition.append((block_index, row_start, row_end))

        pixels = np.empty((self.config.height, self.config.width, 3), dtype=np.uint8)
        summaries: list[dict[str, Any]] = []
        block_audits: list[dict[str, Any]] = []
        completion_order: list[str] = []
        wave_number = 0
        for wave_start in range(0, block_count, self.config.recompute_tile_budget):
            wave = partition[wave_start : wave_start + self.config.recompute_tile_budget]
            wave_number += 1
            futures = {
                executor.submit(
                    self._run_task,
                    self._decode_block,
                    state=state,
                    frame=frame,
                    block_index=block_index,
                    row_start=row_start,
                    row_end=row_end,
                    latent=latent.array,
                    latent_fp=latent.content_fp,
                    decode_key=decode_key,
                ): (block_index, row_start, row_end)
                for block_index, row_start, row_end in wave
            }
            for future in as_completed(futures):
                block_pixels, summary, audit = future.result()
                block_index, row_start, row_end = futures[future]
                summary["wave"] = wave_number
                summaries.append(summary)
                block_audits.append(audit)
                completion_order.append(summary["sort_key"])
                pixels[row_start:row_end, :, :] = block_pixels
                if self._progress:
                    self._progress.mark_block(
                        frame, block_index, row_end - row_start, summary["sort_key"]
                    )

        summaries.sort(key=lambda item: int(item["block_index"]))
        block_audits.sort(key=lambda item: int(item["block_index"]))
        png = encode_png(pixels)
        params = {
            "pixel_shape": [self.config.height, self.config.width, 3],
            "latent_fp": latent.content_fp,
            "frame": frame,
            "projection_fp": self.projection_fp,
            "height": self.config.height,
            "width": self.config.width,
            "fixed_point_bits": self.config.fixed_point_bits,
            "scale": self.config.scale,
            "quantization_level": state.quantization_level,
            "quantization_step": state.quantization_step,
            "chunk_rows": state.chunk_rows,
            "block_partition": [
                {"block_index": index, "row_start": start, "row_end": end}
                for index, start, end in partition
            ],
            "canonical_block_completion_order": [
                item["sort_key"] for item in summaries
            ],
            "decoder": DECODER_VERSION,
            "rounding": ROUNDING_VERSION,
            "output_gain": self.config.output_gain,
        }
        del completion_order  # thread timing is observable in progress, not in artifact identity
        artifact = make_decoded_artifact(
            key=decode_key,
            pixels=pixels,
            png=png,
            blocks=tuple(summaries),
            params=params,
            deps=(latent.content_fp, self.projection_fp, self.condition_matrix_fp),
        )
        self.cache.put(artifact)
        records = [
            self._audit_record(
                stage="decode",
                status="recompute",
                key=decode_key,
                content_fp=artifact.content_fp,
                params=params,
                deps=artifact.deps,
                frame=frame,
                sort_key=f"decode:{frame:06d}",
            )
        ]
        records.extend(block_audits)
        if self._progress:
            self._progress.mark_frame(frame)
        return artifact, "recompute", records

    # ------------------------------------------------------------------
    # Stage 4: composition and high-level orchestration
    # ------------------------------------------------------------------
    def _compose(
        self,
        state: RenderState,
        decodes: list[DecodedFrameArtifact],
    ) -> tuple[CompositionArtifact, str]:
        frame_fps = tuple(frame.content_fp for frame in decodes)
        frame_keys = tuple(frame.key for frame in decodes)
        key = hash_object(
            {
                "stage": "composition",
                "frame_fps": list(frame_fps),
                "frame_keys": list(frame_keys),
                "frame_count": state.frame_count,
                "chunk_rows": state.chunk_rows,
                "output_layout": "horizontal-strip-v1",
            },
            domain="fixedvideo-cache-key",
        )
        found = self.cache.compositions.get(key)
        if found is not None:
            record = self._audit_record(
                stage="composition",
                status="reuse",
                key=key,
                content_fp=found.content_fp,
                params=found.params,
                deps=found.deps,
                sort_key="composition:000000",
            )
            self._append_audit_batch([record])
            return found, "reuse"

        separator = np.full((self.config.height, 1, 3), 32, dtype=np.uint8)
        pieces: list[np.ndarray] = []
        for index, frame in enumerate(decodes):
            if index:
                pieces.append(separator)
            pieces.append(frame.pixels)
        strip = np.concatenate(pieces, axis=1)
        preview_png = encode_png(strip)
        params = {
            "output_layout": "horizontal-strip-v1",
            "frame_count": state.frame_count,
            "chunk_rows": state.chunk_rows,
            "frame_resolution": [self.config.height, self.config.width],
            "strip_resolution": [int(strip.shape[0]), int(strip.shape[1])],
            "frame_order": list(range(state.frame_count)),
            "separator_rgb": [32, 32, 32],
        }
        artifact = make_composition_artifact(
            key=key,
            frame_fps=frame_fps,
            frame_keys=frame_keys,
            preview_png=preview_png,
            params=params,
            deps=frame_fps,
        )
        self.cache.put(artifact)
        record = self._audit_record(
            stage="composition",
            status="recompute",
            key=key,
            content_fp=artifact.content_fp,
            params=params,
            deps=frame_fps,
            sort_key="composition:000000",
        )
        self._append_audit_batch([record])
        return artifact, "recompute"

    def render(
        self,
        state: RenderState,
        effect: CommandEffect,
        *,
        command_id: str,
        progress_callback: ProgressCallback | None = None,
    ) -> RenderResult:
        started = time.perf_counter()
        self._audit = []
        self._stage_stats = defaultdict(lambda: {"reuse": 0, "recompute": 0})
        self._max_active = 0
        config_fp = effective_config_fingerprint(self.config, state)
        self._progress = ProgressTracker(state, self.config, effect, config_fp, progress_callback)
        self._progress.stage("encoding")

        prompt, _ = self._encode_prompt(state)
        latent_config_fp = self.config.latent_parameter_fingerprint(state.quantization_level)
        image_artifact, _, image_audits = self._encode_initial_image(state)
        if image_audits:
            self._append_audit_batch(image_audits)
            if self._progress:
                self._progress.mark_linear_stage("image_encoding", 0)

        init_artifacts: dict[int, LatentArtifact] = {}
        init_audits: list[dict[str, Any]] = []
        for frame in range(state.frame_count):
            artifact, _, record = self._initialize_frame(state, prompt, frame, latent_config_fp)
            init_artifacts[frame] = artifact
            init_audits.append(record)
        if init_audits:
            init_audits.sort(key=lambda item: int(item["frame"]))
            self._append_audit_batch(init_audits)
            for frame in range(state.frame_count):
                if self._progress:
                    self._progress.mark_linear_stage("initialization", frame)

        use_image_path = image_artifact is not None and self.config.initial_skip_steps > 0
        previous_by_frame: dict[int, LatentArtifact] = {}
        step_records: list[dict[str, Any]] = []
        frame_step_artifacts: dict[int, list[LatentArtifact]] = {
            frame: [] for frame in range(state.frame_count)
        }
        image_fp = image_artifact.content_fp if image_artifact is not None else None

        if self._progress:
            self._progress.stage("denoise", step=0, frame=0)
        executor = ThreadPoolExecutor(max_workers=self.config.worker_count, thread_name_prefix="fixedvideo")
        try:
            for step in range(state.steps):
                replacement = use_image_path and step < self.config.initial_skip_steps
                mode = "image_replace" if replacement else ("image_after_skip" if use_image_path else "text")
                tasks = []
                for frame in range(state.frame_count):
                    if replacement:
                        previous = None
                    elif step == 0:
                        previous = init_artifacts[frame]
                    else:
                        previous = previous_by_frame[frame]
                    tasks.append(
                        executor.submit(
                            self._run_task,
                            self._compute_step,
                            state=state,
                            prompt=prompt,
                            frame=frame,
                            step=step,
                            mode=mode,
                            previous=previous,
                            image_artifact=image_artifact,
                            latent_config_fp=latent_config_fp,
                        )
                    )
                step_results = [future.result() for future in tasks]
                audits: list[dict[str, Any]] = []
                for frame, (artifact, _, record) in enumerate(step_results):
                    previous_by_frame[frame] = artifact
                    frame_step_artifacts[frame].append(artifact)
                    audits.append(record)
                    if self._progress:
                        self._progress.mark_denoise_task(step, frame)
                self._append_audit_batch(audits)
                step_records.append(
                    {
                        "step": step,
                        "mode": mode,
                        "replacement": replacement,
                        "frame_fps": [item[0].content_fp for item in step_results],
                    }
                )

            decodes: list[DecodedFrameArtifact] = []
            decode_audits: list[dict[str, Any]] = []
            for frame in range(state.frame_count):
                if self._progress:
                    self._progress.stage("decode", step=state.steps - 1, frame=frame)
                final_latent = previous_by_frame[frame]
                decoded, _, records = self._decode_frame(
                    state=state,
                    frame=frame,
                    latent=final_latent,
                    executor=executor,
                )
                decodes.append(decoded)
                decode_audits.extend(records)
            self._append_audit_batch(decode_audits)

            if self._progress:
                self._progress.stage("composition")
            composition, _ = self._compose(state, decodes)

            frame_traces: list[dict[str, Any]] = []
            all_step_traces: list[dict[str, Any]] = []
            for frame in range(state.frame_count):
                trace: dict[str, Any] = {
                    "frame": frame,
                    "initialization": (
                        {
                            "mode": "image",
                            "image_fp": image_fp,
                            "image_encoding_fp": image_artifact.content_fp if image_artifact else None,
                        }
                        if use_image_path
                        else {
                            "mode": "text",
                            "initialization_fp": init_artifacts[frame].content_fp,
                        }
                    ),
                    "steps": [],
                    "final_latent_fp": previous_by_frame[frame].content_fp,
                    "decode_fp": decodes[frame].content_fp,
                    "pixel_sha256": array_sha256(decodes[frame].pixels),
                    "blocks": [
                        {
                            "block_index": int(block["block_index"]),
                            "row_start": int(block["row_start"]),
                            "row_end": int(block["row_end"]),
                            "pixel_sha256": block["pixel_sha256"],
                        }
                        for block in decodes[frame].blocks
                    ],
                }
                for artifact in frame_step_artifacts[frame]:
                    step_trace = {
                        "frame": frame,
                        "step": int(artifact.metadata["step"]),
                        "mode": artifact.metadata["mode"],
                        "content_fp": artifact.content_fp,
                        "array_sha256": artifact.metadata["array_sha256"],
                        "latent_summary_fp": artifact.metadata["latent_summary_fp"],
                        "residual": artifact.metadata["residual"],
                        "patches": [patch["patch_id"] for patch in artifact.metadata.get("patches", [])],
                    }
                    trace["steps"].append(step_trace)
                    all_step_traces.append(step_trace)
                trace["steps"].sort(key=lambda item: int(item["step"]))
                frame_traces.append(trace)

            trace_canonical = {
                "config_fingerprint": config_fp,
                "prompt_fp": prompt.content_fp,
                "projection_fp": self.projection_fp,
                "frames": frame_traces,
                "composition_fp": composition.content_fp,
            }
            trace_fp = hash_object(trace_canonical, domain="fixedvideo-render-trace")
            self._progress.complete()
            progress = self._progress.snapshot()

            reuse_count = sum(stats["reuse"] for stats in self._stage_stats.values())
            recompute_count = sum(stats["recompute"] for stats in self._stage_stats.values())
            total = reuse_count + recompute_count
            hit_rate = reuse_count / total if total else 0.0
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            result = RenderResult(
                state=state,
                effect=effect,
                command_id=command_id,
                config_fingerprint=config_fp,
                prompt_fp=prompt.content_fp,
                projection_fp=self.projection_fp,
                trace_fingerprint=trace_fp,
                frame_fps=tuple(frame.content_fp for frame in decodes),
                frame_keys=tuple(frame.key for frame in decodes),
                composition_fp=composition.content_fp,
                composition_key=composition.key,
                preview_png=composition.preview_png,
                step_traces=tuple(sorted(all_step_traces, key=lambda item: (int(item["frame"]), int(item["step"])))),
                frame_traces=tuple(frame_traces),
                audit=tuple(self._audit),
                stage_stats={stage: dict(stats) for stage, stats in sorted(self._stage_stats.items())},
                reused_count=reuse_count,
                recomputed_count=recompute_count,
                hit_rate=hit_rate,
                progress=progress,
                max_concurrency=self._max_active,
                elapsed_ms=elapsed_ms,
                seed=state.seed,
                steps=state.steps,
            )
            self.last_result = result
            return result
        finally:
            executor.shutdown(wait=True, cancel_futures=False)

    def determinism_metadata(self) -> dict[str, Any]:
        return {
            "pipeline_version": PIPELINE_VERSION,
            "projection_fingerprint": self.projection_fp,
            "condition_matrix_fingerprint": self.condition_matrix_fp,
            "projection_shape": list(self.projection.shape),
            "pixel_bias_shape": list(self.pixel_bias.shape),
            "condition_matrix_shape": list(self.condition_matrix.shape),
            "worker_count": self.config.worker_count,
            "single_thread_equivalent": True,
        }
