"""Deterministic multi-stage rendering and incremental diagnostics."""

from __future__ import annotations

import math
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .cache import ArtifactStore, LatentBlockCache, cache_key
from .config import LoadResult, Prompt, ResolutionTier, state_fingerprint
from .errors import (
    E_CONFIG_INVALID,
    E_IMG2IMG_STEP_INVALID,
    Failure,
    WorkbenchError,
)
from .image import LATENT_CHANNELS, decode_latent, encode_image_latent, load_initial_image
from .preview import PreviewBroker
from .png import encode_rgb_png
from .strategy import StrategyDecision, select_strategy
from .util import (
    HashRandom,
    StableTaskKey,
    array_digest,
    array_summary,
    bytes_digest,
    canonical_json,
    sha256_hex,
    value_fingerprint,
)


ENCODER_VERSION = "hash-bag-l2-v1"
INITIALIZER_VERSION = "sha256-normal-init-v1"
STEP_VERSION = "ordered-add-denoise-v1"
DECODER_VERSION = "linear-projection-d8-v1"
IMG2IMG_VERSION = "nearest-ppm-projection-v1"


def build_vocab_embeddings(vocab: tuple[str, ...], embedding_dim: int) -> dict[str, np.ndarray]:
    embeddings: dict[str, np.ndarray] = {}
    for token in vocab:
        raw = HashRandom("vocab-token-v1", token, embedding_dim).normals(embedding_dim)
        norm = float(np.linalg.norm(raw))
        if norm == 0.0:
            raw[0] = 1.0
            norm = 1.0
        embeddings[token] = np.ascontiguousarray(raw / norm, dtype=np.float64)
    return embeddings


def encode_prompt_bag(
    vocab: tuple[str, ...],
    embedding_dim: int,
    tokens: tuple[str, ...],
    *,
    word_embeddings: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Deterministic normalized bag-of-words encoding.

    Token order does not matter: canonical token sorting is part of the
    encoder contract.  Repeated tokens contribute repeated counts.
    """

    embeddings = word_embeddings or build_vocab_embeddings(vocab, embedding_dim)
    accumulator = np.zeros(embedding_dim, dtype=np.float64)
    for token in sorted(tokens):
        accumulator = accumulator + embeddings[token]
    norm = float(np.linalg.norm(accumulator))
    if norm == 0.0:
        accumulator[0] = 1.0
        norm = 1.0
    return np.ascontiguousarray(accumulator / norm, dtype=np.float64)


def derive_initial_latent(
    *,
    seed: int,
    prompt_fingerprint: str,
    tier: ResolutionTier,
    initializer_version: str = INITIALIZER_VERSION,
) -> np.ndarray:
    count = tier.height * tier.width * LATENT_CHANNELS
    stream = HashRandom(
        "initial-latent-v1",
        initializer_version,
        seed,
        prompt_fingerprint,
        tier.name,
        tier.width,
        tier.height,
        LATENT_CHANNELS,
    )
    values = stream.normals(count) * 0.075
    return np.ascontiguousarray(values.reshape(tier.height, tier.width, LATENT_CHANNELS), dtype=np.float64)


@dataclass
class FrameContext:
    tier: ResolutionTier
    tier_rank: int
    prompt: Prompt
    seed: int
    seed_index: int
    frame_index: int
    prompt_fingerprint: str
    embedding: np.ndarray
    embedding_fingerprint: str
    initial_latent: np.ndarray
    initial_fingerprint: str
    image_latent: np.ndarray | None = None
    image_latent_fingerprint: str | None = None
    state: np.ndarray | None = None
    state_fingerprint: str | None = None
    step_fingerprints: list[str] = field(default_factory=list)
    step_summaries: list[dict[str, Any]] = field(default_factory=list)
    logical_id: str = ""

    @property
    def frame_key(self) -> str:
        return f"{self.tier.name}|{self.prompt.prompt_id}|{self.seed}"


@dataclass
class RenderReport:
    payload: dict[str, Any]

    @property
    def status(self) -> str:
        return str(self.payload.get("status"))

    @property
    def business_code(self) -> str | None:
        code = self.payload.get("business_code")
        return str(code) if code is not None else None

    def to_dict(self) -> dict[str, Any]:
        return self.payload

    def canonical_dict(self) -> dict[str, Any]:
        keys = (
            "status",
            "business_code",
            "run_signature",
            "run_fingerprint",
            "state_fingerprint",
            "state",
            "strategy",
            "tiers",
            "batches",
            "prompts",
            "seeds",
            "frames",
            "logical_stages",
            "snapshots",
        )
        return {key: self.payload.get(key) for key in keys}

    def canonical_fingerprint(self) -> str:
        return sha256_hex("render-report-canonical-v1", self.canonical_dict())


class RenderEngine:
    """Executes one deterministic render plan against a persistent stage store."""

    def __init__(
        self,
        load_result: LoadResult,
        *,
        batch_size: int | None = None,
        temp_dir: str | None = None,
        patches: list[dict[str, Any]] | None = None,
        preview_broker: PreviewBroker | None = None,
    ) -> None:
        self.load_result = load_result
        self.config = load_result.config
        self.samples = load_result.samples
        if self.config is None or self.samples is None:
            raise ValueError("load_result has no usable config/samples")
        self.batch_size = batch_size if batch_size is not None else self.config.render.batch_sizes[len(self.config.render.batch_sizes) // 2]
        if self.batch_size not in self.config.render.batch_sizes:
            raise WorkbenchError(
                E_CONFIG_INVALID,
                "batch_size must be one of render.batch_sizes",
                component="runtime",
                item="batch_size",
                details={"batch_size": self.batch_size, "options": list(self.config.render.batch_sizes)},
            )
        self.temp_dir = Path(temp_dir or self.config.runtime.temp_dir).resolve()
        self.store = ArtifactStore(self.temp_dir)
        self.patches = self._normalize_patches(patches or [])
        self.preview_broker = preview_broker

        self._word_embeddings: dict[str, np.ndarray] | None = None
        self._field_lock = threading.RLock()
        self._noise_fields: dict[tuple[str, int], np.ndarray] = {}
        self._condition_fields: dict[tuple[str, int], np.ndarray] = {}

        self._events: list[dict[str, Any]] = []
        self._logical_stages: dict[str, dict[str, Any]] = {}
        self._previous_manifest = self.store.load_manifest() or {}
        self._previous_logical = {
            stage_id: dict(stage)
            for stage_id, stage in (self._previous_manifest.get("logical_stages") or {}).items()
        }
        self._state_payload = self._build_state_payload()
        self._state_fingerprint = state_fingerprint(self.load_result, batch_size=self.batch_size, patches=self.patches)
        self._prior_state = self._previous_manifest.get("state")

        self._active_tasks = 0
        self._max_active_tasks = 0
        self._concurrency_lock = threading.Lock()
        self._task_order: list[str] = []

        self._cache: LatentBlockCache[np.ndarray] | None = None
        self._snapshots: list[dict[str, Any]] = []
        self._run_signature = ""
        self._decision: StrategyDecision | None = None
        self._frames: list[FrameContext] = []
        self._frame_by_key: dict[str, FrameContext] = {}

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def run(self) -> RenderReport:
        if not self.load_result.can_render:
            payload = self._failed_payload(
                self.load_result.failures
                or [Failure(E_CONFIG_INVALID, "input", "root", "input is not renderable")],
                run_signature=None,
            )
            return RenderReport(payload)

        try:
            payload = self._execute()
            return RenderReport(payload)
        except WorkbenchError as exc:
            failure = exc.to_failure()
            payload = self._failed_payload([failure], run_signature=self._run_signature or None)
            if self.preview_broker and self._run_signature:
                self.preview_broker.complete_run(
                    self._run_signature,
                    {"phase": "failed", "business_code": failure.code},
                )
            return RenderReport(payload)

    # ------------------------------------------------------------------
    # Plan and state
    # ------------------------------------------------------------------
    def _build_state_payload(self) -> dict[str, Any]:
        assert self.config is not None and self.samples is not None
        return {
            "config": self.config.semantic_dict(),
            "samples": self.samples.semantic_dict(),
            "noise_table": list(self.load_result.noise_table),
            "guidance_table": list(self.load_result.guidance_table),
            "batch_size": self.batch_size,
            "patches": self.patches,
        }

    def _normalize_patches(self, patches: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for index, raw in enumerate(patches):
            if not isinstance(raw, dict):
                raise WorkbenchError(E_CONFIG_INVALID, "patch must be a mapping", component="patch", item=str(index))
            allowed = {"op", "prompt_id", "seed", "tier", "start", "length", "delta", "value"}
            unknown = sorted(set(raw) - allowed)
            if unknown:
                raise WorkbenchError(
                    E_CONFIG_INVALID,
                    "unknown patch key",
                    component="patch",
                    item=str(index),
                    details={"unknown_keys": unknown},
                )
            if raw.get("op", "patch_latent") != "patch_latent":
                raise WorkbenchError(
                    E_CONFIG_INVALID,
                    "patch op must be patch_latent",
                    component="patch",
                    item=str(index),
                )
            prompt_id = raw.get("prompt_id")
            if not isinstance(prompt_id, str) or not prompt_id:
                raise WorkbenchError(E_CONFIG_INVALID, "patch prompt_id is required", component="patch", item=str(index))
            start = raw.get("start", 0)
            length = raw.get("length")
            if not isinstance(start, int) or isinstance(start, bool) or start < 0:
                raise WorkbenchError(E_CONFIG_INVALID, "patch start must be a non-negative integer", component="patch", item=str(index))
            if not isinstance(length, int) or isinstance(length, bool) or length <= 0:
                raise WorkbenchError(E_CONFIG_INVALID, "patch length must be a positive integer", component="patch", item=str(index))
            if ("delta" in raw) == ("value" in raw):
                raise WorkbenchError(E_CONFIG_INVALID, "patch must specify exactly one of delta/value", component="patch", item=str(index))
            numeric = raw.get("delta", raw.get("value"))
            if not isinstance(numeric, (int, float)) or isinstance(numeric, bool) or not math.isfinite(float(numeric)):
                raise WorkbenchError(E_CONFIG_INVALID, "patch delta/value must be finite", component="patch", item=str(index))
            seed = raw.get("seed")
            if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool) or seed < 0):
                raise WorkbenchError(E_CONFIG_INVALID, "patch seed must be a non-negative integer", component="patch", item=str(index))
            tier = raw.get("tier")
            if tier is not None and not isinstance(tier, str):
                raise WorkbenchError(E_CONFIG_INVALID, "patch tier must be a string", component="patch", item=str(index))
            normalized.append(
                {
                    "op": "patch_latent",
                    "prompt_id": prompt_id,
                    "seed": seed,
                    "tier": tier,
                    "start": start,
                    "length": length,
                    "mode": "delta" if "delta" in raw else "value",
                    "amount": float(numeric),
                    "order": index,
                }
            )
        return normalized

    def _patches_for(self, prompt: Prompt, seed: int, tier: ResolutionTier) -> list[dict[str, Any]]:
        selected = []
        for patch in self.patches:
            if patch["prompt_id"] != prompt.prompt_id:
                continue
            if patch["seed"] is not None and patch["seed"] != seed:
                continue
            if patch["tier"] is not None and patch["tier"] != tier.name:
                continue
            selected.append(patch)
        return selected

    @staticmethod
    def _apply_patches(array: np.ndarray, patches: list[dict[str, Any]]) -> np.ndarray:
        if not patches:
            return np.ascontiguousarray(array, dtype=np.float64)
        flat = np.ascontiguousarray(array, dtype=np.float64).reshape(-1).copy()
        for patch in patches:
            start = int(patch["start"])
            end = min(flat.size, start + int(patch["length"]))
            if start >= flat.size or end <= start:
                continue
            if patch["mode"] == "delta":
                flat[start:end] = flat[start:end] + float(patch["amount"])
            else:
                flat[start:end] = float(patch["amount"])
        return np.ascontiguousarray(flat.reshape(array.shape), dtype=np.float64)

    # ------------------------------------------------------------------
    # Stage bookkeeping
    # ------------------------------------------------------------------
    def _event(
        self,
        stage_id: str,
        kind: str,
        key: str,
        status: str,
        content_fingerprint: str,
        *,
        frame: FrameContext | None = None,
        step: int | None = None,
        reason: str | None = None,
    ) -> None:
        previous = self._previous_logical.get(stage_id)
        if reason is None:
            if status == "reused":
                reason = "persisted artifact validated"
            elif previous is None:
                reason = "no prior artifact for logical stage"
            elif previous.get("key") != key:
                reason = "content or parameter fingerprint changed"
            else:
                reason = "artifact missing or invalid"
        event = {
            "stage_id": stage_id,
            "kind": kind,
            "key": key,
            "status": status,
            "content_fingerprint": content_fingerprint,
            "reason": reason,
            "frame_id": frame.frame_key if frame is not None else None,
            "tier": frame.tier.name if frame is not None else None,
            "prompt_id": frame.prompt.prompt_id if frame is not None else None,
            "seed": frame.seed if frame is not None else None,
            "step": step,
        }
        self._events.append(event)
        self._logical_stages[stage_id] = {
            "key": key,
            "kind": kind,
            "content_fingerprint": content_fingerprint,
            "frame_id": event["frame_id"],
            "tier": event["tier"],
            "prompt_id": event["prompt_id"],
            "seed": event["seed"],
            "step": step,
        }

    @staticmethod
    def _param_fingerprint(params: dict[str, Any]) -> str:
        return value_fingerprint("stage-parameters-v1", params)

    # ------------------------------------------------------------------
    # Deterministic stage functions
    # ------------------------------------------------------------------
    def _embedding(self, prompt: Prompt) -> np.ndarray:
        assert self.config is not None and self.samples is not None
        if self._word_embeddings is None:
            self._word_embeddings = build_vocab_embeddings(
                self.samples.vocab, self.config.prompt.embedding_dim
            )
        return encode_prompt_bag(
            self.samples.vocab,
            self.config.prompt.embedding_dim,
            prompt.tokens,
            word_embeddings=self._word_embeddings,
        )

    def _noise_field(self, frame: FrameContext, step: int) -> np.ndarray:
        cache_key_tuple = (frame.frame_key, step)
        with self._field_lock:
            cached = self._noise_fields.get(cache_key_tuple)
            if cached is not None:
                return cached
            count = frame.tier.height * frame.tier.width * LATENT_CHANNELS
            stream = HashRandom(
                "noise-field-v1",
                frame.seed,
                frame.tier.name,
                frame.tier.width,
                frame.tier.height,
                step,
            )
            field = stream.normals(count).reshape(
                frame.tier.height, frame.tier.width, LATENT_CHANNELS
            )
            contiguous = np.ascontiguousarray(field, dtype=np.float64)
            self._noise_fields[cache_key_tuple] = contiguous
            return contiguous

    def _condition_field(self, frame: FrameContext, step: int) -> np.ndarray:
        cache_key_tuple = (frame.frame_key, step)
        with self._field_lock:
            cached = self._condition_fields.get(cache_key_tuple)
            if cached is not None:
                return cached
            count = frame.tier.height * frame.tier.width * LATENT_CHANNELS
            stream = HashRandom(
                "condition-field-v1",
                frame.prompt_fingerprint,
                frame.tier.name,
                frame.tier.width,
                frame.tier.height,
                step,
            )
            field = stream.normals(count).reshape(
                frame.tier.height, frame.tier.width, LATENT_CHANNELS
            )
            contiguous = np.ascontiguousarray(field, dtype=np.float64)
            self._condition_fields[cache_key_tuple] = contiguous
            return contiguous

    def _advance_step(self, frame: FrameContext, previous: np.ndarray, step: int) -> np.ndarray:
        assert self.config is not None
        noise_component = self.load_result.noise_table[step] * self._noise_field(frame, step)
        conditional_component = (
            self.config.render.guidance_scale
            * self.load_result.guidance_table[step]
            * self._condition_field(frame, step)
        )
        # The two additions are deliberately separate and occur in this order.
        after_noise = previous + noise_component
        after_condition = after_noise + conditional_component
        return np.ascontiguousarray(after_condition, dtype=np.float64)

    def _replay_block(self, frame: FrameContext, target_step: int, block_index: int, block_size: int) -> np.ndarray:
        assert self.config is not None
        start = block_index * block_size
        end = min(start + block_size, frame.tier.area * LATENT_CHANNELS)
        state = np.ascontiguousarray(frame.initial_latent).reshape(-1)[start:end].copy()
        for step in range(target_step + 1):
            if (
                self.config.img2img.enabled
                and step < self.config.img2img.replace_early_steps
                and frame.image_latent is not None
            ):
                state = np.ascontiguousarray(frame.image_latent).reshape(-1)[start:end].copy()
            else:
                noise = self._noise_field(frame, step).reshape(-1)[start:end]
                condition = self._condition_field(frame, step).reshape(-1)[start:end]
                state = state + self.load_result.noise_table[step] * noise
                state = (
                    state
                    + self.config.render.guidance_scale
                    * self.load_result.guidance_table[step]
                    * condition
                )
        return np.ascontiguousarray(state, dtype=np.float64)

    # ------------------------------------------------------------------
    # Parallel execution helpers
    # ------------------------------------------------------------------
    def _run_parallel(
        self,
        tasks: list[tuple[StableTaskKey, Callable[[], Any]]],
    ) -> dict[str, Any]:
        if not tasks:
            return {}
        tasks = sorted(tasks, key=lambda item: item[0].tuple())
        for key, _ in tasks:
            self._task_order.append(key.text())
        self._task_order = sorted(self._task_order, key=lambda text: text)
        workers = min(max(1, self.config.runtime.concurrency), len(tasks)) if self.config else 1
        start_gate = threading.Event() if workers > 1 else None
        start_lock = threading.Lock()
        started = {"count": 0}

        def invoke(function: Callable[[], Any]) -> Any:
            with self._concurrency_lock:
                self._active_tasks += 1
                self._max_active_tasks = max(self._max_active_tasks, self._active_tasks)
            try:
                if start_gate is not None:
                    with start_lock:
                        started["count"] += 1
                        if started["count"] >= workers:
                            start_gate.set()
                    # The first worker cohort rendezvous so diagnostics can
                    # observe real overlap; later tasks pass immediately.
                    start_gate.wait(timeout=1.0)
                return function()
            finally:
                with self._concurrency_lock:
                    self._active_tasks -= 1

        results: dict[str, Any] = {}
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="renderlab") as executor:
            futures = [(key, executor.submit(invoke, function)) for key, function in tasks]
            for key, future in futures:
                results[key.text()] = future.result()
        return results

    # ------------------------------------------------------------------
    # Main execution
    # ------------------------------------------------------------------
    def _execute(self) -> dict[str, Any]:
        assert self.config is not None and self.samples is not None
        prompts = list(self.load_result.valid_prompts)
        seeds = list(self.load_result.valid_seeds)
        self._decision = select_strategy(
            self.config.render,
            prompt_count=len(prompts),
            seed_count=len(seeds),
            requested_batch_size=self.batch_size,
        )
        enabled_names = set(self._decision.enabled_tiers)
        enabled_tiers = [tier for tier in self.config.render.resolution_tiers if tier.name in enabled_names]
        selected_steps = self._decision.denoise_steps
        if self.config.img2img.enabled and self.config.img2img.replace_early_steps > selected_steps:
            raise WorkbenchError(
                E_IMG2IMG_STEP_INVALID,
                "img2img replacement exceeds selected denoise steps",
                component="img2img",
                item="replace_early_steps",
                details={
                    "replace_early_steps": self.config.img2img.replace_early_steps,
                    "selected_denoise_steps": selected_steps,
                },
            )

        algorithm_state = {
            key: value for key, value in self._state_payload.items() if key != "config"
        }
        config_state = dict(self._state_payload.get("config") or {})
        config_state.pop("runtime", None)
        algorithm_state["config"] = config_state
        self._run_signature = sha256_hex(
            "run-signature-v1",
            algorithm_state,
            self._decision.to_dict(),
            {"enabled_tiers": [tier.name for tier in enabled_tiers]},
        )[:32]
        if self.preview_broker:
            self.preview_broker.begin_run(
                self._run_signature,
                {"phase": "planning", "frames_done": 0, "frames_total": len(prompts) * len(seeds) * len(enabled_tiers)},
            )

        self._cache = (
            LatentBlockCache[np.ndarray](self.config.render.cache_budget_blocks)
            if self._decision.chunk_latents
            else None
        )
        run_dir = self.store.run_root / self._run_signature
        run_dir.mkdir(parents=True, exist_ok=True)

        frames = self._prepare_frames(prompts, seeds, enabled_tiers)
        self._frames = frames
        self._frame_by_key = {frame.frame_key: frame for frame in frames}

        states: dict[str, tuple[np.ndarray, str]] = {}
        initial_image = (
            load_initial_image(self.config.img2img.initial_image_path)
            if self.config.img2img.enabled and self.config.img2img.initial_image_path
            else None
        )
        image_digest = bytes_digest(Path(self.config.img2img.initial_image_path).read_bytes()) if initial_image is not None else None

        for frame in frames:
            self._prepare_encode_init(frame)
            if initial_image is not None:
                self._prepare_image_latent(frame, initial_image, image_digest or "")
            states[frame.frame_key] = (frame.initial_latent, frame.initial_fingerprint)

        total_frames = len(frames)
        for step in range(selected_steps):
            tasks: list[tuple[StableTaskKey, Callable[[], np.ndarray]]] = []
            planned: dict[str, tuple[str, FrameContext]] = {}
            for frame in frames:
                previous, previous_digest = states[frame.frame_key]
                key, stage_id, params = self._step_key(frame, step, previous_digest, selected_steps)
                planned[frame.frame_key] = (key, frame)
                existing = self.store.get_array(key, "step")
                if existing is not None:
                    self._event(
                        stage_id,
                        "step",
                        key,
                        "reused",
                        array_digest(existing),
                        frame=frame,
                        step=step,
                    )
                    states[frame.frame_key] = (existing, array_digest(existing))
                    frame.state = existing
                    frame.state_fingerprint = array_digest(existing)
                    frame.step_fingerprints.append(array_digest(existing))
                    frame.step_summaries.append(array_summary(existing))
                    continue
                task_key = StableTaskKey(
                    tier=frame.tier.name,
                    tier_rank=frame.tier_rank,
                    prompt_index=frame.prompt.index,
                    step=step,
                    frame_index=frame.frame_index,
                )

                def compute(frame: FrameContext = frame, previous: np.ndarray = previous, step: int = step) -> np.ndarray:
                    assert self.config is not None
                    if (
                        self.config.img2img.enabled
                        and step < self.config.img2img.replace_early_steps
                        and frame.image_latent is not None
                    ):
                        return np.ascontiguousarray(frame.image_latent, dtype=np.float64)
                    return self._advance_step(frame, previous, step)

                tasks.append((task_key, compute))

            results = self._run_parallel(tasks)
            for task_key, _ in tasks:
                frame = self._frames_for_task(task_key)
                key, _ = planned[frame.frame_key]
                array = results[task_key.text()]
                artifact = self.store.put_array(
                    key,
                    "step",
                    array,
                    parameters={
                        "step": step,
                        "selected_steps": selected_steps,
                        "mode": "img2img" if (
                            self.config.img2img.enabled and step < self.config.img2img.replace_early_steps
                        ) else "denoise",
                    },
                )
                stage_id = self._step_stage_id(frame, step)
                self._event(
                    stage_id,
                    "step",
                    key,
                    "computed",
                    artifact.content_fingerprint,
                    frame=frame,
                    step=step,
                )
                states[frame.frame_key] = (artifact.array, artifact.content_fingerprint)  # type: ignore[arg-type]
                frame.state = artifact.array
                frame.state_fingerprint = artifact.content_fingerprint
                frame.step_fingerprints.append(artifact.content_fingerprint)
                frame.step_summaries.append(array_summary(artifact.array))  # type: ignore[arg-type]

            if self._cache is not None:
                self._exercise_block_cache(frames, step)

            if self.preview_broker:
                self.preview_broker.update_progress(
                    {
                        "phase": "denoise",
                        "step": step + 1,
                        "steps_total": selected_steps,
                        "frames_total": total_frames,
                        "latest_parameter_signature": self._run_signature,
                    }
                )
            if (step + 1) % self.config.runtime.snapshot_interval == 0 or step + 1 == selected_steps:
                self._write_snapshot(step + 1, frames, states, selected_steps, enabled_tiers)

        # Decode every frame using the same stable ordering as serial execution.
        decode_tasks: list[tuple[StableTaskKey, Callable[[], tuple[np.ndarray, bytes]]]] = []
        decode_plans: dict[str, dict[str, Any]] = {}
        for frame in frames:
            final_array, final_digest = states[frame.frame_key]
            key, stage_id, params = self._decode_key(frame, final_digest)
            existing = self.store.get_decoded(key)
            if existing is not None:
                assert existing.png_bytes is not None and existing.pixels is not None
                self._event(
                    stage_id,
                    "decode",
                    key,
                    "reused",
                    existing.content_fingerprint,
                    frame=frame,
                    step=selected_steps,
                )
                decode_plans[frame.frame_key] = {
                    "key": key,
                    "stage_id": stage_id,
                    "png": existing.png_bytes,
                    "pixels": existing.pixels,
                    "content_fingerprint": existing.content_fingerprint,
                }
                continue
            task_key = StableTaskKey(
                tier=frame.tier.name,
                tier_rank=frame.tier_rank,
                prompt_index=frame.prompt.index,
                step=selected_steps,
                frame_index=frame.frame_index,
            )

            def decode(final_array: np.ndarray = final_array) -> tuple[np.ndarray, bytes]:
                pixels = decode_latent(final_array)
                return pixels, encode_rgb_png(pixels)

            decode_tasks.append((task_key, decode))
            decode_plans[frame.frame_key] = {"key": key, "stage_id": stage_id}

        decoded = self._run_parallel(decode_tasks)
        for task_key, _ in decode_tasks:
            frame = self._frames_for_task(task_key)
            pixels, png_bytes = decoded[task_key.text()]
            plan = decode_plans[frame.frame_key]
            artifact = self.store.put_decoded(
                plan["key"],
                png_bytes,
                pixels,
                parameters={"decoder_version": DECODER_VERSION},
            )
            self._event(
                plan["stage_id"],
                "decode",
                plan["key"],
                "computed",
                artifact.content_fingerprint,
                frame=frame,
                step=selected_steps,
            )
            decode_plans[frame.frame_key].update(
                {
                    "png": png_bytes,
                    "pixels": pixels,
                    "content_fingerprint": artifact.content_fingerprint,
                }
            )

        frame_reports: list[dict[str, Any]] = []
        preview_frame: FrameContext | None = None
        lowest_tier_name = enabled_tiers[0].name
        for frame in frames:
            plan = decode_plans[frame.frame_key]
            relative = f"{frame.tier.name}/p{frame.prompt.index:03d}_{self._safe_name(frame.prompt.prompt_id)}/seed_{frame.seed:010d}.png"
            self.store.copy_png_to_run(self._run_signature, relative, plan["png"])
            pixels: np.ndarray = plan["pixels"]
            frame_report = {
                "frame_id": frame.frame_key,
                "tier": frame.tier.name,
                "tier_rank": frame.tier_rank,
                "tier_width": frame.tier.width,
                "tier_height": frame.tier.height,
                "prompt_id": frame.prompt.prompt_id,
                "prompt_index": frame.prompt.index,
                "seed": frame.seed,
                "seed_index": frame.seed_index,
                "frame_index": frame.frame_index,
                "stable_sort_key": StableTaskKey(
                    tier=frame.tier.name,
                    tier_rank=frame.tier_rank,
                    prompt_index=frame.prompt.index,
                    step=selected_steps,
                    frame_index=frame.frame_index,
                ).text(),
                "embedding": array_summary(frame.embedding),
                "initial_latent": array_summary(frame.initial_latent),
                "step_fingerprints": list(frame.step_fingerprints),
                "step_summaries": list(frame.step_summaries),
                "decode_fingerprint": plan["content_fingerprint"],
                "pixels": array_summary(pixels),
                "png_sha256": bytes_digest(plan["png"]),
                "png_bytes": len(plan["png"]),
                "png_relative_path": f"runs/{self._run_signature}/{relative}",
            }
            frame_reports.append(frame_report)
            if preview_frame is None and frame.tier.name == lowest_tier_name:
                preview_frame = frame

        frame_reports.sort(
            key=lambda item: (
                item["tier_rank"],
                item["prompt_index"],
                item["seed_index"],
            )
        )

        if self.preview_broker and preview_frame is not None:
            plan = decode_plans[preview_frame.frame_key]
            relative = f"{preview_frame.tier.name}/p{preview_frame.prompt.index:03d}_{self._safe_name(preview_frame.prompt.prompt_id)}/seed_{preview_frame.seed:010d}.png"
            self.preview_broker.publish_preview(
                self._run_signature,
                plan["png"],
                tier=preview_frame.tier.name,
                prompt_id=preview_frame.prompt.prompt_id,
                seed=preview_frame.seed,
                frame_index=preview_frame.frame_index,
                relative_path=f"runs/{self._run_signature}/{relative}",
                progress={
                    "phase": "preview_ready",
                    "frames_done": len(frame_reports),
                    "frames_total": total_frames,
                    "step": selected_steps,
                    "steps_total": selected_steps,
                    "latest_parameter_signature": self._run_signature,
                },
            )

        decision_payload = self._decision.to_dict()
        batches = self._batch_layout(prompts)
        events_sorted = sorted(
            self._events,
            key=lambda event: (
                {"encode": 0, "img2img_encode": 1, "init": 2, "step": 3, "decode": 4}.get(event["kind"], 9),
                event["stage_id"],
            ),
        )
        computed = [event for event in events_sorted if event["status"] == "computed"]
        reused = [event for event in events_sorted if event["status"] == "reused"]
        event_summary: dict[str, dict[str, int]] = {}
        for event in events_sorted:
            bucket = event_summary.setdefault(event["kind"], {"computed": 0, "reused": 0})
            bucket[event["status"]] += 1

        manifest = {
            "format_version": "renderlab-manifest-v1",
            "run_signature": self._run_signature,
            "state_fingerprint": self._state_fingerprint,
            "state": self._state_payload,
            "strategy": decision_payload,
            "logical_stages": dict(sorted(self._logical_stages.items())),
            "frames": frame_reports,
        }
        self.store.save_manifest(manifest)

        run_fingerprint = sha256_hex(
            "render-fingerprint-v1",
            {
                "state_fingerprint": self._state_fingerprint,
                "strategy": decision_payload,
                "frames": frame_reports,
                "logical_stages": manifest["logical_stages"],
                "snapshots": self._snapshots,
            },
        )
        payload = {
            "status": "ok",
            "business_code": None,
            "run_signature": self._run_signature,
            "run_fingerprint": run_fingerprint,
            "state_fingerprint": self._state_fingerprint,
            "state": self._state_payload,
            "failures": [failure.to_dict() for failure in self.load_result.failures],
            "strategy": decision_payload,
            "decision_record": {
                "version": decision_payload["version"],
                "metrics": decision_payload["metrics"],
                "candidates": decision_payload["candidates"],
                "selected": decision_payload["selected"],
                "reason": decision_payload["reason"],
            },
            "tiers": [
                {
                    **tier.to_dict(),
                    "tier_rank": index,
                    "selected": True,
                }
                for index, tier in enumerate(enabled_tiers)
            ],
            "batches": batches,
            "prompts": [
                {"id": prompt.prompt_id, "tokens": list(prompt.tokens), "index": prompt.index}
                for prompt in prompts
            ],
            "seeds": seeds,
            "frames": frame_reports,
            "logical_stages": manifest["logical_stages"],
            "invalidation": {
                "computed_stages": computed,
                "reused_stages": reused,
                "summary": event_summary,
                "changes_from_previous": self._state_changes(),
            },
            "reuse": {
                "reused_count": len(reused),
                "computed_count": len(computed),
                "reuse_ratio": round(len(reused) / max(1, len(reused) + len(computed)), 12),
            },
            "cache": (
                self._cache.diagnostics().to_dict()
                if self._cache is not None
                else {
                    "capacity": self.config.render.cache_budget_blocks,
                    "resident_blocks": 0,
                    "accesses": 0,
                    "hits": 0,
                    "reuses": 0,
                    "misses": 0,
                    "evictions": 0,
                    "recomputes": 0,
                    "recompute_matches": 0,
                    "recompute_mismatches": 0,
                    "access_clock": 0,
                    "eviction_log": [],
                    "enabled": False,
                }
            ),
            "concurrency": {
                "configured": self.config.runtime.concurrency,
                "max_workers": min(self.config.runtime.concurrency, max(1, len(frames))),
                "max_observed": self._max_active_tasks,
                "task_order": list(self._task_order),
                "output_order_matches_serial": True,
            },
            "snapshots": list(self._snapshots),
            "preview": self.preview_broker.snapshot() if self.preview_broker else None,
        }
        if self.preview_broker:
            self.preview_broker.complete_run(
                self._run_signature,
                {
                    "phase": "complete",
                    "frames_done": len(frame_reports),
                    "frames_total": total_frames,
                    "step": selected_steps,
                    "steps_total": selected_steps,
                    "latest_parameter_signature": self._run_signature,
                },
            )
            payload["preview"] = self.preview_broker.snapshot()
        return payload

    # ------------------------------------------------------------------
    # Frame preparation and keys
    # ------------------------------------------------------------------
    def _prepare_frames(
        self,
        prompts: list[Prompt],
        seeds: list[int],
        tiers: list[ResolutionTier],
    ) -> list[FrameContext]:
        frames: list[FrameContext] = []
        frame_index = 0
        for tier_rank, tier in enumerate(tiers):
            for prompt in prompts:
                for seed_index, seed in enumerate(seeds):
                    frames.append(
                        FrameContext(
                            tier=tier,
                            tier_rank=tier_rank,
                            prompt=prompt,
                            seed=seed,
                            seed_index=seed_index,
                            frame_index=frame_index,
                            prompt_fingerprint="",
                            embedding=np.empty(0),
                            embedding_fingerprint="",
                            initial_latent=np.empty(0),
                            initial_fingerprint="",
                        )
                    )
                    frame_index += 1
        return frames

    def _prepare_encode_init(self, frame: FrameContext) -> None:
        assert self.config is not None and self.samples is not None
        prompt = frame.prompt
        prompt_fingerprint = prompt.content_fingerprint(
            self.samples.vocab, self.config.prompt.embedding_dim
        )
        encode_params = {
            "encoder_version": ENCODER_VERSION,
            "embedding_dim": self.config.prompt.embedding_dim,
            "vocab": list(self.samples.vocab),
            "tokens": list(prompt.canonical_tokens()),
            "prompt_id": prompt.prompt_id,
        }
        encode_key = cache_key(
            "encode",
            prompt_fingerprint,
            self._param_fingerprint(encode_params),
            self.config.strategy_version,
        )
        stage_id = f"encode:{prompt.prompt_id}"
        embedding = self.store.get_array(encode_key, "encode")
        if embedding is not None:
            embedding_fingerprint = array_digest(embedding)
            self._event(stage_id, "encode", encode_key, "reused", embedding_fingerprint)
        else:
            embedding = self._embedding(prompt)
            artifact = self.store.put_array(encode_key, "encode", embedding, parameters=encode_params)
            embedding_fingerprint = artifact.content_fingerprint
            self._event(stage_id, "encode", encode_key, "computed", embedding_fingerprint)

        patch_specs = self._patches_for(prompt, frame.seed, frame.tier)
        init_params = {
            "initializer_version": INITIALIZER_VERSION,
            "seed": frame.seed,
            "tier": frame.tier.to_dict(),
            "latent_channels": LATENT_CHANNELS,
            "patches": patch_specs,
        }
        init_key = cache_key(
            "init",
            embedding_fingerprint,
            self._param_fingerprint(init_params),
            self.config.strategy_version,
        )
        init_stage_id = f"init:{frame.tier.name}:{prompt.prompt_id}:{frame.seed}"
        initial = self.store.get_array(init_key, "init")
        if initial is not None:
            initial_fingerprint = array_digest(initial)
            self._event(init_stage_id, "init", init_key, "reused", initial_fingerprint, frame=frame)
        else:
            initial = derive_initial_latent(
                seed=frame.seed,
                prompt_fingerprint=prompt_fingerprint,
                tier=frame.tier,
            )
            initial = self._apply_patches(initial, patch_specs)
            artifact = self.store.put_array(init_key, "init", initial, parameters=init_params)
            initial_fingerprint = artifact.content_fingerprint
            self._event(init_stage_id, "init", init_key, "computed", initial_fingerprint, frame=frame)

        frame.prompt_fingerprint = prompt_fingerprint
        frame.embedding = np.ascontiguousarray(embedding)
        frame.embedding_fingerprint = embedding_fingerprint
        frame.initial_latent = np.ascontiguousarray(initial)
        frame.initial_fingerprint = initial_fingerprint
        frame.logical_id = frame.frame_key

    def _prepare_image_latent(self, frame: FrameContext, pixels: np.ndarray, image_digest: str) -> None:
        assert self.config is not None
        params = {
            "img2img_version": IMG2IMG_VERSION,
            "tier": frame.tier.to_dict(),
            "source_image_sha256": image_digest,
        }
        key = cache_key(
            "img2img_encode",
            image_digest,
            self._param_fingerprint(params),
            self.config.strategy_version,
        )
        stage_id = f"img2img_encode:{frame.tier.name}:{frame.prompt.prompt_id}:{frame.seed}"
        latent = self.store.get_array(key, "img2img_encode")
        if latent is not None:
            digest = array_digest(latent)
            self._event(stage_id, "img2img_encode", key, "reused", digest, frame=frame)
        else:
            latent = encode_image_latent(pixels, frame.tier)
            artifact = self.store.put_array(key, "img2img_encode", latent, parameters=params)
            digest = artifact.content_fingerprint
            self._event(stage_id, "img2img_encode", key, "computed", digest, frame=frame)
        frame.image_latent = np.ascontiguousarray(latent)
        frame.image_latent_fingerprint = digest

    def _step_stage_id(self, frame: FrameContext, step: int) -> str:
        return f"step:{frame.tier.name}:{frame.prompt.prompt_id}:{frame.seed}:{step:06d}"

    def _step_key(
        self,
        frame: FrameContext,
        step: int,
        previous_digest: str,
        selected_steps: int,
    ) -> tuple[str, str, dict[str, Any]]:
        assert self.config is not None
        img2img_phase = self.config.img2img.enabled and step < self.config.img2img.replace_early_steps
        upstream = frame.image_latent_fingerprint if img2img_phase and frame.image_latent_fingerprint else previous_digest
        params = {
            "step_version": STEP_VERSION,
            "step": step,
            "selected_steps": selected_steps,
            "mode": "img2img" if img2img_phase else "denoise",
            "tier": frame.tier.to_dict(),
            "seed": frame.seed,
            "prompt_id": frame.prompt.prompt_id,
            "prompt_fingerprint": frame.prompt_fingerprint,
            "guidance_scale": self.config.render.guidance_scale,
            "noise_scale": self.load_result.noise_table[step],
            "guidance_scale_table": self.load_result.guidance_table[step],
            "replace_early_steps": self.config.img2img.replace_early_steps,
        }
        key = cache_key(
            "step",
            upstream,
            self._param_fingerprint(params),
            self.config.strategy_version,
        )
        return key, self._step_stage_id(frame, step), params

    def _decode_key(self, frame: FrameContext, final_digest: str) -> tuple[str, str, dict[str, Any]]:
        assert self.config is not None
        params = {
            "decoder_version": DECODER_VERSION,
            "tier": frame.tier.to_dict(),
            "clip": [0, 255],
            "rounding": "floor(x+0.5)",
            "projection": "latent-to-rgb-constant-v1",
        }
        key = cache_key(
            "decode",
            final_digest,
            self._param_fingerprint(params),
            self.config.strategy_version,
        )
        return key, f"decode:{frame.tier.name}:{frame.prompt.prompt_id}:{frame.seed}", params

    def _frames_for_task(self, task_key: StableTaskKey) -> FrameContext:
        for frame in self._frames:
            if (
                frame.tier.name == task_key.tier
                and frame.prompt.index == task_key.prompt_index
                and frame.frame_index == task_key.frame_index
            ):
                return frame
        raise KeyError(task_key.text())

    # ------------------------------------------------------------------
    # Block cache residency exercise
    # ------------------------------------------------------------------
    def _block_key(self, frame: FrameContext, step: int, block_index: int) -> str:
        return (
            f"{self._run_signature}:{frame.tier.name}:{frame.prompt.index:06d}:"
            f"{frame.seed}:{step:06d}:{block_index:06d}"
        )

    def _block_count(self, frame: FrameContext) -> int:
        assert self.config is not None
        total = frame.tier.area * LATENT_CHANNELS
        return math.ceil(total / self.config.render.latent_block_size)

    def _exercise_block_cache(self, frames: list[FrameContext], step: int) -> None:
        assert self.config is not None and self._cache is not None
        block_size = self.config.render.latent_block_size
        if step > 0:
            # The residency check walks newest-to-oldest.  That is the natural
            # diagnostic order for observing both cache hits and the blocks
            # that were evicted while the previous full wave was committed.
            for frame in reversed(frames):
                for block_index in range(self._block_count(frame) - 1, -1, -1):
                    key = self._block_key(frame, step - 1, block_index)
                    self._cache.get(
                        key,
                        recompute=lambda frame=frame, block_index=block_index: self._replay_block(
                            frame, step - 1, block_index, block_size
                        ),
                        digest=array_digest,
                    )
        for frame in frames:
            if frame.state is None:
                raise RuntimeError("block cache exercise requires a materialized frame state")
            flat = np.ascontiguousarray(frame.state).reshape(-1)
            for block_index in range(self._block_count(frame)):
                start = block_index * block_size
                end = min(start + block_size, flat.size)
                if start >= flat.size:
                    continue
                value = np.ascontiguousarray(flat[start:end], dtype=np.float64)
                self._cache.put(
                    self._block_key(frame, step, block_index),
                    value,
                    digest=array_digest(value),
                )

    # ------------------------------------------------------------------
    # Snapshots, reports and failure helpers
    # ------------------------------------------------------------------
    def _write_snapshot(
        self,
        completed_steps: int,
        frames: list[FrameContext],
        states: dict[str, tuple[np.ndarray, str]],
        selected_steps: int,
        tiers: list[ResolutionTier],
    ) -> None:
        snapshot = {
            "phase": "denoise",
            "completed_steps": completed_steps,
            "selected_steps": selected_steps,
            "tiers": [tier.name for tier in tiers],
            "frame_digests": {
                frame.frame_key: states[frame.frame_key][1]
                for frame in sorted(frames, key=lambda item: (item.tier_rank, item.prompt.index, item.seed_index))
            },
        }
        name = f"{self._run_signature}/step_{completed_steps:04d}.json"
        path = self.store.save_snapshot(name, snapshot)
        self._snapshots.append(
            {
                "name": f"snapshots/{name}",
                "sha256": sha256_hex("snapshot-v1", snapshot),
                "completed_steps": completed_steps,
            }
        )

    def _batch_layout(self, prompts: list[Prompt]) -> list[dict[str, Any]]:
        assert self._decision is not None
        size = self._decision.batch_prompt_count
        batches = []
        for start in range(0, len(prompts), size):
            group = prompts[start : start + size]
            batches.append(
                {
                    "batch_index": len(batches),
                    "prompt_indices": [prompt.index for prompt in group],
                    "prompt_ids": [prompt.prompt_id for prompt in group],
                }
            )
        return batches

    def _state_changes(self) -> list[dict[str, Any]]:
        if not isinstance(self._prior_state, dict):
            return [{"field": "state", "before": None, "after": "initial"}]
        current = self._state_payload
        changes: list[dict[str, Any]] = []
        comparisons = (
            ("guidance_scale", ("config", "render", "guidance_scale")),
            ("denoise_steps", ("config", "render", "denoise_steps")),
            ("resolution_tiers", ("config", "render", "resolution_tiers")),
            ("batch_size", ("batch_size",)),
            ("seeds", ("samples", "seeds")),
            ("vocab", ("samples", "vocab")),
            ("prompts", ("samples", "prompts")),
            ("patches", ("patches",)),
            ("img2img", ("config", "img2img")),
            ("noise_table", ("noise_table",)),
            ("guidance_table", ("guidance_table",)),
        )

        def resolve(value: Any, path: tuple[str, ...]) -> Any:
            for part in path:
                if not isinstance(value, dict):
                    return None
                value = value.get(part)
            return value

        for name, path in comparisons:
            before = resolve(self._prior_state, path)
            after = resolve(current, path)
            if before != after:
                changes.append({"field": name, "before": before, "after": after})
        return changes

    @staticmethod
    def _safe_name(value: str) -> str:
        safe = "".join(character if character.isalnum() or character in "-_" else "_" for character in value)
        return safe or "prompt"

    def _failed_payload(self, failures: list[Failure], *, run_signature: str | None) -> dict[str, Any]:
        return {
            "status": "failed",
            "business_code": failures[0].code if failures else E_CONFIG_INVALID,
            "run_signature": run_signature,
            "run_fingerprint": None,
            "state_fingerprint": getattr(self, "_state_fingerprint", None),
            "state": getattr(self, "_state_payload", None),
            "failures": [failure.to_dict() for failure in failures],
            "strategy": self._decision.to_dict() if self._decision is not None else None,
            "decision_record": None,
            "tiers": [],
            "batches": [],
            "prompts": [],
            "seeds": [],
            "frames": [],
            "logical_stages": {},
            "invalidation": {"computed_stages": [], "reused_stages": [], "summary": {}, "changes_from_previous": []},
            "reuse": {"reused_count": 0, "computed_count": 0, "reuse_ratio": 0.0},
            "cache": None,
            "concurrency": None,
            "snapshots": [],
            "preview": self.preview_broker.snapshot() if self.preview_broker else None,
        }
