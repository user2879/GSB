"""Incremental deterministic render orchestration and budget scheduling."""

from __future__ import annotations

import base64
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator

from .canonical import b64_decode, b64_encode, content_digest, make_key, sha256_bytes
from .conditions import FrameInput, ResolvedConditions, encode_image, encode_motion, encode_text
from .config import RenderConfig
from .errors import E_BUDGET_PENDING, E_SNAPSHOT_CORRUPT, fail
from .ledger import Ledger
from .pipeline import (
    RECIPE_VERSION,
    chunk_partition,
    decode_rows,
    denoise_step,
    fuse_conditions,
    initial_latent,
    projection_recipe,
    stage_records_for_step,
)
from .png import encode_png_rgb
from .snapshot import SnapshotStore
from .workspace import Workspace


class ConcurrencyProbe:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.current = 0
        self.maximum = 0
        self.executed = 0

    @contextmanager
    def slot(self) -> Iterator[None]:
        with self._lock:
            self.current += 1
            self.executed += 1
            self.maximum = max(self.maximum, self.current)
        try:
            yield
        finally:
            with self._lock:
                self.current -= 1


@dataclass
class TaskResult:
    logical_id: str
    artifact: dict[str, Any]
    hit: bool
    key: str
    deps: list[dict[str, str]]
    params: dict[str, Any]


class Renderer:
    def __init__(
        self,
        workspace: Workspace,
        snapshot_dir: str,
        *,
        budget_override: int | None = None,
    ) -> None:
        self.workspace = workspace
        self.config = workspace.config
        self.conditions = workspace.conditions
        self.store = SnapshotStore(snapshot_dir)
        committed = self.store.load_committed()
        self.prior_state: dict[str, Any] = committed.state if committed else {}
        self.prior_manifest = committed.manifest if committed else None
        self.artifacts: dict[str, dict[str, Any]] = dict(self.prior_state.get("artifacts", {}))
        self.prior_artifact_keys = set(self.artifacts)
        self.ledger = Ledger(self.prior_state.get("descriptors", {}))
        self.batch_budget = (
            budget_override if budget_override is not None else self.config.max_denoise_points
        )
        if self.batch_budget is not None and self.batch_budget < 0:
            fail("E_CONFIG_INVALID", "budget must be non-negative", budget=self.batch_budget)
        self.batch_spent = 0
        self.budget_events: list[dict[str, Any]] = []
        self.probe = ConcurrencyProbe()
        try:
            self.task_delay_seconds = max(0.0, float(os.environ.get("MMFM_TASK_DELAY_MS", "0"))) / 1000.0
        except ValueError:
            self.task_delay_seconds = 0.0
        self.run_id = content_digest(
            {
                "recipe_version": RECIPE_VERSION,
                "config": self.config.raw,
                "input_hash": self.workspace.input_hash,
            }
        )
        self.completed_steps: dict[int, int] = {frame.frame: 0 for frame in self.conditions.frames}
        self.step_artifacts: dict[int, dict[int, dict[str, Any]]] = {
            frame.frame: {} for frame in self.conditions.frames
        }
        self.latents: dict[int, dict[str, Any]] = {}
        self.fusions: dict[int, dict[str, Any]] = {}
        self.condition_artifacts: dict[int, dict[str, dict[str, Any]]] = {}
        self.cursors: dict[str, Any] = {
            "denoise_next_step": {str(frame.frame): 1 for frame in self.conditions.frames},
            "completed_steps": {str(frame.frame): 0 for frame in self.conditions.frames},
            "decoded_frames": [],
            "completed_frames": [],
            "chunk_rows": {str(frame.frame): [] for frame in self.conditions.frames},
            "chunk_completion_order": [],
        }
        self.pending: list[dict[str, Any]] = []
        self.outputs: dict[str, Any] = {}
        self.frame_artifacts: dict[int, dict[str, Any]] = {}
        self.chunk_artifacts: dict[int, dict[int, dict[str, Any]]] = {
            frame.frame: {} for frame in self.conditions.frames
        }
        self.manifest: dict[str, Any] | None = None

    def _cache_get(self, key: str) -> dict[str, Any] | None:
        return self.artifacts.get(key)

    def _store_artifact(self, key: str, artifact: dict[str, Any]) -> None:
        self.artifacts[key] = artifact

    def _condition_params(self, frame: FrameInput, condition_type: str) -> tuple[dict[str, Any], Callable[[], dict[str, Any]], list[dict[str, str]]]:
        profile = self.config.quant_profile
        common = {
            "recipe_version": RECIPE_VERSION,
            "encoder_version": f"{condition_type}-encoder-v1",
            "condition_type": condition_type,
            "condition_dim": self.config.condition_dim,
            "quant_profile": profile.name,
            "fraction_bits": profile.fraction_bits,
            "exchange_version": self.config.exchange_version,
        }
        if condition_type == "text":
            params = {
                **common,
                "prompt": {
                    "entities": sorted(set(frame.prompt["entities"])),
                    "attributes": sorted(set(frame.prompt["attributes"])),
                },
                "vocab_digest": self.conditions.vocab.digest,
            }

            def compute() -> dict[str, Any]:
                return encode_text(
                    frame.prompt,
                    self.conditions.vocab,
                    profile,
                    self.config.condition_dim,
                )
        elif condition_type == "image":
            params = {
                **common,
                "asset_sha256": frame.image_sha256,
                "width": self.config.image.width,
                "height": self.config.image.height,
                "mode": self.config.image.mode,
                "tile_size": self.config.image.tile_size,
            }

            def compute() -> dict[str, Any]:
                return encode_image(
                    frame.image_png,
                    config=self.config,
                    profile=profile,
                    asset_id=frame.image_id,
                )
        elif condition_type == "motion":
            params = {
                **common,
                "asset_sha256": frame.motion_sha256,
                "block_size": self.config.motion.block_size,
                "grid_shape": list(self.config.motion.grid_shape(self.config.image)),
            }

            def compute() -> dict[str, Any]:
                return encode_motion(
                    frame.motion_vectors,
                    config=self.config,
                    profile=profile,
                    asset_id=frame.motion_id,
                )
        else:  # pragma: no cover - guarded by callers
            fail("E_CONFIG_INVALID", "unknown condition type", condition_type=condition_type)
        return params, compute, []

    def _prepare_conditions(self) -> None:
        for frame in self.conditions.frames:
            frame_conditions: dict[str, dict[str, Any]] = {}
            for condition_type in ("image", "motion", "text"):
                params, compute, deps = self._condition_params(frame, condition_type)
                key = make_key("condition", params)
                logical_id = f"condition:{frame.frame}:{condition_type}"
                cached = self._cache_get(key)
                if cached is not None:
                    artifact = cached
                    hit = True
                    hit_origin = "committed" if key in self.prior_artifact_keys else "current_run"
                else:
                    artifact = compute()
                    artifact = {
                        **artifact,
                        "cache_key": key,
                        "logical_id": logical_id,
                        "source_version": self.config.exchange_version,
                    }
                    self._store_artifact(key, artifact)
                    hit = False
                    hit_origin = None
                self.ledger.descriptor(
                    stage="condition",
                    logical_id=logical_id,
                    cache_key=key,
                    deps=deps,
                    params=params,
                    digest=artifact["digest"],
                    hit=hit,
                    hit_origin=hit_origin,
                    extra={"condition_type": condition_type, "frame": frame.frame},
                )
                frame_conditions[condition_type] = artifact
            self.condition_artifacts[frame.frame] = frame_conditions

    def _prepare_fusion(self) -> None:
        for frame in self.conditions.frames:
            logical_id = f"fusion:{frame.frame}"
            sources = self.condition_artifacts[frame.frame]
            deps = [
                {
                    "logical_id": f"condition:{frame.frame}:{condition_type}",
                    "cache_key": sources[condition_type]["cache_key"],
                }
                for condition_type in sorted(sources)
            ]
            params = {
                "recipe_version": RECIPE_VERSION,
                "fusion_version": "sorted-groups-v1",
                "condition_cache_keys": {condition_type: sources[condition_type]["cache_key"] for condition_type in sorted(sources)},
                "condition_digests": {condition_type: sources[condition_type]["digest"] for condition_type in sorted(sources)},
                "weights_permille": {
                    condition_type: self.config.fusion_weights_permille[condition_type]
                    for condition_type in sorted(self.config.fusion_weights_permille)
                },
                "quant_profile": self.config.quant_profile.name,
                "fraction_bits": self.config.quant_profile.fraction_bits,
                "exchange_version": self.config.exchange_version,
            }
            key = make_key("fusion", params)
            cached = self._cache_get(key)
            if cached is not None:
                artifact = cached
                hit = True
                hit_origin = "committed" if key in self.prior_artifact_keys else "current_run"
            else:
                artifact = fuse_conditions(sources, config=self.config)
                artifact = {**artifact, "cache_key": key, "logical_id": logical_id}
                self._store_artifact(key, artifact)
                hit = False
                hit_origin = None
            self.ledger.descriptor(
                stage="fusion",
                logical_id=logical_id,
                cache_key=key,
                deps=deps,
                params=params,
                digest=artifact["digest"],
                hit=hit,
                hit_origin=hit_origin,
                extra={"frame": frame.frame},
            )
            self.fusions[frame.frame] = artifact

    def _prepare_latents(self) -> None:
        for frame in self.conditions.frames:
            logical_id = f"latent_init:{frame.frame}"
            fusion = self.fusions[frame.frame]
            deps = [{"logical_id": f"fusion:{frame.frame}", "cache_key": fusion["cache_key"]}]
            params = {
                "recipe_version": RECIPE_VERSION,
                "latent_version": "sha256-counter-v1",
                "fusion_digest": fusion["digest"],
                "fusion_key": fusion["cache_key"],
                "seed": self.config.seed,
                "frame": frame.frame,
                "seed_offset": frame.seed_offset,
                "latent_dim": self.config.latent_dim,
                "latent_config_fingerprint": self.config.latent_config_fingerprint,
                "quant_profile": self.config.quant_profile.name,
                "exchange_version": self.config.exchange_version,
            }
            key = make_key("latent_init", params)
            cached = self._cache_get(key)
            if cached is not None:
                artifact = cached
                hit = True
                hit_origin = "committed" if key in self.prior_artifact_keys else "current_run"
            else:
                artifact = initial_latent(
                    frame_number=frame.frame,
                    seed_offset=frame.seed_offset,
                    fusion_digest=fusion["digest"],
                    config=self.config,
                )
                artifact = {**artifact, "cache_key": key, "logical_id": logical_id}
                self._store_artifact(key, artifact)
                hit = False
                hit_origin = None
            self.ledger.descriptor(
                stage="latent_init",
                logical_id=logical_id,
                cache_key=key,
                deps=deps,
                params=params,
                digest=artifact["digest"],
                hit=hit,
                hit_origin=hit_origin,
                extra={"frame": frame.frame},
            )
            self.latents[frame.frame] = artifact

    def _step_params(
        self,
        frame: FrameInput,
        step: int,
        previous: dict[str, Any],
    ) -> dict[str, Any]:
        fusion = self.fusions[frame.frame]
        return {
            "recipe_version": RECIPE_VERSION,
            "denoise_version": "guidance-noise-decay-v1",
            "frame": frame.frame,
            "step": step,
            "previous_digest": previous["digest"],
            "previous_key": previous["cache_key"],
            "fusion_digest": fusion["digest"],
            "fusion_key": fusion["cache_key"],
            "guidance_permille": frame.guidance_permille,
            "noise_scheme": self.config.noise_scheme,
            "noise_code": self.config.noise_schemes[self.config.noise_scheme],
            "quant_profile": self.config.quant_profile.name,
            "fraction_bits": self.config.quant_profile.fraction_bits,
            "latent_dim": self.config.latent_dim,
            "condition_dim": self.config.condition_dim,
            "seed": self.config.seed,
            "seed_offset": frame.seed_offset,
            "exchange_version": self.config.exchange_version,
        }

    def _reserve_budget(self, misses: list[dict[str, Any]]) -> list[dict[str, Any]]:
        approved: list[dict[str, Any]] = []
        for task in sorted(misses, key=lambda item: (item["step"], item["frame"])):
            if self.batch_budget is None:
                allowed = True
            else:
                allowed = self.batch_spent < self.batch_budget
            if allowed:
                before = self.batch_spent
                if self.batch_budget is not None:
                    self.batch_spent += 1
                remaining = None if self.batch_budget is None else self.batch_budget - self.batch_spent
                self.budget_events.append(
                    {
                        "sequence": len(self.budget_events) + 1,
                        "stable_sort_key": task["sort_key"],
                        "logical_id": task["logical_id"],
                        "cost": 1,
                        "remaining_before": None if self.batch_budget is None else self.batch_budget - before,
                        "remaining_after": remaining,
                    }
                )
                approved.append(task)
            else:
                self.ledger.pending(
                    stage="denoise",
                    logical_id=task["logical_id"],
                    frame=task["frame"],
                    step=task["step"],
                )
        return approved

    def _execute_tasks(
        self,
        executor: ThreadPoolExecutor,
        tasks: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        if not tasks:
            return {}

        def wrapped(task: dict[str, Any]) -> dict[str, Any]:
            with self.probe.slot():
                if self.task_delay_seconds:
                    time.sleep(self.task_delay_seconds)
                return task["compute"]()

        futures: dict[Future[dict[str, Any]], dict[str, Any]] = {}
        for task in sorted(tasks, key=lambda item: item["sort_key"]):
            futures[executor.submit(wrapped, task)] = task
        results: dict[str, dict[str, Any]] = {}
        for future in as_completed(futures):
            task = futures[future]
            results[task["logical_id"]] = future.result()
        return results

    def _run_denoise(self, executor: ThreadPoolExecutor) -> None:
        for step in range(1, self.config.denoise_steps + 1):
            wave: list[dict[str, Any]] = []
            misses: list[dict[str, Any]] = []
            for frame in self.conditions.frames:
                if self.completed_steps[frame.frame] != step - 1:
                    continue
                previous = self.latents[frame.frame]
                params = self._step_params(frame, step, previous)
                key = make_key("denoise", params)
                logical_id = f"denoise:{frame.frame}:{step}"
                deps = [
                    {
                        "logical_id": f"latent_init:{frame.frame}" if step == 1 else f"denoise:{frame.frame}:{step - 1}",
                        "cache_key": previous["cache_key"],
                    },
                    {"logical_id": f"fusion:{frame.frame}", "cache_key": self.fusions[frame.frame]["cache_key"]},
                ]
                cached = self._cache_get(key)
                task = {
                    "frame": frame.frame,
                    "step": step,
                    "logical_id": logical_id,
                    "sort_key": ["denoise", step, frame.frame],
                    "key": key,
                    "deps": deps,
                    "params": params,
                    "hit": cached is not None,
                    "artifact": cached,
                    "frame_input": frame,
                    "previous": previous,
                }
                if cached is None:
                    def make_compute(current_task: dict[str, Any] = task) -> Callable[[], dict[str, Any]]:
                        frame_input = current_task["frame_input"]
                        previous_artifact = current_task["previous"]
                        step_number = current_task["step"]
                        step_key = current_task["key"]
                        step_logical_id = current_task["logical_id"]

                        def compute() -> dict[str, Any]:
                            artifact = denoise_step(
                                frame_number=frame_input.frame,
                                seed_offset=frame_input.seed_offset,
                                step=step_number,
                                previous=previous_artifact,
                                fusion=self.fusions[frame_input.frame],
                                guidance_permille=frame_input.guidance_permille,
                                config=self.config,
                                condition_refs=[
                                    self.condition_artifacts[frame_input.frame][condition_type]["cache_key"]
                                    for condition_type in sorted(self.condition_artifacts[frame_input.frame])
                                ],
                            )
                            return {
                                **artifact,
                                "cache_key": step_key,
                                "logical_id": step_logical_id,
                            }

                        return compute

                    task["compute"] = make_compute()
                    misses.append(task)
                wave.append(task)
            approved = self._reserve_budget(misses)
            compute_tasks = [task for task in approved if not task["hit"]]
            computed = self._execute_tasks(executor, compute_tasks)
            for task in sorted(wave, key=lambda item: item["sort_key"]):
                if task["hit"]:
                    artifact = task["artifact"]
                elif task in approved:
                    artifact = computed[task["logical_id"]]
                    self._store_artifact(task["key"], artifact)
                else:
                    continue
                self.ledger.descriptor(
                    stage="denoise",
                    logical_id=task["logical_id"],
                    cache_key=task["key"],
                    deps=task["deps"],
                    params=task["params"],
                    digest=artifact["digest"],
                    hit=task["hit"],
                    hit_origin=(
                        ("committed" if task["key"] in self.prior_artifact_keys else "current_run")
                        if task["hit"]
                        else None
                    ),
                    extra={"frame": task["frame"], "step": task["step"]},
                )
                self.latents[task["frame"]] = artifact
                self.step_artifacts[task["frame"]][task["step"]] = artifact
                self.completed_steps[task["frame"]] = task["step"]
                self.cursors["completed_steps"][str(task["frame"])] = task["step"]
                self.cursors["denoise_next_step"][str(task["frame"])] = task["step"] + 1
            self._update_pending()
            self._checkpoint()

    def _update_pending(self) -> None:
        pending: list[dict[str, Any]] = []
        for frame in self.conditions.frames:
            for step in range(1, self.config.denoise_steps + 1):
                if step in self.step_artifacts[frame.frame]:
                    continue
                pending.append(
                    {
                        "stage": "denoise",
                        "logical_id": f"denoise:{frame.frame}:{step}",
                        "frame": frame.frame,
                        "step": step,
                        "sort_key": ["denoise", step, frame.frame],
                        "code": (
                            E_BUDGET_PENDING
                            if self.batch_budget is not None and self.batch_spent >= self.batch_budget
                            else "PENDING_SCHEDULED"
                        ),
                    }
                )
        for frame in self.conditions.frames:
            missing = [item for item in pending if item["frame"] == frame.frame]
            if missing:
                self.cursors["denoise_next_step"][str(frame.frame)] = min(item["step"] for item in missing)
        pending.sort(key=lambda item: tuple(item["sort_key"]))
        self.pending = pending

    def _frame_recipe(
        self,
        frame: FrameInput,
        final_latent: dict[str, Any],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        step_chain = [
            {
                "step": step,
                "cache_key": self.step_artifacts[frame.frame][step]["cache_key"],
                "artifact_digest": self.step_artifacts[frame.frame][step]["digest"],
                "latent_digest": self.step_artifacts[frame.frame][step]["digest"],
                "round_directions": self.step_artifacts[frame.frame][step]["round_directions"],
            }
            for step in sorted(self.step_artifacts[frame.frame])
        ]
        params = {
            "recipe_version": RECIPE_VERSION,
            "decode_version": "linear-projection-v1",
            "frame": frame.frame,
            "final_latent_digest": final_latent["digest"],
            "final_latent_key": final_latent["cache_key"],
            "projection_recipe": projection_recipe(self.config),
            "quant_profile": self.config.quant_profile.name,
            "exchange_version": self.config.exchange_version,
            "image": {
                "width": self.config.image.width,
                "height": self.config.image.height,
                "mode": self.config.image.mode,
            },
            "step_chain_digest": content_digest(step_chain),
        }
        return params, step_chain

    def _decode_and_output(self, executor: ThreadPoolExecutor) -> None:
        complete_frames = [
            frame
            for frame in self.conditions.frames
            if len(self.step_artifacts[frame.frame]) == self.config.denoise_steps
        ]
        frame_recipes: dict[int, tuple[dict[str, Any], str, dict[str, Any], list[dict[str, Any]]]] = {}
        chunk_tasks: list[dict[str, Any]] = []
        for frame in complete_frames:
            final_latent = self.latents[frame.frame]
            params, step_chain = self._frame_recipe(frame, final_latent)
            frame_key = make_key("frame_decode", params)
            frame_recipes[frame.frame] = (params, frame_key, final_latent, step_chain)
            partitions = chunk_partition(self.config.image.height, self.config.chunk_rows)
            for partition in partitions:
                chunk_params = {
                    "recipe_version": RECIPE_VERSION,
                    "chunk_version": "row-band-png-v1",
                    "frame": frame.frame,
                    "frame_cache_key": frame_key,
                    "final_latent_digest": final_latent["digest"],
                    "chunk_index": partition["index"],
                    "start_row": partition["start_row"],
                    "end_row": partition["end_row"],
                    "projection_recipe": projection_recipe(self.config),
                    "quant_profile": self.config.quant_profile.name,
                    "exchange_version": self.config.exchange_version,
                }
                chunk_key = make_key("chunk", chunk_params)
                cached = self._cache_get(chunk_key)
                task = {
                    "frame": frame.frame,
                    "chunk_index": partition["index"],
                    "partition": partition,
                    "logical_id": f"chunk:{frame.frame}:{partition['index']}",
                    "sort_key": ["decode_chunk", frame.frame, partition["index"]],
                    "key": chunk_key,
                    "params": chunk_params,
                    "hit": cached is not None,
                    "artifact": cached,
                    "latent": final_latent,
                    "frame_key": frame_key,
                }
                if cached is None:
                    def make_chunk_compute(current_task: dict[str, Any] = task) -> Callable[[], dict[str, Any]]:
                        latent = current_task["latent"]
                        partition = current_task["partition"]
                        frame_number = current_task["frame"]
                        frame_cache_key = current_task["frame_key"]
                        chunk_key_value = current_task["key"]
                        chunk_logical_id = current_task["logical_id"]

                        def compute() -> dict[str, Any]:
                            pixels = decode_rows(
                                latent["vector"],
                                config=self.config,
                                start_row=partition["start_row"],
                                end_row=partition["end_row"],
                            )
                            png = encode_png_rgb(pixels)
                            raw = pixels.tobytes(order="C")
                            return {
                                "logical_id": chunk_logical_id,
                                "cache_key": chunk_key_value,
                                "frame": frame_number,
                                "chunk_index": partition["index"],
                                "row_range": [partition["start_row"], partition["end_row"]],
                                "frame_refs": [
                                    {
                                        "frame": frame_number,
                                        "frame_range": [frame_number, frame_number],
                                        "frame_cache_key": frame_cache_key,
                                    }
                                ],
                                "raw_pixels_b64": b64_encode(raw),
                                "pixel_digest": sha256_bytes(raw),
                                "png_b64": b64_encode(png),
                                "png_sha256": sha256_bytes(png),
                                "quant_profile": self.config.quant_profile.name,
                                "exchange_version": self.config.exchange_version,
                            }

                        return compute

                    task["compute"] = make_chunk_compute()
                chunk_tasks.append(task)
        misses = [task for task in chunk_tasks if not task["hit"]]
        computed = self._execute_tasks(executor, misses)
        for frame in complete_frames:
            frame_params, frame_key, final_latent, step_chain = frame_recipes[frame.frame]
            frame_tasks = [task for task in chunk_tasks if task["frame"] == frame.frame]
            frame_tasks.sort(key=lambda item: item["chunk_index"])
            bands: list[tuple[dict[str, Any], dict[str, Any]]] = []
            for task in frame_tasks:
                if task["hit"]:
                    artifact = task["artifact"]
                else:
                    artifact = computed[task["logical_id"]]
                    self._store_artifact(task["key"], artifact)
                deps = [{"logical_id": f"frame_decode:{frame.frame}", "cache_key": frame_key}]
                self.ledger.descriptor(
                    stage="chunk",
                    logical_id=task["logical_id"],
                    cache_key=task["key"],
                    deps=deps,
                    params=task["params"],
                    digest=artifact["png_sha256"],
                    hit=task["hit"],
                    hit_origin=(
                        ("committed" if task["key"] in self.prior_artifact_keys else "current_run")
                        if task["hit"]
                        else None
                    ),
                    extra={
                        "frame": frame.frame,
                        "chunk": task["chunk_index"],
                        "row_range": artifact["row_range"],
                        "frame_refs": artifact["frame_refs"],
                    },
                )
                self.chunk_artifacts[frame.frame][task["chunk_index"]] = artifact
                bands.append((task, artifact))
            raw_parts = [b64_decode(item["raw_pixels_b64"]) for _, item in bands]
            raw_full = b"".join(raw_parts)
            pixel_digest = sha256_bytes(raw_full)
            cached_frame = self._cache_get(frame_key)
            if cached_frame is not None:
                if cached_frame.get("pixel_digest") != pixel_digest:
                    fail(
                        E_SNAPSHOT_CORRUPT,
                        "cached frame pixels disagree with committed chunk bands",
                        frame=frame.frame,
                    )
                frame_artifact = cached_frame
                frame_hit = True
            else:
                import numpy as np

                full_shape = (self.config.image.height, self.config.image.width, 3)
                full_pixels = np.frombuffer(raw_full, dtype=np.uint8).reshape(full_shape)
                frame_png = encode_png_rgb(full_pixels)
                frame_artifact = {
                    "logical_id": f"frame_decode:{frame.frame}",
                    "cache_key": frame_key,
                    "frame": frame.frame,
                    "final_latent_digest": final_latent["digest"],
                    "step_chain": step_chain,
                    "step_refs": [
                        {
                            "step": item["step"],
                            "cache_key": item["cache_key"],
                            "artifact_digest": item["artifact_digest"],
                        }
                        for item in step_chain
                    ],
                    "chunk_keys": [item["key"] for item in frame_tasks],
                    "chunk_rows": self.config.chunk_rows,
                    "row_bands": [
                        {"index": artifact["chunk_index"], "row_range": artifact["row_range"]}
                        for _, artifact in bands
                    ],
                    "raw_pixels_b64": b64_encode(raw_full),
                    "pixel_digest": pixel_digest,
                    "png_b64": b64_encode(frame_png),
                    "png_sha256": sha256_bytes(frame_png),
                    "projection_recipe": projection_recipe(self.config),
                }
                self._store_artifact(frame_key, frame_artifact)
                frame_hit = False
            deps = [
                {
                    "logical_id": f"denoise:{frame.frame}:{step}",
                    "cache_key": self.step_artifacts[frame.frame][step]["cache_key"],
                }
                for step in range(1, self.config.denoise_steps + 1)
            ]
            self.ledger.descriptor(
                stage="frame_decode",
                logical_id=f"frame_decode:{frame.frame}",
                cache_key=frame_key,
                deps=deps,
                params=frame_params,
                digest=pixel_digest,
                hit=frame_hit,
                hit_origin=(
                    ("committed" if frame_key in self.prior_artifact_keys else "current_run")
                    if frame_hit
                    else None
                ),
                extra={"frame": frame.frame, "step_refs": frame_artifact["step_refs"]},
            )
            self.frame_artifacts[frame.frame] = frame_artifact
            self.cursors["decoded_frames"].append(frame.frame)
            self.cursors["chunk_rows"][str(frame.frame)] = [
                task["chunk_index"] for task in frame_tasks
            ]
            self.outputs[str(frame.frame)] = {
                "png_b64": frame_artifact["png_b64"],
                "png_sha256": frame_artifact["png_sha256"],
                "pixel_digest": frame_artifact["pixel_digest"],
                "chunks": [
                    {
                        "index": artifact["chunk_index"],
                        "row_range": artifact["row_range"],
                        "png_b64": artifact["png_b64"],
                        "png_sha256": artifact["png_sha256"],
                    }
                    for _, artifact in bands
                ],
            }
            self.cursors["chunk_completion_order"].extend(
                task["logical_id"] for task in frame_tasks
            )
            self.cursors["completed_frames"].append(frame.frame)

    def _state(self) -> dict[str, Any]:
        budget_spent_total = int(self.prior_state.get("budget_spent_total", 0)) + self.batch_spent
        return {
            "schema": "mmfm-render-state-v1",
            "recipe_version": RECIPE_VERSION,
            "run_id": self.run_id,
            "workspace": self.workspace.serialisable(),
            "config_hash": self.workspace.config_hash,
            "input_hash": self.workspace.input_hash,
            "artifacts": self.artifacts,
            "descriptors": self.ledger.merged_descriptors(),
            "cursor": self.cursors,
            "pending": self.pending,
            "budget": {
                "batch_limit": self.batch_budget,
                "batch_spent": self.batch_spent,
                "spent_total": budget_spent_total,
                "deduction_sequence": self.budget_events,
            },
            "budget_spent_total": budget_spent_total,
            "ledger_events": self.ledger.events,
            "ledger_summary": self.ledger.summary(),
            "outputs": self.outputs,
            "max_workers": self.config.max_workers,
            "max_observed_concurrency": self.probe.maximum,
            "executed_task_count": self.probe.executed,
            "pending_count": len(self.pending),
        }

    def _checkpoint(self) -> None:
        self.manifest = self.store.commit(
            self._state(),
            config_hash=self.workspace.config_hash,
            input_hash=self.workspace.input_hash,
            run_id=self.run_id,
        )

    def _frame_result(self, frame: FrameInput) -> dict[str, Any]:
        if frame.frame in self.frame_artifacts:
            artifact = self.frame_artifacts[frame.frame]
            return {
                "frame": frame.frame,
                "status": "complete",
                "pixel_digest": artifact["pixel_digest"],
                "png_sha256": artifact["png_sha256"],
                "step_refs": artifact["step_refs"],
                "row_bands": artifact["row_bands"],
                "chunks": [
                    {
                        "index": chunk["chunk_index"],
                        "row_range": chunk["row_range"],
                        "pixel_digest": chunk["pixel_digest"],
                        "png_sha256": chunk["png_sha256"],
                    }
                    for chunk in sorted(
                        self.chunk_artifacts[frame.frame].values(), key=lambda item: item["chunk_index"]
                    )
                ],
            }
        return {
            "frame": frame.frame,
            "status": "pending",
            "completed_steps": self.completed_steps[frame.frame],
            "code": next(
                (
                    item["code"]
                    for item in self.pending
                    if item.get("frame") == frame.frame and item.get("step") == self.completed_steps[frame.frame] + 1
                ),
                "PENDING_SCHEDULED",
            ),
        }

    def render(self) -> dict[str, Any]:
        executor = ThreadPoolExecutor(
            max_workers=self.config.max_workers,
            thread_name_prefix="mmfm",
        )
        try:
            self._prepare_conditions()
            self._prepare_fusion()
            self._prepare_latents()
            self._update_pending()
            self._checkpoint()
            self._run_denoise(executor)
            self._decode_and_output(executor)
            self._update_pending()
            self.cursors["completed_frames"] = sorted(set(self.cursors["completed_frames"]))
            self.cursors["decoded_frames"] = sorted(set(self.cursors["decoded_frames"]))
            self._checkpoint()
        finally:
            executor.shutdown(wait=True, cancel_futures=False)
        code = E_BUDGET_PENDING if self.pending else "OK"
        return {
            "ok": not self.pending,
            "code": code,
            "run_id": self.run_id,
            "config_hash": self.workspace.config_hash,
            "input_hash": self.workspace.input_hash,
            "generation": self.manifest.get("generation") if self.manifest else None,
            "global_version": self.manifest.get("global_version") if self.manifest else None,
            "config_version": self.config.exchange_version,
            "quant_profile": self.config.quant_profile.name,
            "denoise_steps": self.config.denoise_steps,
            "chunk_rows": self.config.chunk_rows,
            "max_workers": self.config.max_workers,
            "max_observed_concurrency": self.probe.maximum,
            "executed_task_count": self.probe.executed,
            "frames": [self._frame_result(frame) for frame in self.conditions.frames],
            "pending": self.pending,
            "cursor": self.cursors,
            "budget": {
                "batch_limit": self.batch_budget,
                "batch_spent": self.batch_spent,
                "spent_total": int(self.prior_state.get("budget_spent_total", 0)) + self.batch_spent,
                "deduction_sequence": self.budget_events,
            },
            "ledger_summary": self.ledger.summary(),
            "ledger_events": self.ledger.events,
            "snapshot": self.manifest,
            "uncommitted": self.store.uncommitted_report(),
            "chunk_completion_order": self.cursors["chunk_completion_order"],
        }

    def export_stage_records(self, stage: str = "denoise") -> list[Any]:
        records = []
        if stage == "denoise":
            for frame in self.conditions.frames:
                for step in sorted(self.step_artifacts[frame.frame]):
                    records.append(stage_records_for_step(self.step_artifacts[frame.frame][step]))
        return records
