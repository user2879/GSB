"""Deterministic layered generation, fixed-point mixing, and cache reuse."""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np

from .cache import (
    CacheStore,
    final_synthesis_key,
    material_channel_key,
    noise_layer_key,
)
from .errors import (
    E_INTERNAL,
    E_LAYER_RUNTIME,
    E_MATERIAL_RUNTIME,
    E_TEXTURE_MISSING,
    InjectedFailure,
    Issue,
    make_issue,
)
from .fixed import (
    CHANNELS,
    fixed_to_unit,
    quantize_fixed,
    quantization_residual,
    quantization_rounding,
    to_fixed,
)
from .injection import InjectionRegistry
from .mixing import artifact_from_cache, cache_payload, mix_channel
from .models import (
    BuildResult,
    GenerationConfig,
    LayerArtifact,
    Material,
    MaterialArtifact,
    MaterialChannelArtifact,
    Texture,
)
from .noise import derive_layer_seed, layer_payload, value_noise
from .scheduler import PointBudgetScheduler, WorkItem


@dataclass
class _LayerSpec:
    config_index: int
    texture: Texture
    texture_key: str
    key: str
    upstream_key: str
    cached: CacheEntryOrNone


CacheEntryOrNone = Any


@dataclass
class _ChannelSpec:
    material: Material
    channel: str
    key: str
    layer_keys: dict[int, str]
    cached: Any | None
    failed: bool = False


class MaterialEngine:
    """Thread-safe (serialized build) deterministic material engine."""

    def __init__(
        self,
        cache: CacheStore | None = None,
        injections: InjectionRegistry | None = None,
    ) -> None:
        self.cache = cache or CacheStore()
        self.injections = injections or InjectionRegistry()
        self._seen_texture_keys: set[str] = set()
        self._lock = threading.RLock()
        self.last_scheduler_events: list[dict[str, Any]] = []

    @staticmethod
    def _issue_once(issues: list[Issue], seen: set[tuple[str, str, str, str]], issue: Issue) -> None:
        key = (issue.code, issue.message, issue.source, issue.item)
        if key not in seen:
            seen.add(key)
            issues.append(issue)

    def _load_cached_layer(
        self,
        entry: Any,
        *,
        config: GenerationConfig,
        layer_index: int,
        key: str,
        texture_key: str,
        upstream_key: str,
        frequency: float,
        amplitude: float,
        issues: list[Issue],
        seen_issues: set[tuple[str, str, str, str]],
    ) -> LayerArtifact | None:
        try:
            meta = entry.meta
            arrays = entry.arrays
            if meta.get("layer_index") != layer_index or meta.get("texture_key") != texture_key:
                raise ValueError("cache metadata does not match layer")
            unquantized = np.asarray(arrays["unquantized"], dtype=np.int64)
            quantized = np.asarray(arrays["quantized"], dtype=np.int64)
            residual = np.asarray(arrays["residual"], dtype=np.int64)
            expected_shape = (config.resolution, config.resolution)
            if unquantized.shape != expected_shape or quantized.shape != expected_shape:
                raise ValueError("cache array shape mismatch")
            return LayerArtifact(
                index=layer_index,
                key=key,
                texture_key=texture_key,
                upstream_layer_key=upstream_key,
                frequency=frequency,
                amplitude=amplitude,
                unquantized=unquantized,
                quantized=quantized,
                residual=residual,
                scale=int(meta["scale"]),
                quantization_step=int(meta["quantization_step"]),
                reused=True,
            )
        except (KeyError, TypeError, ValueError) as exc:
            self._issue_once(
                issues,
                seen_issues,
                make_issue(
                    E_INTERNAL,
                    f"discarding malformed cached layer {layer_index}: {exc}",
                    source="cache",
                    item=key,
                ),
            )
            return None

    def build(
        self,
        config: GenerationConfig,
        materials: tuple[Material, ...],
        textures: dict[str, Texture],
    ) -> BuildResult:
        with self._lock:
            started = time.perf_counter()
            self.injections.reset_build()
            issues: list[Issue] = []
            seen_issues: set[tuple[str, str, str, str]] = set()
            ledger: list[dict[str, Any]] = []
            height = width = config.resolution
            scale = config.fixed_point.scale
            quant_rounding = quantization_rounding(config.fixed_point.rounding)
            scheduler = PointBudgetScheduler(config.point_budget)
            self.last_scheduler_events = scheduler.events

            texture_ledger_seen: set[str] = set()
            for layer_cfg in config.noise_layers:
                if layer_cfg.texture in textures and layer_cfg.texture not in texture_ledger_seen:
                    texture_ledger_seen.add(layer_cfg.texture)
                    texture = textures[layer_cfg.texture]
                    reused_texture = texture.cache_key in self._seen_texture_keys
                    self._seen_texture_keys.add(texture.cache_key)
                    ledger.append(
                        {
                            "kind": "texture",
                            "name": texture.name,
                            "key": texture.cache_key,
                            "content_hash": texture.content_hash,
                            "mode": texture.mode,
                            "size": [texture.width, texture.height],
                            "reused": reused_texture,
                            "status": "reused" if reused_texture else "loaded",
                        }
                    )

            specs: list[_LayerSpec] = []
            upstream_key = "ROOT"
            blocked_index: int | None = None
            for layer_cfg in config.noise_layers:
                if blocked_index is not None:
                    break
                texture = textures.get(layer_cfg.texture)
                if texture is None:
                    self._issue_once(
                        issues,
                        seen_issues,
                        make_issue(
                            E_TEXTURE_MISSING,
                            f"layer {layer_cfg.index} cannot resolve texture {layer_cfg.texture}",
                            source="engine",
                            item=layer_cfg.texture,
                        ),
                    )
                    blocked_index = layer_cfg.index
                    break
                key = noise_layer_key(
                    config=config,
                    layer_index=layer_cfg.index,
                    texture_key=texture.cache_key,
                    frequency=layer_cfg.frequency,
                    amplitude=layer_cfg.amplitude,
                    upstream_layer_key=upstream_key,
                )
                cached = self.cache.get(key)
                if cached is not None:
                    try:
                        if cached.meta.get("layer_index") != layer_cfg.index:
                            raise ValueError("layer index mismatch")
                        if cached.meta.get("texture_key") != texture.cache_key:
                            raise ValueError("texture key mismatch")
                        for array_name in ("unquantized", "quantized", "residual"):
                            if np.asarray(cached.arrays[array_name]).shape != (height, width):
                                raise ValueError(f"{array_name} shape mismatch")
                    except (KeyError, TypeError, ValueError) as exc:
                        self._issue_once(
                            issues,
                            seen_issues,
                            make_issue(
                                E_INTERNAL,
                                f"discarding malformed cached layer before scheduling: {exc}",
                                source="cache",
                                item=key,
                            ),
                        )
                        cached = None
                specs.append(
                    _LayerSpec(
                        config_index=layer_cfg.index,
                        texture=texture,
                        texture_key=texture.cache_key,
                        key=key,
                        upstream_key=upstream_key,
                        cached=cached,
                    )
                )
                upstream_key = key

            failed_layers: set[int] = set()
            layer_preflight_failure: dict[int, str] = {}
            for layer_cfg in config.noise_layers:
                try:
                    self.injections.before_layer(layer_cfg.index)
                except InjectedFailure as exc:
                    failed_layers.add(layer_cfg.index)
                    layer_preflight_failure[layer_cfg.index] = exc.injection_id
                    self._issue_once(
                        issues,
                        seen_issues,
                        make_issue(
                            exc.code,
                            str(exc),
                            source="layer_noise",
                            item=str(layer_cfg.index),
                            details={"injection_id": exc.injection_id},
                        ),
                    )

            # A cumulative layer cannot remain valid after an earlier layer fails.
            if failed_layers:
                first_failure = min(failed_layers)
                failed_layers.update(range(first_failure, config.noise_layer_count))

            noise_rows: dict[int, np.ndarray] = {}
            if any(spec.cached is None and spec.config_index not in failed_layers for spec in specs):
                noise_tasks: list[WorkItem] = []
                rows_per_task = max(1, min(8, config.point_budget // width))
                for spec in sorted(specs, key=lambda item: item.config_index):
                    if spec.cached is not None or spec.config_index in failed_layers:
                        continue
                    for row in range(0, height, rows_per_task):
                        row_end = min(height, row + rows_per_task)
                        noise_tasks.append(
                            WorkItem(
                                stage="noise",
                                stage_rank=0,
                                layer_index=spec.config_index,
                                row=row,
                                col_start=0,
                                col_end=width,
                                width=width,
                                payload={"layer_index": spec.config_index},
                                row_end=row_end,
                            )
                        )
                scheduler.add(noise_tasks)

            def run_noise(item: WorkItem) -> tuple[int, int, int, int, np.ndarray]:
                payload = item.payload or {}
                index = int(payload["layer_index"])
                layer_cfg = config.noise_layers[index]
                spec = next(spec for spec in specs if spec.config_index == index)
                seed_payload = layer_payload(
                    seed=config.seed,
                    index=index,
                    texture_key=spec.texture_key,
                    frequency=layer_cfg.frequency,
                    amplitude=layer_cfg.amplitude,
                    resolution=config.resolution,
                )
                seed = derive_layer_seed(seed_payload)
                noise = value_noise(seed, layer_cfg.frequency, item.row, item.row_stop, width)
                return index, item.row, item.row_stop, item.col_start, item.col_end, noise[:, item.col_start : item.col_end]

            def execute_batches(
                executor: ThreadPoolExecutor,
                runner: Any,
                failure_kind: str,
            ) -> list[tuple[WorkItem, Any]]:
                completed: list[tuple[WorkItem, Any]] = []
                while scheduler.pending:
                    batch = scheduler.take_batch()
                    if not batch:
                        break
                    future_map: dict[Future[Any], WorkItem] = {
                        executor.submit(runner, item): item for item in batch
                    }
                    for future, item in future_map.items():
                        try:
                            completed.append((item, future.result()))
                        except InjectedFailure as exc:
                            target = item.payload.get("layer_index", item.layer_index) if item.payload else item.layer_index
                            self._issue_once(
                                issues,
                                seen_issues,
                                make_issue(
                                    exc.code,
                                    str(exc),
                                    source=failure_kind,
                                    item=str(target),
                                    details={"injection_id": exc.injection_id},
                                ),
                            )
                            if failure_kind == "layer_noise":
                                failed_layers.add(int(target))
                        except Exception as exc:  # isolate one failed task
                            self._issue_once(
                                issues,
                                seen_issues,
                                make_issue(
                                    E_INTERNAL if failure_kind == "layer_noise" else E_MATERIAL_RUNTIME,
                                    f"{failure_kind} task failed: {exc}",
                                    source=failure_kind,
                                    item=str(item.sort_key),
                                ),
                            )
                return completed

            try:
                executor = ThreadPoolExecutor(max_workers=config.concurrency, thread_name_prefix="matgen")
            except Exception as exc:
                self._issue_once(
                    issues,
                    seen_issues,
                    make_issue(E_INTERNAL, f"cannot create thread pool: {exc}", source="scheduler"),
                )
                executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="matgen-fallback")

            with executor:
                for item, value in execute_batches(executor, run_noise, "layer_noise"):
                    index, row_start, row_end, col_start, col_end, noise = value
                    if index not in failed_layers:
                        noise_rows.setdefault(index, np.empty((height, width), dtype=np.float64))
                        noise_rows[index][row_start:row_end, col_start:col_end] = noise

                if failed_layers:
                    first = min(failed_layers)
                    if first not in layer_preflight_failure:
                        self._issue_once(
                            issues,
                            seen_issues,
                            make_issue(
                                E_LAYER_RUNTIME,
                                f"layer chain is incomplete after layer {first}",
                                source="layer_noise",
                                item=str(first),
                            ),
                        )
                    failed_layers.update(range(first, config.noise_layer_count))

                layers: list[LayerArtifact] = []
                layers_by_index: dict[int, LayerArtifact] = {}
                layer_keys: dict[int, str] = {spec.config_index: spec.key for spec in specs}
                previous_normalized: np.ndarray | None = None
                for layer_cfg in config.noise_layers:
                    spec = next((row for row in specs if row.config_index == layer_cfg.index), None)
                    if spec is None or layer_cfg.index in failed_layers:
                        continue
                    artifact: LayerArtifact | None = None
                    if spec.cached is not None:
                        artifact = self._load_cached_layer(
                            spec.cached,
                            config=config,
                            layer_index=layer_cfg.index,
                            key=spec.key,
                            texture_key=spec.texture_key,
                            upstream_key=spec.upstream_key,
                            frequency=layer_cfg.frequency,
                            amplitude=layer_cfg.amplitude,
                            issues=issues,
                            seen_issues=seen_issues,
                        )
                    if artifact is None:
                        noise = noise_rows.get(layer_cfg.index)
                        if noise is None or noise.shape != (height, width):
                            failed_layers.add(layer_cfg.index)
                            continue
                        texture_unit = spec.texture.luma.astype(np.float64) / 255.0
                        signal = np.clip(texture_unit + layer_cfg.amplitude * noise, 0.0, 1.0)
                        if layer_cfg.index == 0 or previous_normalized is None:
                            composed = signal
                        else:
                            composed = (previous_normalized * float(layer_cfg.index) + signal) / float(layer_cfg.index + 1)
                        composed = np.clip(composed, 0.0, 1.0)
                        unquantized = to_fixed(
                            composed,
                            config.fixed_point.fraction_bits,
                            config.fixed_point.bit_width,
                            quant_rounding,
                        )
                        quantized = quantize_fixed(unquantized, config.quantization_step, quant_rounding)
                        residual = quantization_residual(unquantized, quantized)
                        artifact = LayerArtifact(
                            index=layer_cfg.index,
                            key=spec.key,
                            texture_key=spec.texture_key,
                            upstream_layer_key=spec.upstream_key,
                            frequency=layer_cfg.frequency,
                            amplitude=layer_cfg.amplitude,
                            unquantized=unquantized.astype(np.int64),
                            quantized=quantized.astype(np.int64),
                            residual=residual.astype(np.int64),
                            scale=scale,
                            quantization_step=config.quantization_step,
                            reused=False,
                        )
                        self.cache.put(
                            spec.key,
                            kind="noise_layer",
                            meta={
                                "layer_index": layer_cfg.index,
                                "texture_key": spec.texture_key,
                                "upstream_layer_key": spec.upstream_key,
                                "frequency": layer_cfg.frequency,
                                "amplitude": layer_cfg.amplitude,
                                "scale": scale,
                                "quantization_step": config.quantization_step,
                            },
                            arrays={
                                "unquantized": artifact.unquantized,
                                "quantized": artifact.quantized,
                                "residual": artifact.residual,
                            },
                        )
                    layers.append(artifact)
                    layers_by_index[layer_cfg.index] = artifact
                    previous_normalized = fixed_to_unit(artifact.quantized, scale)
                    ledger.append(
                        {
                            "kind": "noise_layer",
                            "layer_index": layer_cfg.index,
                            "key": artifact.key,
                            "texture_key": artifact.texture_key,
                            "upstream_layer_key": artifact.upstream_layer_key,
                            "frequency": layer_cfg.frequency,
                            "amplitude": layer_cfg.amplitude,
                            "quantization_step": artifact.quantization_step,
                            "content_hash": artifact.content_hash(),
                            "reused": artifact.reused,
                            "status": "reused" if artifact.reused else "recomputed",
                        }
                    )

                channel_specs: list[_ChannelSpec] = []
                material_preflight_failed: set[str] = set()
                for material in materials:
                    try:
                        self.injections.before_material(material.material_id)
                    except InjectedFailure as exc:
                        material_preflight_failed.add(material.material_id)
                        self._issue_once(
                            issues,
                            seen_issues,
                            make_issue(
                                exc.code,
                                str(exc),
                                source="material_mix",
                                item=material.material_id,
                                details={"injection_id": exc.injection_id},
                            ),
                        )
                    for channel in CHANNELS:
                        referenced = [layer.layer_index for layer in material.layers if layer.weights[channel] > 0]
                        missing_refs = [index for index in referenced if index not in layers_by_index]
                        if missing_refs or material.material_id in material_preflight_failed:
                            if missing_refs:
                                self._issue_once(
                                    issues,
                                    seen_issues,
                                    make_issue(
                                        E_MATERIAL_RUNTIME,
                                        f"material {material.material_id} channel {channel} depends on unavailable layers",
                                        source="material_mix",
                                        item=f"{material.material_id}:{channel}",
                                        details={"missing_layers": missing_refs},
                                    ),
                                )
                            channel_specs.append(
                                _ChannelSpec(
                                    material=material,
                                    channel=channel,
                                    key="",
                                    layer_keys=layer_keys,
                                    cached=None,
                                    failed=True,
                                )
                            )
                            continue
                        key = material_channel_key(
                            material=material,
                            channel=channel,
                            layer_keys=layer_keys,
                            config=config,
                        )
                        cached = self.cache.get(key)
                        if cached is not None:
                            try:
                                if np.asarray(cached.arrays["mixed"]).shape != (height, width):
                                    raise ValueError("mixed shape mismatch")
                                if np.asarray(cached.arrays["final"]).shape != (height, width):
                                    raise ValueError("final shape mismatch")
                            except (KeyError, TypeError, ValueError) as exc:
                                self._issue_once(
                                    issues,
                                    seen_issues,
                                    make_issue(
                                        E_INTERNAL,
                                        f"discarding malformed cached channel before scheduling: {exc}",
                                        source="cache",
                                        item=key,
                                    ),
                                )
                                cached = None
                        channel_specs.append(
                            _ChannelSpec(
                                material=material,
                                channel=channel,
                                key=key,
                                layer_keys=layer_keys,
                                cached=cached,
                            )
                        )

                partial_accumulators: dict[str, dict[str, Any]] = {}
                material_tasks: list[WorkItem] = []
                for material_index, material in enumerate(materials):
                    for channel_spec in [row for row in channel_specs if row.material.material_id == material.material_id]:
                        if channel_spec.failed or channel_spec.cached is not None:
                            continue
                        accumulator_key = f"{material.material_id}:{channel_spec.channel}"
                        partial_accumulators[accumulator_key] = {
                            "mixed": np.zeros((height, width), dtype=np.int64),
                            "final": np.zeros((height, width), dtype=np.int64),
                            "contributions": {
                                layer.layer_index: np.zeros((height, width), dtype=np.int64)
                                for layer in material.layers
                                if layer.weights[channel_spec.channel] > 0
                            },
                            "samples": [],
                            "boundary_total": 0,
                            "failed": False,
                            "completed_points": 0,
                        }
                        rows_per_task = max(1, min(8, config.point_budget // width))
                        channel_rank = CHANNELS.index(channel_spec.channel)
                        for row in range(0, height, rows_per_task):
                            row_end = min(height, row + rows_per_task)
                            material_tasks.append(
                                WorkItem(
                                    stage="material",
                                    stage_rank=1,
                                    layer_index=(
                                        config.noise_layer_count
                                        + material_index * len(CHANNELS)
                                        + channel_rank
                                    ),
                                    row=row,
                                    col_start=0,
                                    col_end=width,
                                    width=width,
                                    payload={
                                        "material_id": material.material_id,
                                        "channel": channel_spec.channel,
                                        "key": channel_spec.key,
                                    },
                                    row_end=row_end,
                                )
                            )

                def run_material(item: WorkItem) -> tuple[str, str, int, int, int, int, MaterialChannelArtifact]:
                    payload = item.payload or {}
                    material_id = str(payload["material_id"])
                    channel = str(payload["channel"])
                    key = str(payload["key"])
                    material = next(row for row in materials if row.material_id == material_id)
                    partial = mix_channel(
                        material=material,
                        channel=channel,
                        layers=layers_by_index,
                        key=key,
                        scale=scale,
                        weight_fraction_bits=config.fixed_point.weight_fraction_bits,
                        mix_mode=config.material_mix_mode,
                        rounding=config.fixed_point.rounding[channel],
                        max_boundary_samples=config.max_boundary_samples,
                        row_start=item.row,
                        row_end=item.row_stop,
                        col_start=item.col_start,
                        col_end=item.col_end,
                    )
                    return material_id, channel, item.row, item.row_stop, item.col_start, item.col_end, partial

                if material_tasks:
                    scheduler.add(material_tasks)
                    for item, value in execute_batches(executor, run_material, "material_mix"):
                        material_id, channel, row_start, row_end, col_start, col_end, partial = value
                        accumulator = partial_accumulators[f"{material_id}:{channel}"]
                        accumulator["mixed"][row_start:row_end, col_start:col_end] = partial.mixed
                        accumulator["final"][row_start:row_end, col_start:col_end] = partial.final
                        for index, values in partial.contributions.items():
                            accumulator["contributions"][index][row_start:row_end, col_start:col_end] = values
                        accumulator["boundary_total"] += partial.boundary_total
                        accumulator["completed_points"] += (row_end - row_start) * (col_end - col_start)
                        remaining = max(0, config.max_boundary_samples - len(accumulator["samples"]))
                        if remaining:
                            accumulator["samples"].extend(partial.boundary_samples[:remaining])

                material_artifacts: dict[str, MaterialArtifact] = {}
                for material in materials:
                    channel_artifacts: dict[str, MaterialChannelArtifact] = {}
                    for channel_spec in [row for row in channel_specs if row.material.material_id == material.material_id]:
                        if channel_spec.failed:
                            continue
                        artifact: MaterialChannelArtifact | None = None
                        if channel_spec.cached is not None:
                            try:
                                artifact = artifact_from_cache(
                                    channel_spec.cached,
                                    material_id=material.material_id,
                                    channel=channel_spec.channel,
                                )
                            except (KeyError, TypeError, ValueError) as exc:
                                self._issue_once(
                                    issues,
                                    seen_issues,
                                    make_issue(
                                        E_INTERNAL,
                                        f"discarding malformed cached channel: {exc}",
                                        source="cache",
                                        item=channel_spec.key,
                                    ),
                                )
                                artifact = None
                        if artifact is None:
                            accumulator = partial_accumulators.get(f"{material.material_id}:{channel_spec.channel}")
                            if (
                                accumulator is None
                                or accumulator["failed"]
                                or accumulator["completed_points"] != height * width
                            ):
                                self._issue_once(
                                    issues,
                                    seen_issues,
                                    make_issue(
                                        E_MATERIAL_RUNTIME,
                                        "material channel work is incomplete",
                                        source="material_mix",
                                        item=f"{material.material_id}:{channel_spec.channel}",
                                        details={
                                            "completed_points": 0
                                            if accumulator is None
                                            else accumulator["completed_points"],
                                            "required_points": height * width,
                                        },
                                    ),
                                )
                                continue
                            referenced = [
                                layer.layer_index
                                for layer in material.layers
                                if layer.weights[channel_spec.channel] > 0
                            ]
                            artifact = MaterialChannelArtifact(
                                key=channel_spec.key,
                                material_id=material.material_id,
                                channel=channel_spec.channel,
                                layer_keys=tuple(layers_by_index[index].key for index in referenced),
                                layer_indices=tuple(referenced),
                                mixed=accumulator["mixed"],
                                final=accumulator["final"],
                                contributions=accumulator["contributions"],
                                boundary_samples=accumulator["samples"],
                                boundary_total=accumulator["boundary_total"],
                                reused=False,
                            )
                            meta, arrays = cache_payload(artifact)
                            self.cache.put(
                                channel_spec.key,
                                kind="material_channel",
                                meta=meta,
                                arrays=arrays,
                            )
                        channel_artifacts[channel_spec.channel] = artifact
                        ledger.append(
                            {
                                "kind": "material_channel",
                                "material_id": material.material_id,
                                "channel": channel_spec.channel,
                                "key": artifact.key,
                                "layer_keys": list(artifact.layer_keys),
                                "layer_indices": list(artifact.layer_indices),
                                "content_hash": artifact.content_hash(),
                                "boundary_total": artifact.boundary_total,
                                "reused": artifact.reused,
                                "status": "reused" if artifact.reused else "recomputed",
                            }
                        )
                    if channel_artifacts:
                        material_artifacts[material.material_id] = MaterialArtifact(
                            material_id=material.material_id,
                            channels=channel_artifacts,
                        )

                complete = all(
                    material.material_id in material_artifacts
                    and len(material_artifacts[material.material_id].channels) == len(CHANNELS)
                    for material in materials
                )
                channel_keys = {
                    material_id: {
                        channel: artifact.key
                        for channel, artifact in artifact_group.channels.items()
                    }
                    for material_id, artifact_group in material_artifacts.items()
                }
                if complete:
                    final_key = final_synthesis_key(channel_keys=channel_keys, config=config)
                    all_channels_reused = all(
                        artifact.reused
                        for artifact_group in material_artifacts.values()
                        for artifact in artifact_group.channels.values()
                    )
                    final_reused = self.cache.contains(final_key) and all_channels_reused
                    if not final_reused:
                        self.cache.put(
                            final_key,
                            kind="final_synthesis",
                            meta={"channel_keys": channel_keys, "material_count": len(material_artifacts)},
                        )
                    ledger.append(
                        {
                            "kind": "final_synthesis",
                            "key": final_key,
                            "reused": final_reused,
                            "status": "reused" if final_reused else "recomputed",
                            "material_count": len(material_artifacts),
                        }
                    )
                else:
                    final_key = final_synthesis_key(
                        channel_keys={key: {channel: "INCOMPLETE" for channel in CHANNELS} for key in channel_keys},
                        config=config,
                    )

            recomputed = [
                f"{entry['kind']}:{entry.get('layer_index', entry.get('material_id', ''))}:{entry.get('channel', '')}:{entry['key'][:12]}"
                for entry in ledger
                if entry["status"] == "recomputed"
            ]
            reused = [
                f"{entry['kind']}:{entry.get('layer_index', entry.get('material_id', ''))}:{entry.get('channel', '')}:{entry['key'][:12]}"
                for entry in ledger
                if entry["status"] in {"reused", "loaded"}
            ]
            affected_layers = sorted(
                int(entry["layer_index"])
                for entry in ledger
                if entry["kind"] == "noise_layer" and entry["status"] == "recomputed"
            )
            affected_channels = [
                f"{entry['material_id']}:{entry['channel']}"
                for entry in ledger
                if entry["kind"] == "material_channel" and entry["status"] == "recomputed"
            ]
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self.last_scheduler_events = scheduler.events
            return BuildResult(
                config=config,
                materials=materials,
                textures=textures,
                layers=layers,
                material_artifacts=material_artifacts,
                final_key=final_key,
                final_reused=final_reused if complete else False,
                issues=issues,
                ledger_entries=ledger,
                scheduler_events=list(scheduler.events),
                recomputed=recomputed,
                reused=reused,
                affected_layers=affected_layers,
                affected_channels=affected_channels,
                batches_consumed=scheduler.batch_number,
                build_elapsed_ms=elapsed_ms,
            )
