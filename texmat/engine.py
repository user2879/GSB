"""Orchestration: incremental cache reuse, staged execution and snapshots."""

from __future__ import annotations

import io
import signal
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .cache import (DependencyGraph, build_graph, channel_node_id, final_node_id,
                    layer_node_id)
from .config import CHANNELS, GenerationConfig, MaterialSpec
from .diagnostics import compare_gears, run_gear
from .hashing import config_hash
from .pipeline import LayerResult, compose_material, compute_layer
from .loader import DecodedTexture
from .scheduler import Scheduler, Task, run_in_key_order, stable_channel_key, stable_final_key, stable_layer_key
from .snapshot import ArtifactState, Manifest, SnapshotStore


@dataclass
class EngineRunResult:
    complete: bool
    batches: int
    reused: list[str]
    recomputed: list[str]
    budget_events: list[dict[str, Any]]
    budget_remaining: int
    pending: list[str]
    failures: list[dict[str, Any]]
    stopped_by_signal: bool
    cursor: list[str]
    manifest_version: int
    recovery: dict[str, Any]
    mismatch: dict[str, Any] | None


class TextureEngine:
    def __init__(self, config: GenerationConfig, materials: list[MaterialSpec],
                 textures: dict[str, DecodedTexture], snapshot_dir: str | Path,
                 material_raw: list[dict[str, Any]], *,
                 strict_resume: bool = False, stop_after_commits: int | None = None):
        self.config = config
        self.materials = materials
        self.textures = textures
        self.material_raw = material_raw
        self.snapshot_dir = Path(snapshot_dir)
        self.store = SnapshotStore(self.snapshot_dir)
        self.graph: DependencyGraph = build_graph(
            config, materials, {name: tex.content_fingerprint for name, tex in textures.items()})
        self.config_hash_value = config_hash(config.raw_json, material_raw)
        self.strict_resume = strict_resume
        self.stop_after_commits = stop_after_commits
        self._commits_this_run = 0
        self._stop_requested = threading.Event()
        self._manifest: Manifest | None = None

    def install_signal_handlers(self) -> dict[int, Any]:
        previous: dict[int, Any] = {}
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                previous[sig] = signal.signal(sig, self._handle_signal)
            except (ValueError, OSError):
                continue
        return previous

    def _handle_signal(self, signum: int, frame: Any) -> None:
        self._stop_requested.set()

    def _load_or_init_manifest(self) -> tuple[Manifest, dict[str, Any], dict[str, Any] | None]:
        mismatch: dict[str, Any] | None = None
        recovery: dict[str, Any] = {}
        try:
            manifest = self.store.check_resume(self.config_hash_value, strict=self.strict_resume)
        except Exception:
            raise
        if manifest is not None and manifest.config_hash != self.config_hash_value:
            mismatch = {"code": "E_CONFIG_MISMATCH",
                        "stored": manifest.config_hash, "current": self.config_hash_value}
            recovery = self.store.recovery_point(manifest)
            manifest = None
        if manifest is None:
            manifest = Manifest(
                global_version=0, config_hash=self.config_hash_value,
                fractional_bits=self.config.fractional_bits, blend_mode=self.config.blend_mode,
                completed_cursor=[], artifacts={},
            )
        else:
            recovery = self.store.recovery_point(manifest)
        self._manifest = manifest
        return manifest, recovery, mismatch

    def _cache_hit(self, node_id: str, manifest: Manifest) -> ArtifactState | None:
        state = manifest.artifacts.get(node_id)
        expected = self.graph.nodes[node_id]
        if state is None or state.cache_key != expected.cache_key:
            return None
        if not self.store.artifact_exists(state.payload_path):
            return None
        try:
            data = self.store.load_npz(state.payload_path)
        except Exception:
            return None
        verify = data.get("fingerprint")
        if verify is None or int(verify[0]) != expected.fingerprint:
            return None
        return state

    def _plan_tasks(self, manifest: Manifest) -> tuple[list[Task], list[str]]:
        tasks: list[Task] = []
        reusable: list[str] = []
        pixels = self.config.active_tier * self.config.active_tier
        for spec in self.config.noise_layers:
            node_id = layer_node_id(spec.index)
            if self._cache_hit(node_id, manifest) is not None:
                reusable.append(node_id)
                continue
            tasks.append(Task(key=stable_layer_key(spec.index), group=0, sequence=spec.index,
                              points=pixels, node_id=node_id, payload={"layer_index": spec.index}))
        for material_order, material in enumerate(self.materials):
            for channel_order, channel in enumerate(CHANNELS):
                node_id = channel_node_id(material.name, channel)
                if self._cache_hit(node_id, manifest) is not None:
                    reusable.append(node_id)
                    continue
                tasks.append(Task(key=stable_channel_key(material_order, channel_order), group=1,
                                  sequence=material_order * len(CHANNELS) + channel_order,
                                  points=pixels, node_id=node_id,
                                  payload={"material": material.name, "channel": channel}))
            final_id = final_node_id(material.name)
            if self._cache_hit(final_id, manifest) is not None:
                reusable.append(final_id)
                continue
            tasks.append(Task(key=stable_final_key(material_order), group=2,
                              sequence=material_order, points=pixels * len(CHANNELS),
                              node_id=final_id, payload={"material": material.name}))
        return tasks, sorted(reusable)

    def _previous_wide(self, index: int) -> np.ndarray | None:
        if index == 0:
            return None
        spec = self.config.noise_layers[index]
        previous_spec = self.config.noise_layers[index - 1]
        if spec.source_texture != previous_spec.source_texture:
            return None
        prev_state = self._manifest.artifacts.get(layer_node_id(index - 1))  # type: ignore[union-attr]
        if prev_state is None or not self.store.artifact_exists(prev_state.payload_path):
            return None
        return self.store.load_npz(prev_state.payload_path)["accum"]

    def _execute_layer(self, task: Task, gear_f: int) -> dict[str, np.ndarray]:
        index = task.payload["layer_index"]
        spec = self.config.noise_layers[index]
        texture = self.textures[spec.source_texture]
        previous_wide = self._previous_wide(index) if gear_f == self.config.fractional_bits else None
        result = compute_layer(spec, texture, previous_wide, self.config,
                               f_bits=gear_f, quantize=self.config.quantize,
                               rounding=self.config.rounding)
        node_id = layer_node_id(spec.index)
        node_fp = self.graph.nodes[node_id].fingerprint if gear_f == self.config.fractional_bits else result.fingerprint
        return {
            "accum": result.accum,
            "quantized": result.quantized if result.quantized.size else np.empty(0, dtype=np.int64),
            "noise_q": result.noise_q,
            "fingerprint": np.array([node_fp], dtype=np.uint64),
            "summary": np.array([result.summary], dtype=np.int64),
        }

    def _load_layer_results(self, gear_f: int) -> dict[str, LayerResult]:
        results: dict[str, LayerResult] = {}
        for spec in self.config.noise_layers:
            node_id = layer_node_id(spec.index)
            state = self._manifest.artifacts[node_id]  # type: ignore[union-attr]
            data = self.store.load_npz(state.payload_path)
            quantized = data["quantized"] if data["quantized"].size else np.empty(0, dtype=np.int64)
            node_id = layer_node_id(spec.index)
            node_fp = self.graph.nodes[node_id].fingerprint
            results[f"layer{spec.index:02d}"] = LayerResult(
                spec=spec, accum=data["accum"], quantized=quantized,
                is_quantized=bool(quantized.size), noise_q=data["noise_q"],
                summary=int(data["summary"][0]), fingerprint=node_fp,
            )
        return results

    def _execute_channel(self, task: Task, gear_f: int) -> dict[str, np.ndarray]:
        material = next(item for item in self.materials if item.name == task.payload["material"])
        channel = task.payload["channel"]
        if gear_f == self.config.fractional_bits:
            layer_results = self._load_layer_results(gear_f)
        else:
            layer_results = self._compute_gear_layers(gear_f)
        result = compose_material(material, layer_results, self.config, gear_f, self.config.rounding)
        channel_result = result.channels[channel]
        boundary = np.asarray(channel_result.boundary_pixels, dtype=np.int64).reshape(-1, 2)
        tie = np.asarray(channel_result.tie_pixels, dtype=np.int64).reshape(-1, 2)
        node_id = channel_node_id(material.name, channel)
        node_fp = self.graph.nodes[node_id].fingerprint if gear_f == self.config.fractional_bits else channel_result.fingerprint
        return {
            "values": channel_result.values,
            "contributions": channel_result.contributions,
            "boundary": boundary,
            "tie": tie,
            "fingerprint": np.array([node_fp], dtype=np.uint64),
        }

    def _execute_final(self, task: Task, gear_f: int) -> dict[str, np.ndarray]:
        material = next(item for item in self.materials if item.name == task.payload["material"])
        if gear_f == self.config.fractional_bits:
            layer_results = self._load_layer_results(gear_f)
        else:
            layer_results = self._compute_gear_layers(gear_f)
        result = compose_material(material, layer_results, self.config, gear_f, self.config.rounding)
        node_id = final_node_id(material.name)
        node_fp = self.graph.nodes[node_id].fingerprint if gear_f == self.config.fractional_bits else result.fingerprint
        return {"rgba": result.rgba, "fingerprint": np.array([node_fp], dtype=np.uint64)}

    def _compute_gear_layers(self, gear_f: int) -> dict[str, LayerResult]:
        results: dict[str, LayerResult] = {}
        previous_wide: np.ndarray | None = None
        previous_source: str | None = None
        for spec in self.config.noise_layers:
            texture = self.textures[spec.source_texture]
            if previous_wide is None or spec.source_texture != previous_source:
                previous_wide = None
            result = compute_layer(spec, texture, previous_wide, self.config,
                                   f_bits=gear_f, quantize=self.config.quantize,
                                   rounding=self.config.rounding)
            results[f"layer{spec.index:02d}"] = result
            previous_wide = result.accum
            previous_source = spec.source_texture
        return results

    def _write_png(self, material_name: str, rgba: np.ndarray) -> None:
        image = Image.fromarray(rgba, mode="RGBA")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        self.store.save_png_bytes(f"png/{material_name}.png", buffer.getvalue())

    def _commit_task(self, task: Task, data: dict[str, np.ndarray], manifest: Manifest) -> None:
        record = self.graph.nodes[task.node_id]
        relative = record.payload_path or f"tmp/{task.node_id}.npz"
        self.store.save_npz(relative, **data)
        manifest.artifacts[task.node_id] = ArtifactState(
            node_id=task.node_id, cache_key=record.cache_key,
            version=manifest.global_version + 1, payload_path=relative,
            fingerprint=record.fingerprint,
        )
        manifest.global_version += 1
        if task.key not in manifest.completed_cursor:
            manifest.completed_cursor.append(task.key)
        if task.node_id.startswith("final:"):
            self._write_png(task.payload["material"], data["rgba"])
        if len(manifest.completed_cursor) % self.config.snapshot_interval == 0:
            self.store.commit_manifest(manifest)

    def _run_phase(self, phase_tasks: list[Task], manifest: Manifest,
                   budget_events: list[dict[str, Any]], budget_remaining: list[int],
                   recomputed: list[str], pending_nodes: list[str],
                   failures: list[dict[str, Any]]) -> bool:
        """Run one dependency phase; returns False when a stop was requested."""
        scheduler = Scheduler(budget_remaining[0])
        admitted = scheduler.admit(phase_tasks)
        budget_events.extend(scheduler.deduction_sequence())
        budget_remaining[0] = scheduler.remaining
        pending_nodes.extend(task.node_id for task in scheduler.pending)
        failures.extend(failure.to_dict() for failure in scheduler.pending_failures())

        def execute(task: Task) -> dict[str, np.ndarray]:
            if task.node_id.startswith("layer:"):
                return self._execute_layer(task, self.config.fractional_bits)
            if task.node_id.startswith("channel:"):
                return self._execute_channel(task, self.config.fractional_bits)
            return self._execute_final(task, self.config.fractional_bits)

        results = run_in_key_order(admitted, self.config.workers, execute)
        recomputed.extend(task.node_id for task in admitted)
        for task in sorted(admitted, key=lambda item: (item.group, item.sequence, item.key)):
            self._commit_task(task, results[task.key], manifest)
            self._commits_this_run += 1
            if self.stop_after_commits is not None and self._commits_this_run >= self.stop_after_commits:
                self._stop_requested.set()
        self.store.commit_manifest(manifest)
        return not self._stop_requested.is_set()

    def execute_run(self) -> EngineRunResult:
        """Execute up to one run: phases advance until a recovery point appears."""
        manifest, recovery, mismatch = self._load_or_init_manifest()
        all_tasks, reusable = self._plan_tasks(manifest)
        phases = [
            [task for task in all_tasks if task.group == group]
            for group in (0, 1, 2)
        ]
        budget_events: list[dict[str, Any]] = []
        budget_remaining = [self.config.point_budget]
        recomputed: list[str] = []
        pending_nodes: list[str] = []
        failures: list[dict[str, Any]] = []
        batches = 0
        stopped = False
        for phase_index, phase_tasks in enumerate(phases):
            if not phase_tasks:
                continue
            batches += 1
            proceed = self._run_phase(phase_tasks, manifest, budget_events, budget_remaining,
                                      recomputed, pending_nodes, failures)
            if not proceed:
                stopped = True
                break
            # A phase with pending tasks is a recovery point: downstream phases
            # are retained with explicit pending markers and the run exits so a
            # fresh process can resume from the committed manifest.
            if pending_nodes:
                downstream = [task for later in phases[phase_index + 1:] for task in later]
                for task in sorted(downstream, key=lambda item: (item.group, item.sequence, item.key)):
                    pending_nodes.append(task.node_id)
                    failures.append({
                        "code": "E_POINT_BUDGET_PENDING",
                        "target": task.node_id,
                        "message": "upstream phase incomplete; retained until resume",
                        "order_key": task.key,
                        "details": {"blocked_by_budget": True},
                    })
                break
            budget_remaining[0] = self.config.point_budget
        complete = not pending_nodes and not stopped
        return EngineRunResult(
            complete=complete, batches=batches, reused=reusable,
            recomputed=sorted(set(recomputed)), budget_events=budget_events,
            budget_remaining=budget_remaining[0], pending=sorted(set(pending_nodes)),
            failures=failures, stopped_by_signal=stopped,
            cursor=list(manifest.completed_cursor),
            manifest_version=manifest.global_version, recovery=recovery, mismatch=mismatch,
        )

    def execute_batch(self) -> EngineRunResult:
        return self.execute_run()

    def run_to_completion(self, max_runs: int = 10000) -> EngineRunResult:
        """Single execution up to the first recovery point.

        Multi-run resume is driven by the caller (CLI/test harness) with a fresh
        engine instance, which genuinely resets the per-run point budget and
        re-reads the committed manifest.
        """
        return self.execute_run()

    def build_diagnostics(self) -> dict[str, Any]:
        base_gear = run_gear(self.config, self.materials, self.textures, self.config.fractional_bits)
        ref_gear = run_gear(self.config, self.materials, self.textures, self.config.fractional_bits + 1)
        return compare_gears(base_gear, ref_gear, self.config, self.materials)

    def full_reference_run(self) -> dict[str, Any]:
        gear = run_gear(self.config, self.materials, self.textures, self.config.fractional_bits)
        return {
            "layers": {
                node_id: {"accum": result.accum, "quantized": result.quantized,
                          "summary": result.summary, "fingerprint": result.fingerprint}
                for node_id, result in gear.layers.items()
            },
            "materials": {
                name: {"rgba": result.rgba, "fingerprint": result.fingerprint,
                       "channels": {channel: channel_result.values
                                    for channel, channel_result in result.channels.items()}}
                for name, result in gear.materials.items()
            },
        }
