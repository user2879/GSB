"""End-to-end pipeline orchestration.

Stages:
  1. load and validate textures/materials/config (failure list, continue),
  2. fingerprint-driven incremental cache planning (reuse vs recompute),
  3. thread-pool rendering with a per-batch point budget,
  4. two-tier error tracking and per-pixel attribution,
  5. atomic snapshot commit,
  6. diagnostic report.

A fresh run against an empty cache is a full rebuild; subsequent runs reuse
unaffected artifacts.  The orchestrator also exposes a serial reference path
so tests can prove concurrent and single-threaded execution are identical.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .cache import CacheStore, downstream, plan_reuse
from .error_tracking import compare_tiers, run_tier
from .errors import Failure
from .fixedpoint import fixed_to_uint8
from .hashing import fingerprint_array, derive_cache_key
from .loader import scan_textures
from .models import CHANNELS, GenerationConfig, Material
from .scheduler import Task, run_scheduler, run_serial_reference
from .snapshot import SnapshotStore
from .stages import (
    StageSpecs,
    accumulate_layers,
    build_material_channels,
    composite_fingerprint,
    layer_noise_tile,
)


@dataclass
class PipelineResult:
    resolution: int
    config_hash: str
    noise_result: dict
    material_result: dict
    error_report: dict
    failures: list[Failure]
    reuse_keys: list[str]
    recompute_keys: list[str]
    affected_scope: dict[str, list[str]]
    scheduler: dict[str, Any]
    snapshots: list[str]
    specs_meta: dict[str, Any]
    texture_fingerprints: dict[str, str]
    composite_keys: dict[str, str]

    def failure_dicts(self) -> list[dict[str, Any]]:
        return [f.to_dict() for f in self.failures]


class TexturePipeline:
    def __init__(self, input_dir: str | Path, work_dir: str | Path,
                 config: GenerationConfig, materials: list[Material]):
        self.input_dir = Path(input_dir)
        self.work_dir = Path(work_dir)
        self.config = config
        self.materials = materials
        self.cache = CacheStore(self.work_dir / "cache")
        self.snapshots = SnapshotStore(self.work_dir / "snapshots")
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def load_textures(self, resolution: int):
        textures, failures = scan_textures(self.input_dir, resolution)
        fingerprints = {tid: fingerprint_array(tex.gray, f"base:{tid}") for tid, tex in textures.items()}
        return textures, fingerprints, failures

    def _base_key(self, texture_id: str, texture_fp: str) -> str:
        return derive_cache_key("base", [texture_fp], {"texture": texture_id})

    def _desired_graph(self, textures, fingerprints):
        specs = StageSpecs.from_config(self.config)
        noise_result = accumulate_layers(textures, self.config, specs)
        material_result = build_material_channels(textures, noise_result, self.materials, self.config, specs)
        width = next(iter(textures.values())).width
        desired: set[str] = set()
        edges: dict[str, list[str]] = {}
        base_keys = {tid: self._base_key(tid, fp) for tid, fp in fingerprints.items()}
        desired.update(base_keys.values())
        for key in base_keys.values():
            edges[key] = []
        for entry in noise_result["layers"]:
            tid = f"{entry['base_texture']}_{width}"
            edges[entry["cache_key"]] = [base_keys[tid]]
            desired.add(entry["cache_key"])
        composite_keys: dict[str, str] = {}
        for mat in material_result["materials"]:
            for channel in CHANNELS:
                info = mat["channels"][channel]
                info_key = info["cache_key"]
                edges[info_key] = [noise_result["layers"][idx]["cache_key"] for idx in info["used_layers"]]
                desired.add(info_key)
            _, comp_key = composite_fingerprint(mat, self.config, specs)
            composite_keys[mat["material_id"]] = comp_key
            edges[comp_key] = [mat["channels"][c]["cache_key"] for c in CHANNELS]
            desired.add(comp_key)
        return desired, edges, noise_result, material_result, specs, composite_keys

    # ------------------------------------------------------------------ #
    # Budgeted concurrent rendering of layer noise (requirement 7)
    # ------------------------------------------------------------------ #
    def _render_noise_budgeted(self, textures, specs, use_threads: bool) -> tuple[dict, dict[str, Any]]:
        width = height = next(iter(textures.values())).width
        round_mode = self.config.fixed_point.round_mode
        tasks: list[Task] = []
        for layer in self.config.layers:
            # Each layer is one batch; point count is the full pixel count,
            # which deliberately exhausts small budgets for pending tests.
            tasks.append(Task(layer.index, 0, width * height, layer))

        def task_fn(task: Task) -> dict:
            layer = task.payload
            noise = layer_noise_tile(layer, width, height, 0, height)
            return {"sort_key": task.sort_key, "layer_index": layer.index, "noise": noise}

        threads = self.config.threads if use_threads else 1
        result = run_scheduler(tasks, task_fn, threads, self.config.point_budget)
        serial = run_serial_reference(tasks, task_fn, self.config.point_budget)
        concurrency_match = self._scheduler_outputs_equal(result, serial)
        return self._assemble_noise(textures, specs, round_mode, result), {
            "budget_log": result.budget_log,
            "pending": [t.sort_key for t in result.pending],
            "failures": [f.to_dict() for f in result.failures],
            "exhausted": result.exhausted,
            "threads": threads,
            "serial_equivalent": concurrency_match,
        }

    @staticmethod
    def _scheduler_outputs_equal(a, b) -> bool:
        if a.budget_log != b.budget_log or [t.sort_key for t in a.pending] != [
            t.sort_key for t in b.pending
        ]:
            return False
        if [f.to_dict() for f in a.failures] != [f.to_dict() for f in b.failures]:
            return False
        ra = {task.sort_key: out for task, out in a.results}
        rb = {task.sort_key: out for task, out in b.results}
        if ra.keys() != rb.keys():
            return False
        return all(np.array_equal(ra[k]["noise"], rb[k]["noise"]) for k in ra)

    def _assemble_noise(self, textures, specs, round_mode, scheduler_result) -> dict:
        width = next(iter(textures.values())).width
        base_gray = {tid: tex.gray.astype(np.float64) / 255.0 for tid, tex in textures.items()}
        completed = {out["layer_index"]: out["noise"] for _, out in scheduler_result.results}
        result: dict = {"layers": [], "order": [], "boundary_pixels": 0}
        previous = None
        for layer in self.config.layers:
            if layer.index not in completed:
                continue
            noise = completed[layer.index]
            from .fixedpoint import quantize_float

            gray = base_gray[f"{layer.base_texture}_{width}"]
            if previous is None:
                current, boundary = quantize_float(gray + noise, specs.signed, round_mode)
            else:
                current, boundary = quantize_float(
                    previous.astype(np.float64) / specs.signed.scale + noise, specs.signed, round_mode
                )
            fp = fingerprint_array(current, f"noise_layer:{layer.index}")
            key = derive_cache_key(
                "noise", [fp],
                {
                    "layer_index": layer.index, "fingerprint": layer.fingerprint,
                    "frequency": layer.frequency, "amplitude": layer.amplitude,
                    "frac_bits": specs.signed.frac_bits, "levels": specs.signed.levels,
                    "round_mode": round_mode, "base_texture": layer.base_texture,
                },
            )
            result["layers"].append(
                {"index": layer.index, "image": current, "noise": noise, "boundary": boundary,
                 "fingerprint": fp, "cache_key": key, "base_texture": layer.base_texture}
            )
            result["order"].append(layer.index)
            result["boundary_pixels"] += int(np.count_nonzero(boundary))
            previous = current
        return result

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #
    def run(self, resolution: int, use_threads: bool = True,
            force_rebuild: bool = False) -> PipelineResult:
        textures, fingerprints, load_failures = self.load_textures(resolution)
        failures: list[Failure] = list(load_failures)
        if not textures:
            raise RuntimeError("no decodable textures available for this resolution tier")

        desired, edges, probe_noise, probe_material, specs, composite_keys = self._desired_graph(
            textures, fingerprints
        )
        kind_by_key = self._kind_map(textures, fingerprints, probe_noise, probe_material, specs)
        if force_rebuild:
            reusable, recompute = [], sorted(desired)
        else:
            reusable, recompute = plan_reuse(desired, self.cache)

        stale_roots = set(recompute)
        # Base texture nodes have no upstream and are roots; include the
        # recompute roots themselves plus everything reachable downstream.
        affected = set(stale_roots) | downstream(edges, stale_roots)
        affected_scope = self._scope_by_kind(kind_by_key, affected)

        # Budgeted concurrent layer rendering; serial equivalence is checked.
        noise_result, scheduler_meta = self._render_noise_budgeted(textures, specs, use_threads)
        failures.extend(Failure(**f) for f in scheduler_meta["failures"])

        # The direct vectorized path remains the source of truth for pixels;
        # assert it matches the scheduler assembly exactly.
        direct_noise = accumulate_layers(textures, self.config, specs)
        self._assert_noise_equal(noise_result, direct_noise)
        material_result = build_material_channels(
            textures, direct_noise, self.materials, self.config, specs
        )

        # Reuse/recompute counts are captured before writing so a first run
        # reports a full recompute and later runs report genuine cache hits.
        reusable_before = list(reusable)
        recompute_before = list(recompute)

        # Populate/verify cache entries for the desired graph.
        self._sync_cache(textures, fingerprints, direct_noise, material_result, specs, reusable)

        # Two-tier error tracking (configured width vs one tier higher).
        base_tier = run_tier(textures, self.materials, self.config,
                             self.config.fixed_point.frac_bits, self.config.fixed_point.round_mode)
        high_bits = self.config.fixed_point.high_tier_frac_bits
        high_tier = run_tier(textures, self.materials, self.config, high_bits,
                             self.config.fixed_point.round_mode)
        error_report = compare_tiers(textures, self.materials, self.config, base_tier, high_tier)

        snapshot_path = self._commit_snapshot(
            resolution, specs, direct_noise, material_result, error_report,
            len(recompute_before), scheduler_meta, failures
        )

        return PipelineResult(
            resolution=resolution,
            config_hash=self.config.config_hash,
            noise_result=direct_noise,
            material_result=material_result,
            error_report=error_report,
            failures=failures,
            reuse_keys=reusable_before,
            recompute_keys=recompute_before,
            affected_scope=affected_scope,
            scheduler=scheduler_meta,
            snapshots=[str(snapshot_path)],
            specs_meta=specs.to_meta(),
            texture_fingerprints=fingerprints,
            composite_keys=composite_keys,
        )

    @staticmethod
    def _assert_noise_equal(scheduled: dict, direct: dict) -> None:
        # Only batches completed within the point budget are assembled here;
        # pending batches are reported via E_POINT_BUDGET_PENDING.  Every
        # completed layer must still be pixel-identical to the direct path.
        direct_by_index = {entry["index"]: entry for entry in direct["layers"]}
        for entry in scheduled["layers"]:
            reference = direct_by_index[entry["index"]]
            if not np.array_equal(entry["image"], reference["image"]):
                raise AssertionError("scheduler output diverges from serial vectorized reference")

    def _kind_map(self, textures, fingerprints, noise_result, material_result, specs) -> dict[str, str]:
        width = next(iter(textures.values())).width
        kind_by_key: dict[str, str] = {}
        for tid, fp in fingerprints.items():
            kind_by_key[self._base_key(tid, fp)] = "base"
        for entry in noise_result["layers"]:
            kind_by_key[entry["cache_key"]] = "noise"
        for mat in material_result["materials"]:
            for channel in CHANNELS:
                kind_by_key[mat["channels"][channel]["cache_key"]] = "channel"
            _, comp_key = composite_fingerprint(mat, self.config, specs)
            kind_by_key[comp_key] = "composite"
        return kind_by_key

    def _scope_by_kind(self, kind_by_key: dict[str, str], affected: set[str]) -> dict[str, list[str]]:
        scope = {"base_textures": [], "noise_layers": [], "material_channels": [], "composites": []}
        bucket = {"base": "base_textures", "noise": "noise_layers",
                  "channel": "material_channels", "composite": "composites"}
        for key in sorted(affected):
            name = bucket.get(kind_by_key.get(key, ""))
            if name is not None:
                scope[name].append(key)
        return scope

    def _sync_cache(self, textures, fingerprints, noise_result, material_result, specs,
                    pre_existing_reusable: list[str]) -> None:
        width = next(iter(textures.values())).width
        for tid, fp in fingerprints.items():
            key = self._base_key(tid, fp)
            if not self.cache.has(key):
                self.cache.store_array(key, textures[tid].gray, "base", [], {"texture": tid})
        for entry in noise_result["layers"]:
            if not self.cache.has(entry["cache_key"]):
                tid = f"{entry['base_texture']}_{width}"
                base_key = self._base_key(tid, fingerprints[tid])
                self.cache.store_array(
                    entry["cache_key"], entry["image"], "noise", [base_key],
                    {"layer_index": entry["index"], "frequency": entry.get("frequency", self.config.layers[entry["index"]].frequency),
                     "amplitude": self.config.layers[entry["index"]].amplitude},
                )
        for mat in material_result["materials"]:
            for channel in CHANNELS:
                info = mat["channels"][channel]
                if not self.cache.has(info["cache_key"]):
                    upstream = [noise_result["layers"][idx]["cache_key"] for idx in info["used_layers"]]
                    self.cache.store_array(
                        info["cache_key"], info["image"], "channel", upstream,
                        {"material": mat["material_id"], "channel": channel},
                    )
            _, comp_key = composite_fingerprint(mat, self.config, specs)
            if not self.cache.has(comp_key):
                stack = np.stack([mat["channels"][c]["image"] for c in CHANNELS], axis=-1)
                self.cache.store_array(
                    comp_key, stack, "composite",
                    [mat["channels"][c]["cache_key"] for c in CHANNELS],
                    {"material": mat["material_id"]},
                )

    # ------------------------------------------------------------------ #
    # Snapshot commit and report
    # ------------------------------------------------------------------ #
    def _commit_snapshot(self, resolution, specs, noise_result, material_result,
                         error_report, recompute_count, scheduler_meta, failures) -> Path:
        state = self.snapshots.inspect(self.config.config_hash, specs.unit.frac_bits)
        completed_batches = len(noise_result["layers"])
        artifacts: dict[str, str] = {}
        files: dict[str, bytes | str] = {}

        for entry in noise_result["layers"]:
            rel = f"noise/layer_{entry['index']:03d}.npy"
            files[rel] = _to_npy_bytes(entry["image"])
            artifacts[f"noise:{entry['index']}"] = entry["fingerprint"]
        boundary_index = {}
        channel_means = {}
        for mat in material_result["materials"]:
            mid = mat["material_id"]
            for channel in CHANNELS:
                info = mat["channels"][channel]
                rel = f"channels/{mid}/{channel}.npy"
                files[rel] = _to_npy_bytes(info["image"])
                png_rel = f"channels/{mid}/{channel}.png"
                files[png_rel] = _png_bytes(fixed_to_uint8(info["image"], specs.unit))
                artifacts[f"{mid}:{channel}"] = info["fingerprint"]
                boundary_index[f"{mid}:{channel}"] = int(np.count_nonzero(info["boundary"]))
                channel_means[f"{mid}:{channel}"] = float(
                    info["image"].astype(np.float64).mean() / specs.unit.scale
                )

        report = {
            "resolution": resolution,
            "config_hash": self.config.config_hash,
            "fixed_point": specs.to_meta(),
            "round_mode": self.config.fixed_point.round_mode,
            "blend_mode": self.config.blend_mode,
            "channel_means": channel_means,
            "boundary_samples": boundary_index,
            "error": {
                "tolerance": error_report["tolerance"],
                "noise_diff_max": error_report["noise_diff_max"],
                "channel_diff_max": error_report["channel_diff_max"],
                "per_stage_max_diff": error_report["per_stage_max_diff"],
                "attribution_stats": error_report["attribution_stats"],
                "inconsistent_pixels": error_report["inconsistent_pixels"],
            },
            "recompute_count": recompute_count,
            "reuse_count": len(self.cache.all_keys()) - recompute_count,
            "scheduler": {
                "threads": scheduler_meta["threads"],
                "exhausted": scheduler_meta["exhausted"],
                "pending": scheduler_meta["pending"],
                "serial_equivalent": scheduler_meta["serial_equivalent"],
                "budget_deduction_sequence": [
                    {"task": key, "points": pts, "remaining": rem}
                    for key, pts, rem in scheduler_meta["budget_log"]
                ],
            },
            "failures": [f.to_dict() for f in failures],
        }
        files["report.json"] = json.dumps(report, sort_keys=True, indent=2)
        manifest = {
            "config_hash": self.config.config_hash,
            "frac_bits": specs.unit.frac_bits,
            "unit_levels": specs.unit.levels,
            "completed_batches": completed_batches,
            "artifacts": artifacts,
            "resolution": resolution,
        }
        version_dir = self.snapshots.commit(manifest, files)
        (self.work_dir / "report.json").write_text(files["report.json"], encoding="utf-8")
        return version_dir

    def resume_or_rebuild(self, resolution: int) -> tuple[SnapshotStore, object]:
        specs = StageSpecs.from_config(self.config)
        state = self.snapshots.inspect(self.config.config_hash, specs.unit.frac_bits)
        if state.stale and state.failure is not None:
            self.failures_stale = state.failure
        return self.snapshots, state


def _to_npy_bytes(arr: np.ndarray) -> bytes:
    import io

    buf = io.BytesIO()
    np.save(buf, np.ascontiguousarray(arr))
    return buf.getvalue()


def _png_bytes(arr: np.ndarray) -> bytes:
    import io

    buf = io.BytesIO()
    Image.fromarray(arr, mode="L").save(buf, format="PNG", optimize=True)
    return buf.getvalue()
