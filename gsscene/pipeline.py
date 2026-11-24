"""End-to-end deterministic multi-resolution Gaussian scene pipeline."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from . import codes
from .cache import ArtifactCache, composite_artifact_id, kernel_artifact_id, tile_artifact_id
from .config import SceneConfig, kernel_grid_size
from .diagnostics import DiagnosticHistory, analyze_level
from .kernels import (
    KernelPlan,
    compute_kernel_group,
    group_cache_key,
    plan_kernel_group,
    tiles_for_kernel,
)
from .models import CompositeRecord, Failure, KernelGroup, LoadedProject, StoredView, TileRecord, ViewRecord
from .rendering import build_composite_arrays, composite_cache_key, render_tile_arrays, tile_cache_key
from .sample import clone_sample
from .scheduling import ConcurrencyTracker, allocate_kernel_budgets, parallel_map
from .snapshots import SnapshotManager
from .state import EditError, SceneState, StateManager
from .util import array_fingerprint, atomic_write_json, hash_value, jsonable
from .validation import load_project


@dataclass
class GroupBuildTask:
    plan: KernelPlan
    previous_arrays: dict[str, np.ndarray] | None
    previous_keys: list[str] | None
    previous_level_fingerprint: str | None
    changed_indices: list[int]


@dataclass
class GroupBuildResult:
    group: KernelGroup
    computed_indices: list[int]
    reused_indices: list[int]


@dataclass
class TileBuildTask:
    view_index: int
    level: int
    row: int
    col: int
    cache_key: str
    refs: list[int]
    kernels: dict[str, np.ndarray]


class ScenePipeline:
    def __init__(
        self,
        sample_dir: Path,
        work_dir: Path,
        *,
        replay_edits: bool = True,
        force_rebuild: bool = False,
        injections_path: Path | None = None,
        worker_override: int | None = None,
    ) -> None:
        self.sample_dir = Path(sample_dir).resolve()
        self.work_dir = Path(work_dir).resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.replay_edits = bool(replay_edits)
        self.force_rebuild = bool(force_rebuild)
        self.injections_path = injections_path
        self.worker_override = worker_override
        self.state_manager = StateManager(self.work_dir)
        self.cache = ArtifactCache(self.work_dir)
        self.snapshots = SnapshotManager(self.work_dir)
        self.loaded = load_project(self.sample_dir, injections_path)
        self.failures: list[Failure] = list(self.loaded.failures)
        self.kernel_tracker = ConcurrencyTracker()
        self.render_tracker = ConcurrencyTracker()
        self.pending_queue: list[dict[str, Any]] = []
        self.budget_ledger: dict[str, Any] = {}

    def _effective_workers(self, config: SceneConfig) -> int:
        configured = max(1, int(config.concurrency))
        if self.worker_override is None:
            return configured
        return max(1, min(configured, int(self.worker_override)))

    def _prepare_state(self) -> tuple[SceneState | None, str, list[Failure]]:
        decision = "REBUILD"
        snapshot_failures: list[Failure] = []
        if self.loaded.config is None:
            return None, decision, snapshot_failures
        if self.state_manager.exists() and not self.force_rebuild:
            try:
                current_state = self.state_manager.load()
                decision, snapshot_failures = self.snapshots.resume_decision(current_state.config.family_fingerprint())
            except Exception as exc:
                snapshot_failures.append(
                    Failure(code=codes.E_SNAPSHOT_MANIFEST, message=f"cannot load state: {exc}", source="snapshot")
                )
                decision = "REBUILD"
            if decision == "RESUME":
                return current_state, decision, snapshot_failures
            self.state_manager.reset()
            self.cache.reset()
            self.snapshots.reset()
        elif self.force_rebuild:
            self.state_manager.reset()
            self.cache.reset()
            self.snapshots.reset()
        state = self.state_manager.initialize(self.loaded)
        return state, "REBUILD", snapshot_failures

    def _apply_pending_edits(self, state: SceneState, max_new_edits: int | None) -> tuple[list[Failure], list[dict[str, Any]]]:
        failures: list[Failure] = []
        attempted: list[dict[str, Any]] = []
        if not self.replay_edits:
            return failures, attempted
        applied = set(state.applied_edit_seqs)
        pending = [edit for edit in self.loaded.edits if int(edit["seq"]) not in applied]
        if max_new_edits is not None:
            pending = pending[: max(0, int(max_new_edits))]
        for edit in pending:
            attempted.append(edit)
            try:
                self.state_manager.apply_edit(state, edit)
            except EditError as exc:
                failures.append(
                    Failure(
                        code=exc.code,
                        message=f"edit seq {edit.get('seq')} ({edit.get('type')}): {exc.message}",
                        source="edit",
                        view_index=exc.view_index,
                        level=exc.level,
                    )
                )
            except Exception as exc:
                failures.append(
                    Failure(
                        code=codes.E_EDIT_JSON,
                        message=f"edit seq {edit.get('seq')} ({edit.get('type')}) failed: {exc}",
                        source="edit",
                    )
                )
        self.state_manager.save(state)
        return failures, attempted

    def _build_kernel_groups(
        self,
        state: SceneState,
        config: SceneConfig,
        records: dict[int, ViewRecord],
        workers: int,
    ) -> tuple[
        dict[tuple[int, int], KernelGroup],
        list[list[int]],
        list[list[int]],
        set[tuple[int, int]],
        dict[str, Any],
    ]:
        plans: list[KernelPlan] = []
        for view_index in sorted(records):
            for level in range(config.levels):
                if (view_index, level) in self.loaded.excluded_levels:
                    continue
                plans.append(plan_kernel_group(config, records[view_index], level, state.sh_mods))
        adjusted, budget_ledger = allocate_kernel_budgets(plans, config)
        self.budget_ledger = budget_ledger
        for entry in budget_ledger["kernel_scheduler"]["entries"]:
            if entry["status"] != "COMPLETED":
                self.failures.append(
                    Failure(
                        code=entry["status"],
                        message=(
                            f"{entry['status']} for view={entry['view_index']} level={entry['level']} "
                            f"requested={entry['requested']} allowed={entry['allowed_kernel']}"
                        ),
                        source="scheduler",
                        view_index=entry["view_index"],
                        level=entry["level"],
                    )
                )
        self.pending_queue.extend(
            {
                "code": item["code"],
                "count": item["count"],
                "kernel_index_end": item["kernel_index_end"],
                "kernel_index_start": item["kernel_index_start"],
                "level": item["level"],
                "stage": "kernel",
                "view_index": item["view_index"],
            }
            for item in budget_ledger.get("pending", [])
        )

        groups: dict[tuple[int, int], KernelGroup] = {}
        tasks: list[GroupBuildTask] = []
        recompute_kernels: list[list[int]] = []
        reuse_kernels: list[list[int]] = []
        changed_pairs: set[tuple[int, int]] = set()
        for plan in adjusted:
            pair = (plan.view_index, plan.level)
            old_meta, old_arrays = self.cache.load_kernel(plan.view_index, plan.level)
            reusable = (
                old_meta is not None
                and old_arrays is not None
                and old_meta.get("cache_key") == plan.cache_key
                and len(old_meta.get("kernel_keys", [])) == len(plan.kernel_keys)
                and old_meta.get("kernel_keys") == plan.kernel_keys
            )
            if reusable:
                group = KernelGroup(
                    view_index=plan.view_index,
                    level=plan.level,
                    grid_size=plan.grid_size,
                    cache_key=plan.cache_key,
                    kernel_keys=list(plan.kernel_keys),
                    kernel_content_fingerprints=list(old_meta["kernel_content_fingerprints"]),
                    arrays=old_arrays,
                    level_fingerprint=plan.level_fingerprint,
                    view_content_fingerprint=plan.view_content_fingerprint,
                    camera_fingerprint=plan.camera_fingerprint,
                    projection_matrix_fingerprint=str(old_meta["projection_matrix_fingerprint"]),
                )
                groups[pair] = group
                reuse_kernels.extend([[plan.view_index, plan.level, index] for index in range(len(plan.kernel_keys))])
                continue
            previous_keys = list(old_meta.get("kernel_keys", [])) if old_meta else None
            if old_arrays is not None and previous_keys is not None and len(previous_keys) == len(plan.kernel_keys):
                changed_indices = [
                    index
                    for index, key in enumerate(plan.kernel_keys)
                    if index >= len(previous_keys) or previous_keys[index] != key
                ]
            else:
                changed_indices = list(range(len(plan.kernel_keys)))
            tasks.append(
                GroupBuildTask(
                    plan=plan,
                    previous_arrays=old_arrays,
                    previous_keys=previous_keys,
                    previous_level_fingerprint=str(old_meta.get("level_fingerprint")) if old_meta else None,
                    changed_indices=changed_indices,
                )
            )
            if changed_indices:
                changed_pairs.add(pair)

        def build(task: GroupBuildTask) -> GroupBuildResult:
            group, computed, reused = compute_kernel_group(
                config,
                records[task.plan.view_index],
                task.plan.level,
                task.plan,
                state.sh_mods,
                previous_arrays=task.previous_arrays,
                previous_keys=task.previous_keys,
                previous_level_fingerprint=task.previous_level_fingerprint,
            )
            self.cache.write_kernel_group(group, task.plan.requested_count)
            return GroupBuildResult(group=group, computed_indices=computed, reused_indices=reused)

        results = parallel_map(build, tasks, workers, self.kernel_tracker)
        for result in sorted(results, key=lambda item: (item.group.view_index, item.group.level)):
            pair = (result.group.view_index, result.group.level)
            groups[pair] = result.group
            recompute_kernels.extend(
                [[result.group.view_index, result.group.level, index] for index in result.computed_indices]
            )
            reuse_kernels.extend(
                [[result.group.view_index, result.group.level, index] for index in result.reused_indices]
            )
        recompute_kernels.sort()
        reuse_kernels.sort()
        if not adjusted:
            self.budget_ledger = {
                "kernel_scheduler": {"entries": [], "max_concurrency": workers},
                "pending": [],
                "point_scheduler": {"entries": [], "max_concurrency": workers},
            }
        return groups, recompute_kernels, reuse_kernels, changed_pairs, self.budget_ledger

    def _build_tiles(
        self,
        config: SceneConfig,
        groups: dict[tuple[int, int], KernelGroup],
        workers: int,
    ) -> tuple[
        dict[tuple[int, int], dict[tuple[int, int], dict[str, np.ndarray]]],
        list[list[int]],
        list[list[int]],
        set[tuple[int, int]],
    ]:
        tile_rows = int(math.ceil(config.image_size[1] / config.tile_size))
        tile_cols = int(math.ceil(config.image_size[0] / config.tile_size))
        tile_arrays: dict[tuple[int, int], dict[tuple[int, int], dict[str, np.ndarray]]] = {}
        tasks: list[TileBuildTask] = []
        recompute_tiles: list[list[int]] = []
        reuse_tiles: list[list[int]] = []
        changed_pairs: set[tuple[int, int]] = set()
        for pair in sorted(groups):
            view_index, level = pair
            group = groups[pair]
            refs_by_tile: dict[tuple[int, int], list[int]] = {
                (row, col): [] for row in range(tile_rows) for col in range(tile_cols)
            }
            width = group.grid_size[0]
            for index in range(len(group.kernel_keys)):
                row = index // width
                col = index % width
                for tile in tiles_for_kernel(config, level, group.grid_size, row, col):
                    if tile in refs_by_tile:
                        refs_by_tile[tile].append(index)
            per_level: dict[tuple[int, int], dict[str, np.ndarray]] = {}
            for (row, col) in sorted(refs_by_tile):
                refs = sorted(refs_by_tile[(row, col)])
                keys = [group.kernel_keys[index] for index in refs]
                cache_key = tile_cache_key(config, view_index, level, row, col, refs, keys)
                old_meta, old_arrays = self.cache.load_tile(view_index, level, row, col)
                if old_meta is not None and old_arrays is not None and old_meta.get("cache_key") == cache_key:
                    per_level[(row, col)] = old_arrays
                    reuse_tiles.append([view_index, level, row, col])
                else:
                    tasks.append(
                        TileBuildTask(
                            view_index=view_index,
                            level=level,
                            row=row,
                            col=col,
                            cache_key=cache_key,
                            refs=refs,
                            kernels=group.arrays,
                        )
                    )
                    recompute_tiles.append([view_index, level, row, col])
                    changed_pairs.add(pair)
            tile_arrays[pair] = per_level

        def build(task: TileBuildTask) -> tuple[TileBuildTask, dict[str, np.ndarray]]:
            arrays = render_tile_arrays(
                config,
                task.kernels,
                task.refs,
                task.level,
                task.row,
                task.col,
            )
            record = TileRecord(
                view_index=task.view_index,
                level=task.level,
                row=task.row,
                col=task.col,
                cache_key=task.cache_key,
                kernel_refs=task.refs,
                arrays=arrays,
            )
            self.cache.write_tile(record)
            return task, arrays

        results = parallel_map(build, tasks, workers, self.render_tracker)
        for task, arrays in results:
            tile_arrays[(task.view_index, task.level)][(task.row, task.col)] = arrays
        recompute_tiles.sort()
        reuse_tiles.sort()
        return tile_arrays, recompute_tiles, reuse_tiles, changed_pairs

    def _build_composites(
        self,
        config: SceneConfig,
        groups: dict[tuple[int, int], KernelGroup],
        tile_arrays: dict[tuple[int, int], dict[tuple[int, int], dict[str, np.ndarray]]],
    ) -> tuple[
        dict[tuple[int, int], dict[str, np.ndarray]],
        list[list[int]],
        list[list[int]],
        set[tuple[int, int]],
    ]:
        composites: dict[tuple[int, int], dict[str, np.ndarray]] = {}
        recompute: list[list[int]] = []
        reuse: list[list[int]] = []
        changed_pairs: set[tuple[int, int]] = set()
        view_indices = sorted({view for view, _level in groups})
        for view_index in view_indices:
            lower_arrays: dict[str, np.ndarray] | None = None
            lower_key: str | None = None
            lower_tile_refs: list[tuple[int, int, int, int]] = []
            for level in range(config.levels):
                pair = (view_index, level)
                if pair not in groups or pair not in tile_arrays:
                    continue
                per_tile = tile_arrays[pair]
                tile_coords = sorted(per_tile)
                tile_keys: list[str] = []
                for row, col in tile_coords:
                    refs = self._tile_refs_from_arrays(groups[pair], config, row, col)
                    tile_keys.append(
                        tile_cache_key(
                            config,
                            view_index,
                            level,
                            row,
                            col,
                            refs,
                            [groups[pair].kernel_keys[index] for index in refs],
                        )
                    )
                expected_key = composite_cache_key(config, view_index, level, tile_keys, lower_key)
                old_meta, old_arrays = self.cache.load_composite(view_index, level)
                if old_meta is not None and old_arrays is not None and old_meta.get("cache_key") == expected_key:
                    arrays = old_arrays
                    reuse.append([view_index, level])
                else:
                    arrays = build_composite_arrays(config, level, per_tile, lower_arrays)
                    record = CompositeRecord(
                        view_index=view_index,
                        level=level,
                        cache_key=expected_key,
                        tile_refs=[(view_index, level, row, col) for row, col in tile_coords],
                        lower_tile_refs=list(lower_tile_refs),
                        lower_composite_key=lower_key,
                        arrays=arrays,
                    )
                    self.cache.write_composite(record)
                    recompute.append([view_index, level])
                    changed_pairs.add(pair)
                composites[pair] = arrays
                lower_arrays = arrays
                lower_key = expected_key
                lower_tile_refs.extend((view_index, level, row, col) for row, col in tile_coords)
        recompute.sort()
        reuse.sort()
        return composites, recompute, reuse, changed_pairs

    def _tile_refs_from_arrays(self, group: KernelGroup, config: SceneConfig, row: int, col: int) -> list[int]:
        refs: list[int] = []
        width = group.grid_size[0]
        for index in range(len(group.kernel_keys)):
            kernel_row = index // width
            kernel_col = index % width
            if (row, col) in tiles_for_kernel(config, group.level, group.grid_size, kernel_row, kernel_col):
                refs.append(index)
        return refs

    def _report_artifacts(self) -> dict[str, list[dict[str, Any]]]:
        return self.cache.artifact_meta_entries()

    def _make_action_report(
        self,
        recompute_kernels: list[list[int]],
        reuse_kernels: list[list[int]],
        recompute_tiles: list[list[int]],
        reuse_tiles: list[list[int]],
        recompute_composites: list[list[int]],
        reuse_composites: list[list[int]],
    ) -> dict[str, Any]:
        return {
            "recompute_composites": recompute_composites,
            "recompute_kernel_groups": [list(item) for item in sorted({(item[0], item[1]) for item in recompute_kernels})],
            "recompute_kernels": recompute_kernels,
            "recompute_tiles": recompute_tiles,
            "reuse_composites": reuse_composites,
            "reuse_kernel_groups": [list(item) for item in sorted({(item[0], item[1]) for item in reuse_kernels})],
            "reuse_kernels": reuse_kernels,
            "reuse_tiles": reuse_tiles,
            "reused": len(reuse_kernels) + len(reuse_tiles) + len(reuse_composites),
        }

    def _build_diagnostics(
        self,
        config: SceneConfig,
        groups: dict[tuple[int, int], KernelGroup],
        tile_arrays: dict[tuple[int, int], dict[tuple[int, int], dict[str, np.ndarray]]],
        composites: dict[tuple[int, int], dict[str, np.ndarray]],
        changed_kernel_pairs: set[tuple[int, int]],
        changed_composite_pairs: set[tuple[int, int]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        incomplete = {
            (entry["view_index"], entry["level"])
            for entry in self.budget_ledger.get("kernel_scheduler", {}).get("entries", [])
            if entry.get("status") != "COMPLETED"
        }
        diagnostics: list[dict[str, Any]] = []
        for pair in sorted(groups):
            if pair in incomplete or pair in self.loaded.excluded_levels:
                continue
            view_index, level = pair
            per_tile = tile_arrays.get(pair, {})
            if pair not in composites or not per_tile:
                continue
            summary, arrays = analyze_level(
                config,
                groups[pair].arrays,
                per_tile,
                composites[pair],
                view_index,
                level,
            )
            npz_path = self.cache.save_diagnostics(view_index, level, arrays, summary)
            summary["detail_npz"] = str(npz_path.relative_to(self.work_dir))
            diagnostics.append(summary)
        attributions = DiagnosticHistory(self.cache).analyze_attributions(
            diagnostics,
            config,
            changed_kernel_pairs,
            changed_composite_pairs,
        )
        return diagnostics, attributions

    def _snapshot_cycle(
        self,
        state: SceneState,
        records: dict[int, ViewRecord],
        artifacts: dict[str, list[dict[str, Any]]],
        diagnostics: list[dict[str, Any]],
        actions: dict[str, Any],
        edit_step: int,
        completed_levels: list[int],
    ) -> dict[str, Any]:
        camera_hash = hash_value({str(index): records[index].camera.as_dict() for index in sorted(records)})
        payloads = {
            "load": {
                "active_views": sorted(records),
                "failure_count": len(self.failures),
                "injection_hits": self.loaded.injection_hits,
            },
            "kernel": {
                "artifact_count": len(artifacts.get("kernels", [])),
                "budget_ledger": self.budget_ledger,
                "pending": self.pending_queue,
            },
            "render": {"artifact_count": len(artifacts.get("tiles", []))},
            "composite": {
                "artifact_count": len(artifacts.get("composites", [])),
                "diagnostics_count": len(diagnostics),
            },
        }
        for stage, payload in payloads.items():
            payload["invalidation"] = actions
            payload["snapshot_interval"] = state.config.snapshot_interval
        return self.snapshots.commit_cycle(
            config_hash=state.config.fingerprint(),
            config_family_hash=state.config.family_fingerprint(),
            camera_hash=camera_hash,
            frac_bits=state.config.frac_bits,
            quant_gear=state.config.quant_gear,
            completed_levels=completed_levels,
            completed_views=sorted(records),
            edit_step=edit_step,
            stage_payloads=payloads,
        )

    def _empty_report(self, resume_mode: str) -> dict[str, Any]:
        return {
            "active_views": [],
            "actual_invalidation": {},
            "artifacts": {"composites": [], "kernels": [], "tiles": []},
            "attributions": [],
            "budget_concurrency": {"configured": None, "kernel": 0, "render": 0},
            "budget_ledger": {"kernel_scheduler": {"entries": []}, "pending": [], "point_scheduler": {"entries": []}},
            "camera_hash": None,
            "config_fingerprint": None,
            "diagnostics": [],
            "failures": [failure.as_dict() for failure in sorted(self._all_failures(), key=lambda item: item.sort_key())],
            "injection_hits": self.loaded.injection_hits,
            "invalidation": {},
            "key_manifest": {"composites": {}, "kernels": {}, "tiles": {}},
            "resume_mode": resume_mode,
            "snapshot_manifest": self.snapshots.load_manifest() or {},
        }

    def _all_failures(self) -> list[Failure]:
        failures = list(self.failures) + list(self.cache.errors)
        deduped: dict[tuple[Any, ...], Failure] = {}
        for failure in failures:
            key = (
                failure.code,
                failure.source,
                failure.view_index,
                failure.level,
                failure.injection_id,
                failure.message,
            )
            deduped[key] = failure
        return sorted(deduped.values(), key=lambda item: item.sort_key())

    def run(self, max_new_edits: int | None = None) -> dict[str, Any]:
        state, resume_mode, snapshot_failures = self._prepare_state()
        self.failures.extend(snapshot_failures)
        if state is None or self.loaded.config is None:
            report = self._empty_report(resume_mode)
            atomic_write_json(self.work_dir / "report.json", report)
            return report
        edit_failures, attempted_edits = self._apply_pending_edits(state, max_new_edits)
        self.failures.extend(edit_failures)
        records = state.records(self.work_dir)
        config = state.config
        workers = self._effective_workers(config)
        self.pending_queue = []

        groups, recompute_kernels, reuse_kernels, changed_kernel_pairs, budget_ledger = self._build_kernel_groups(
            state, config, records, workers
        )
        tile_arrays, recompute_tiles, reuse_tiles, _tile_changed_pairs = self._build_tiles(config, groups, workers)
        composites, recompute_composites, reuse_composites, changed_composite_pairs = self._build_composites(
            config, groups, tile_arrays
        )
        actions = self._make_action_report(
            recompute_kernels,
            reuse_kernels,
            recompute_tiles,
            reuse_tiles,
            recompute_composites,
            reuse_composites,
        )

        expected = {
            "kernel": {kernel_artifact_id(view, level) for view, level in groups},
            "tile": {
                tile_artifact_id(view, level, row, col)
                for (view, level), per_tile in tile_arrays.items()
                for row, col in per_tile
            },
            "composite": {composite_artifact_id(view, level) for view, level in composites},
        }
        removed = self.cache.cleanup(expected)
        diagnostics, attributions = self._build_diagnostics(
            config,
            groups,
            tile_arrays,
            composites,
            changed_kernel_pairs,
            changed_composite_pairs,
        )
        artifacts = self._report_artifacts()
        key_manifest = self.cache.key_manifest()
        self.cache.write_index(artifacts)
        edit_step = len(state.applied_edits)
        interval = int(config.snapshot_interval)
        snapshot_due = interval == 0 or edit_step == 0 or edit_step % interval == 0 or not attempted_edits
        incomplete_levels = {
            entry["level"]
            for entry in self.budget_ledger.get("kernel_scheduler", {}).get("entries", [])
            if entry.get("status") != "COMPLETED"
        }
        completed_levels = [level for level in range(config.levels) if level not in incomplete_levels]
        if snapshot_due:
            snapshot_manifest = self._snapshot_cycle(
                state,
                records,
                artifacts,
                diagnostics,
                actions,
                edit_step,
                completed_levels,
            )
        else:
            snapshot_manifest = self.snapshots.load_manifest() or {}
        camera_hash = hash_value({str(index): records[index].camera.as_dict() for index in sorted(records)})
        report = {
            "active_views": sorted(records),
            "actual_invalidation": actions,
            "applied_edits": state.applied_edits,
            "artifacts": artifacts,
            "attributions": attributions,
            "budget_concurrency": {
                "configured": config.concurrency,
                "effective": workers,
                "kernel": self.kernel_tracker.as_dict()["max_in_flight"],
                "render": self.render_tracker.as_dict()["max_in_flight"],
            },
            "budget_ledger": budget_ledger,
            "camera_hash": camera_hash,
            "config_fingerprint": config.fingerprint(),
            "config_family_hash": config.family_fingerprint(),
            "diagnostics": diagnostics,
            "excluded_levels": [[view, level] for view, level in sorted(self.loaded.excluded_levels)],
            "excluded_views": sorted(self.loaded.excluded_views),
            "failures": [failure.as_dict() for failure in self._all_failures()],
            "injection_hits": self.loaded.injection_hits,
            "invalidation": actions,
            "key_manifest": key_manifest,
            "pending": self.pending_queue,
            "removed_artifacts": removed,
            "resume_mode": resume_mode,
            "snapshot_committed": snapshot_due,
            "snapshot_manifest": snapshot_manifest,
            "content_manifest_digest": hash_value({"artifacts": artifacts, "keys": key_manifest}),
        }
        atomic_write_json(self.work_dir / "report.json", report)
        return report

    def apply_next_edit(self) -> dict[str, Any]:
        return self.run(max_new_edits=1)


class ExperimentRunner:
    """Convenience wrapper used by tests and scripts."""

    def __init__(self, sample_dir: Path, work_dir: Path, **kwargs: Any) -> None:
        self.pipeline = ScenePipeline(sample_dir, work_dir, **kwargs)

    def baseline(self) -> dict[str, Any]:
        return self.pipeline.run(max_new_edits=0)

    def step(self) -> dict[str, Any]:
        return self.pipeline.run(max_new_edits=1)

    def run_all(self) -> dict[str, Any]:
        return self.pipeline.run()


def compact_report(report: dict[str, Any]) -> dict[str, Any]:
    """A small report that omits large ref/key lists but preserves decisions."""

    result = dict(report)
    for key in ("invalidation", "actual_invalidation"):
        value = result.get(key)
        if isinstance(value, dict):
            result[key] = {
                name: (len(items) if isinstance(items, list) else items)
                for name, items in value.items()
            }
    result["key_manifest_digest"] = hash_value(report.get("key_manifest", {}))
    result.pop("key_manifest", None)
    return result
