"""Incremental segmentation engine with deterministic scheduling and snapshots."""

from __future__ import annotations

import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .classify import TileResult, classify_tile
from .errors import (
    E_FUSION_DEPENDENCY_FAILED,
    E_INTERNAL,
    E_RECOMPUTE_BUDGET_PENDING,
    E_SNAPSHOT_STALE,
    Failure,
    SegmentationError,
)
from .fusion import (
    GlobalResult,
    RegionResult,
    compute_region,
    fuse_global,
    iter_adjacent_pairs,
    pair_id,
)
from .inputs import (
    LoadedTile,
    TileSpec,
    load_config,
    load_index,
    load_registry,
    load_tiles,
)
from .snapshot import PendingTask, RuntimeState, SnapshotStore
from .util import atomic_write_json, canonical_hash, read_json


@dataclass(slots=True)
class RunOptions:
    index_path: Path
    config_path: Path
    fusion_versions_path: Path
    state_dir: Path
    report_path: Path | None = None
    full_rebuild: bool = False
    strict_recovery: bool = False
    injection_manifest_path: Path | None = None
    stop_event: threading.Event = field(default_factory=threading.Event)
    max_batches: int = 1


@dataclass(slots=True)
class _DesiredRegion:
    pair_id: str
    spec_a: TileSpec
    spec_b: TileSpec
    bounds: tuple[int, int, int, int]
    fingerprint: str
    stable_sort_key: str
    cache_key_a: str
    cache_key_b: str


def _region_fingerprint_from_keys(
    tile_a_id: str,
    cache_key_a: str,
    tile_b_id: str,
    cache_key_b: str,
    bounds: tuple[int, int, int, int],
    strategy_version: str,
    weight_power: float,
    scale: float,
    tie_break: str,
) -> str:
    return canonical_hash(
        {
            "tiles": {tile_a_id: cache_key_a, tile_b_id: cache_key_b},
            "overlap_bounds": list(bounds),
            "fusion_version": strategy_version,
            "weight_power": weight_power,
            "scale": scale,
            "tie_break": tie_break,
        },
        prefix="fusion-region-v1|",
    )


class SegmentationEngine:
    def __init__(self, options: RunOptions) -> None:
        self.options = options
        self.report_path = options.report_path or (options.state_dir / "report.json")

    def _load_injection_manifest(self, index_root: Path) -> dict[str, Any]:
        path = self.options.injection_manifest_path
        if path is None:
            candidate = index_root / "injection_manifest.json"
            if candidate.exists():
                path = candidate
        if path is None or not path.exists():
            return {"items": [], "preflight_failures": []}
        try:
            payload = read_json(path)
        except (OSError, ValueError):
            return {"items": [], "preflight_failures": []}
        if not isinstance(payload, dict):
            return {"items": [], "preflight_failures": []}
        payload.setdefault("items", [])
        payload.setdefault("preflight_failures", [])
        return payload

    @staticmethod
    def _annotate_failure(failure: Failure, injection: dict[str, Any]) -> Failure:
        code_to_type = {
            "E_TILE_DECODE": "truncate_png",
            "E_TILE_SHAPE_MISMATCH": "band_count_mismatch",
            "E_BAND_VERSION_UNKNOWN": "unknown_band_version",
            "E_FUSION_VERSION_UNKNOWN": "unknown_fusion_version",
            "E_PATCH_BOUNDS": "patch_bounds",
        }
        expected_type = code_to_type.get(failure.code)
        for item in injection.get("items", []):
            if item.get("type") != expected_type:
                continue
            if item.get("tile_id") not in (None, failure.tile_id):
                continue
            failure.injection_id = item.get("id")
            return failure
        return failure

    def _injection_hits(
        self, injection: dict[str, Any], failures: list[Failure]
    ) -> list[dict[str, Any]]:
        failures_by_id = {
            failure.injection_id: failure for failure in failures if failure.injection_id is not None
        }
        hits = []
        for item in injection.get("items", []):
            item_id = item.get("id")
            failure = failures_by_id.get(item_id)
            hit = {
                "id": item_id,
                "type": item.get("type"),
                "tile_id": item.get("tile_id"),
                "hit": failure is not None,
            }
            if failure is not None:
                hit["code"] = failure.code
                hit["stage"] = failure.stage
            hits.append(hit)
        return hits

    def _fail_report(
        self,
        *,
        code: str,
        failure: Failure,
        injection: dict[str, Any],
        recovery: dict[str, Any] | None = None,
        snapshot_code: str | None = None,
    ) -> dict[str, Any]:
        failures = [failure]
        for item in injection.get("preflight_failures", []):
            preflight = Failure(
                code=item.get("code", E_INTERNAL),
                stage=item.get("stage", "preflight"),
                message=item.get("message", "injected preflight failure"),
                row=item.get("row"),
                col=item.get("col"),
                tile_id=item.get("tile_id"),
                injection_id=item.get("injection_id"),
                details=item.get("details", {}),
            )
            failures.append(self._annotate_failure(preflight, injection))
        failure = self._annotate_failure(failure, injection)
        codes = [code]
        if snapshot_code:
            codes.append(snapshot_code)
        report = {
            "status": "failed",
            "codes": codes,
            "failures": [item.as_dict() for item in failures],
            "injection_hits": self._injection_hits(injection, failures),
            "recovery": recovery,
            "version_convergence": [],
            "class_grid_summary": None,
            "tile_order": [],
            "fusion_region_order": [],
            "budget_deductions": [],
            "reuse_recompute_ledger": {},
            "execution_ledger": [],
        }
        atomic_write_json(self.report_path, report)
        return report

    def run(self) -> dict[str, Any]:
        options = self.options
        store = SnapshotStore(options.state_dir)
        snapshot_code: str | None = None
        recovery_point: dict[str, Any] | None = None
        previous: RuntimeState | None = None
        stale_failure: Failure | None = None

        injection = self._load_injection_manifest(options.index_path.parent)
        try:
            config = load_config(options.config_path)
            registry = load_registry(options.fusion_versions_path)
            strategy = registry.require(config.fusion_version)
            index = load_index(options.index_path)
        except SegmentationError as exc:
            failure = self._annotate_failure(exc.failure(), injection)
            return self._fail_report(
                code=failure.code,
                failure=failure,
                injection=injection,
                recovery=recovery_point,
                snapshot_code=snapshot_code,
            )

        if not options.full_rebuild:
            try:
                expected_config = config.hash if options.strict_recovery else None
                expected_index = index.index_hash if options.strict_recovery else None
                previous = store.load(
                    expected_config_hash=expected_config,
                    expected_index_hash=expected_index,
                )
                if previous is not None:
                    recovery_point = previous.recovery_point
            except SegmentationError as exc:
                if exc.code != E_SNAPSHOT_STALE:
                    raise
                stale_failure = exc.failure()
                snapshot_code = E_SNAPSHOT_STALE
                previous = None

        if config.window_size > index.tile_size[0] or config.window_size > index.tile_size[1]:
            failure = Failure(
                code="E_CONFIG_INVALID",
                stage="classify",
                message="window_size exceeds configured tile size",
            )
            return self._fail_report(
                code=failure.code,
                failure=failure,
                injection=injection,
                recovery=recovery_point,
                snapshot_code=snapshot_code,
            )

        load_result = load_tiles(index, config, registry)
        failures = list(load_result.failures)
        for failure in failures:
            self._annotate_failure(failure, injection)
        for item in injection.get("preflight_failures", []):
            failures.append(
                self._annotate_failure(
                    Failure(
                        code=item.get("code", E_INTERNAL),
                        stage=item.get("stage", "preflight"),
                        message=item.get("message", "injected preflight failure"),
                        row=item.get("row"),
                        col=item.get("col"),
                        tile_id=item.get("tile_id"),
                        injection_id=item.get("injection_id"),
                        details=item.get("details", {}),
                    ),
                    injection,
                )
            )

        loaded = load_result.loaded
        old_tiles = previous.tiles if previous else {}
        old_regions = previous.regions if previous else {}
        state_tiles: dict[str, TileResult] = {}
        dirty_tiles: dict[str, LoadedTile] = {}
        reused_tile_ids: list[str] = []
        for tile_id, tile_input in sorted(
            loaded.items(), key=lambda item: item[1].spec.order_key
        ):
            old = old_tiles.get(tile_id)
            if (
                old is not None
                and old.cache_key == tile_input.cache_key
                and old.content_fingerprint == tile_input.content_fingerprint
                and old.classes.shape == (tile_input.spec.height, tile_input.spec.width)
            ):
                state_tiles[tile_id] = old
                reused_tile_ids.append(tile_id)
            else:
                dirty_tiles[tile_id] = tile_input

        desired_regions: dict[str, _DesiredRegion] = {}
        if loaded:
            for spec_a, spec_b, bounds in iter_adjacent_pairs(index):
                if spec_a.tile_id not in loaded or spec_b.tile_id not in loaded:
                    continue
                tile_input_a = loaded[spec_a.tile_id]
                tile_input_b = loaded[spec_b.tile_id]
                fingerprint = _region_fingerprint_from_keys(
                    spec_a.tile_id,
                    tile_input_a.cache_key,
                    spec_b.tile_id,
                    tile_input_b.cache_key,
                    bounds,
                    strategy.version,
                    strategy.weight_power,
                    strategy.scale,
                    strategy.tie_break,
                )
                pair_key = pair_id(spec_a, spec_b)
                stable = (
                    f"region:{spec_a.row:08d}:{spec_a.col:08d}:"
                    f"{spec_b.row:08d}:{spec_b.col:08d}:{pair_key}"
                )
                desired_regions[pair_key] = _DesiredRegion(
                    pair_id=pair_key,
                    spec_a=spec_a,
                    spec_b=spec_b,
                    bounds=bounds,
                    fingerprint=fingerprint,
                    stable_sort_key=stable,
                    cache_key_a=tile_input_a.cache_key,
                    cache_key_b=tile_input_b.cache_key,
                )
        state_regions: dict[str, RegionResult] = {}
        dirty_regions: dict[str, _DesiredRegion] = {}
        reused_region_ids: list[str] = []
        for pair_key, desired in sorted(
            desired_regions.items(), key=lambda item: item[1].stable_sort_key
        ):
            old = old_regions.get(pair_key)
            old_tile_a = old_tiles.get(desired.spec_a.tile_id)
            old_tile_b = old_tiles.get(desired.spec_b.tile_id)
            can_reuse = (
                old is not None
                and old.fingerprint == desired.fingerprint
                and old_tile_a is not None
                and old_tile_b is not None
                and old_tile_a.cache_key == desired.cache_key_a
                and old_tile_b.cache_key == desired.cache_key_b
            )
            if can_reuse:
                state_regions[pair_key] = old
                reused_region_ids.append(pair_key)
            else:
                dirty_regions[pair_key] = desired

        old_pending = previous.pending if previous else []
        pending_filtered: list[PendingTask] = []
        queued_keys: set[tuple[str, str]] = set()
        for task in old_pending:
            if task.kind == "tile" and task.key in dirty_tiles:
                pending_filtered.append(task)
                queued_keys.add((task.kind, task.key))
            elif task.kind == "region" and task.key in dirty_regions:
                pending_filtered.append(task)
                queued_keys.add((task.kind, task.key))
        new_tasks: list[PendingTask] = []
        for tile_id, tile_input in sorted(dirty_tiles.items(), key=lambda item: item[1].spec.order_key):
            if ("tile", tile_id) not in queued_keys:
                new_tasks.append(
                    PendingTask(
                        kind="tile",
                        key=tile_id,
                        stable_sort_key=tile_input.spec.task_sort_key,
                    )
                )
        for pair_key, desired in sorted(
            dirty_regions.items(), key=lambda item: item[1].stable_sort_key
        ):
            if ("region", pair_key) not in queued_keys:
                new_tasks.append(
                    PendingTask(
                        kind="region",
                        key=pair_key,
                        stable_sort_key=desired.stable_sort_key,
                    )
                )
        ordered_tasks = pending_filtered + new_tasks

        budget = config.recompute_budget
        selected_tasks = ordered_tasks[:budget]
        remaining_tasks = ordered_tasks[budget:]
        budget_deductions = [
            {
                "sequence": index + 1,
                "kind": task.kind,
                "key": task.key,
                "stable_sort_key": task.stable_sort_key,
            }
            for index, task in enumerate(selected_tasks)
        ]
        execution_ledger: list[dict[str, Any]] = []
        for sequence, task in enumerate(selected_tasks, start=1):
            execution_ledger.append(
                {
                    "sequence": sequence,
                    "kind": task.kind,
                    "key": task.key,
                    "stable_sort_key": task.stable_sort_key,
                    "action": "recompute",
                }
            )

        tile_tasks = [task for task in selected_tasks if task.kind == "tile"]
        region_tasks = [task for task in selected_tasks if task.kind == "region"]

        def run_tile(tile_id: str) -> tuple[str, TileResult | None, Failure | None]:
            try:
                result = classify_tile(dirty_tiles[tile_id], config)
                return tile_id, result, None
            except SegmentationError as exc:
                failure = exc.failure()
                spec = dirty_tiles[tile_id].spec
                failure.row = spec.row
                failure.col = spec.col
                failure.tile_id = spec.tile_id
                return tile_id, None, self._annotate_failure(failure, injection)
            except Exception as exc:  # isolation boundary for injected/unknown tile faults
                spec = dirty_tiles[tile_id].spec
                failure = Failure(
                    code=E_INTERNAL,
                    stage="classify",
                    message=f"{type(exc).__name__}: {exc}",
                    row=spec.row,
                    col=spec.col,
                    tile_id=spec.tile_id,
                )
                return tile_id, None, self._annotate_failure(failure, injection)

        processed_selected_keys: set[tuple[str, str]] = set()
        tile_outputs: dict[str, tuple[TileResult | None, Failure | None]] = {}
        if tile_tasks and not options.stop_event.is_set():
            with ThreadPoolExecutor(max_workers=config.max_workers, thread_name_prefix="tile") as executor:
                next_index = 0
                in_flight: dict[Any, PendingTask] = {}
                while True:
                    while (
                        next_index < len(tile_tasks)
                        and len(in_flight) < config.max_workers
                        and not options.stop_event.is_set()
                    ):
                        task = tile_tasks[next_index]
                        next_index += 1
                        future = executor.submit(run_tile, task.key)
                        in_flight[future] = task
                    if not in_flight:
                        break
                    done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                    for future in done:
                        task = in_flight.pop(future)
                        tile_id, result, failure = future.result()
                        tile_outputs[tile_id] = (result, failure)
                        processed_selected_keys.add((task.kind, task.key))
                    if options.stop_event.is_set() and not in_flight:
                        break
                    if next_index >= len(tile_tasks) and not in_flight:
                        break
        for task in tile_tasks:
            outcome = tile_outputs.get(task.key)
            if outcome is None:
                continue
            result, failure = outcome
            if result is not None:
                state_tiles[task.key] = result
            elif failure is not None:
                failures.append(failure)
                state_tiles.pop(task.key, None)

        stopped = options.stop_event.is_set()
        if region_tasks and not stopped:
            for task in region_tasks:
                if options.stop_event.is_set():
                    break
                desired = dirty_regions[task.key]
                tile_a = state_tiles.get(desired.spec_a.tile_id)
                tile_b = state_tiles.get(desired.spec_b.tile_id)
                if tile_a is None or tile_b is None:
                    failures.append(
                        Failure(
                            code=E_FUSION_DEPENDENCY_FAILED,
                            stage="fusion",
                            message="fusion region dependency is missing or failed",
                            row=desired.spec_a.row,
                            col=desired.spec_a.col,
                            tile_id=desired.spec_a.tile_id,
                            details={"pair_id": task.key},
                        )
                    )
                    state_regions.pop(task.key, None)
                    continue
                try:
                    region = compute_region(
                        tile_a,
                        desired.spec_a,
                        tile_b,
                        desired.spec_b,
                        desired.bounds,
                        strategy,
                    )
                    if region.fingerprint != desired.fingerprint:
                        raise RuntimeError("computed region fingerprint diverged from scheduled fingerprint")
                    state_regions[task.key] = region
                    processed_selected_keys.add((task.kind, task.key))
                except Exception as exc:
                    failures.append(
                        Failure(
                            code=E_INTERNAL,
                            stage="fusion",
                            message=f"{type(exc).__name__}: {exc}",
                            row=desired.spec_a.row,
                            col=desired.spec_a.col,
                            tile_id=desired.spec_a.tile_id,
                            details={"pair_id": task.key},
                        )
                    )
                    state_regions.pop(task.key, None)
                    processed_selected_keys.add((task.kind, task.key))

        stopped = stopped or options.stop_event.is_set()
        if stopped and (tile_tasks or region_tasks):
            unprocessed = [
                task
                for task in selected_tasks
                if (task.kind, task.key) not in processed_selected_keys
            ]
            remaining_tasks = unprocessed + remaining_tasks
            execution_ledger = [
                {
                    **entry,
                    "action": "recompute"
                    if (entry["kind"], entry["key"]) in processed_selected_keys
                    else "deferred_by_signal",
                }
                for entry in execution_ledger
            ]

        pending = remaining_tasks
        status = "complete"
        codes: list[str] = []
        if stopped:
            status = "graceful_stop"
        elif pending:
            status = "budget_pending"
            codes.append(E_RECOMPUTE_BUDGET_PENDING)
        if snapshot_code:
            codes.append(snapshot_code)

        global_result: GlobalResult | None = None
        if not pending and not stopped:
            global_result = fuse_global(state_tiles, index, config, strategy)

        state = RuntimeState(
            tiles=state_tiles,
            regions=state_regions,
            global_result=global_result,
            pending=pending,
            failures=failures,
        )
        recovery_before_run = previous.recovery_point if previous is not None else None
        recovery_point = store.commit(
            state,
            config_hash=config.hash,
            index_hash=index.index_hash,
            index_version=index.index_version,
            fusion_version=strategy.version,
            postprocess_fingerprint=config.postprocess_fingerprint,
            band_order_versions=sorted({tile.band_order_version for tile in state_tiles.values()}),
        )

        final_tile_ids = sorted(state_tiles, key=lambda key: state_tiles[key].order_key)
        final_region_ids = sorted(
            state_regions,
            key=lambda key: desired_regions[key].stable_sort_key,
        )
        reuse_ledger: dict[str, Any] = {}
        for tile_id in final_tile_ids:
            tile = state_tiles[tile_id]
            reuse_ledger[f"tile:{tile_id}"] = {
                "kind": "tile",
                "cache_key": tile.cache_key,
                "result_version": tile.result_version,
                "state": "valid",
            }
        for pair_key in final_region_ids:
            region = state_regions[pair_key]
            reuse_ledger[f"region:{pair_key}"] = {
                "kind": "region",
                "fingerprint": region.fingerprint,
                "region_version": region.region_version,
                "fusion_version": region.fusion_version,
                "state": "valid",
            }
        for pending_task in pending:
            reuse_ledger[f"{pending_task.kind}:{pending_task.key}"] = {
                "kind": pending_task.kind,
                "stable_sort_key": pending_task.stable_sort_key,
                "state": "pending",
            }
        report = {
            "status": status,
            "codes": codes,
            "inputs": {
                "index_path": str(options.index_path),
                "index_version": index.index_version,
                "index_hash": index.index_hash,
                "config_path": str(options.config_path),
                "config_hash": config.hash,
                "fusion_version": strategy.version,
                "band_order_versions": sorted({tile.band_order_version for tile in state_tiles.values()}),
                "max_workers": config.max_workers,
                "recompute_budget": config.recompute_budget,
            },
            "recovery": recovery_point,
            "recovery_before_run": recovery_before_run,
            "snapshot_stale": stale_failure.as_dict() if stale_failure is not None else None,
            "version_convergence": [
                {
                    "global_version": recovery_point.get("global_version") if recovery_point else None,
                    "config_hash": config.hash,
                    "index_version": index.index_version,
                    "fusion_version": strategy.version,
                    "recomputed_tiles": sorted(
                        [task.key for task in selected_tasks if task.kind == "tile"]
                    ),
                    "recomputed_regions": sorted(
                        [task.key for task in selected_tasks if task.kind == "region"]
                    ),
                    "reused_tiles": sorted(reused_tile_ids),
                    "reused_regions": sorted(reused_region_ids),
                    "pending": [task.as_dict() for task in pending],
                }
            ],
            "class_grid_summary": global_result.summary if global_result else None,
            "tile_order": final_tile_ids,
            "fusion_region_order": final_region_ids,
            "budget_deductions": budget_deductions,
            "recomputed_tiles": sorted(
                [task.key for task in selected_tasks if task.kind == "tile"]
            ),
            "recomputed_regions": sorted(
                [task.key for task in selected_tasks if task.kind == "region"]
            ),
            "failures": [failure.as_dict() for failure in failures],
            "injection_hits": self._injection_hits(injection, failures),
            "reuse_recompute_ledger": reuse_ledger,
            "execution_ledger": execution_ledger,
            "snapshot": {
                "generation": recovery_point.get("generation") if recovery_point else None,
                "global_version": recovery_point.get("global_version") if recovery_point else None,
                "fusion_version": recovery_point.get("fusion_version") if recovery_point else None,
            },
        }
        atomic_write_json(self.report_path, report)
        return report


def run_until_complete(options: RunOptions) -> dict[str, Any]:
    """Run at most ``max_batches`` budget batches, returning the final report."""

    engine = SegmentationEngine(options)
    reports: list[dict[str, Any]] = []
    for _ in range(options.max_batches):
        report = engine.run()
        reports.append(report)
        if report.get("status") not in {"budget_pending"}:
            break
        if options.stop_event.is_set():
            break
    if not reports:
        raise RuntimeError("no run was attempted")
    if len(reports) == 1:
        return reports[0]

    final = reports[-1]
    combined = dict(final)
    combined["batch_count"] = len(reports)
    combined["recovery_before_run"] = reports[0].get("recovery_before_run")
    combined["snapshot_stale"] = next(
        (report.get("snapshot_stale") for report in reports if report.get("snapshot_stale")),
        None,
    )
    codes = list(final.get("codes", []))
    if any(report.get("snapshot_stale") for report in reports) and "E_SNAPSHOT_STALE" not in codes:
        codes.append("E_SNAPSHOT_STALE")
    combined["codes"] = codes
    convergence: list[dict[str, Any]] = []
    deductions: list[dict[str, Any]] = []
    execution: list[dict[str, Any]] = []
    for batch_number, report in enumerate(reports, start=1):
        for entry in report.get("version_convergence", []):
            convergence.append({**entry, "batch": batch_number})
        for entry in report.get("budget_deductions", []):
            deductions.append({**entry, "batch": batch_number})
        for entry in report.get("execution_ledger", []):
            execution.append({**entry, "batch": batch_number})
    for sequence, entry in enumerate(deductions, start=1):
        entry["sequence"] = sequence
    for sequence, entry in enumerate(execution, start=1):
        entry["sequence"] = sequence
    combined["version_convergence"] = convergence
    combined["budget_deductions"] = deductions
    combined["execution_ledger"] = execution
    combined["batches"] = [
        {
            "batch": index,
            "status": report.get("status"),
            "snapshot": report.get("snapshot"),
            "codes": report.get("codes", []),
            "class_grid_summary": report.get("class_grid_summary"),
        }
        for index, report in enumerate(reports, start=1)
    ]
    combined["recomputed_tiles"] = sorted(
        {
            key
            for entry in convergence
            for key in entry.get("recomputed_tiles", [])
        }
    )
    combined["recomputed_regions"] = sorted(
        {
            key
            for entry in convergence
            for key in entry.get("recomputed_regions", [])
        }
    )
    atomic_write_json(engine.report_path, combined)
    return combined
