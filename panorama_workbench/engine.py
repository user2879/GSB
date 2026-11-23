from __future__ import annotations

import copy
import hashlib
import threading
from concurrent.futures import Future, ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .cache import (
    StateStore,
    atomic_write_json,
    decode_array,
    encode_array,
)
from .commands import CommandResult, apply_commands, read_command_stream
from .config import WorkbenchConfig
from .evaluation import EvaluationResult, evaluate_predictions
from .failures import Failure, failures_as_dicts, sort_failures
from .inference import (
    DeterministicScorer,
    TileGeometry,
    TileInference,
    TileMask,
    classify_tile,
    feature_cache_key,
    infer_tile,
    instance_input_key,
    mask_cache_key,
    tile_content_hash,
    tile_geometries,
)
from .merging import MergeResult, merge_masks
from .snapshot import SnapshotStore
from .validation import Sample, ValidationResult, discover_image_specs, validate_samples


@dataclass
class RunResult:
    exit_code: int
    report: dict[str, Any]
    failures: tuple[Failure, ...]
    recovery_code: str
    stopped: bool
    pending: tuple[str, ...]


@dataclass
class _TileCandidate:
    geometry: TileGeometry
    content_hash: str
    feature_key: str


@dataclass
class _ImagePlan:
    order: int
    prior_pending: bool
    sample: Sample
    old_record: dict[str, Any] | None
    candidates: list[_TileCandidate]
    estimated_cost: int
    exact_reuse: bool
    old_instance_ids: list[int] = field(default_factory=list)
    changed_tile_keys: list[str] = field(default_factory=list)
    invalidated_instance_ids: list[int] = field(default_factory=list)

    @property
    def sort_key(self) -> tuple[int, str]:
        return (0 if self.prior_pending else 1, self.sample.name)


@dataclass
class _ImageExecution:
    plan: _ImagePlan
    record: dict[str, Any]
    instance_payload: dict[str, Any]
    evaluation: EvaluationResult
    failures: list[Failure]
    spent_pixels: int
    tile_hits: int
    tile_misses: int
    mask_hits: int
    mask_recomputes: int
    decision_reuses: int
    instance_hits: int
    reused_tiles: list[str]
    recomputed_tiles: list[str]
    exact_reuse: bool


class _ImageBudgetExceeded(RuntimeError):
    def __init__(self, spent: int, quota: int) -> None:
        super().__init__(f"image mask pixel usage {spent} exceeds quota {quota}")
        self.spent = spent
        self.quota = quota


class Workbench:
    def __init__(self, state_dir: str | Path) -> None:
        self.state_dir = Path(state_dir).resolve()
        self.store = StateStore(self.state_dir)
        self.snapshots = SnapshotStore(self.state_dir)
        self._cache_lock = threading.RLock()
        self._active_lock = threading.Lock()
        self._active_workers = 0
        self._peak_workers = 0

    def run(
        self,
        samples_dir: str | Path,
        config: WorkbenchConfig,
        commands: list[dict[str, Any]] | str | Path | None = None,
        command_base: str | Path | None = None,
        report_path: str | Path | None = None,
        stop_event: threading.Event | None = None,
        stop_after_images: int | None = None,
    ) -> RunResult:
        command_list: list[dict[str, Any]] = []
        if commands is not None:
            if isinstance(commands, (str, Path)):
                command_list = read_command_stream(commands)
                command_base = command_base or Path(commands).resolve().parent
            else:
                command_list = commands
        base_specs = discover_image_specs(samples_dir)
        command_result = apply_commands(
            base_specs,
            config,
            command_list,
            command_base=command_base,
        )
        effective_config = command_result.config
        validation = validate_samples(command_result.specs, effective_config, samples_dir)
        load_failures = list(command_result.failures) + list(validation.failures)

        state, state_error = self.store.load()
        manifest, manifest_error = self.snapshots.load_manifest()
        recovery_code = _recovery_code(
            state,
            state_error,
            manifest,
            manifest_error,
            effective_config.hash,
        )
        if state_error or not _state_is_valid(state):
            state = _empty_state()
        state = copy.deepcopy(state)
        previous_global_version = max(
            int(state.get("global_version", 0)),
            int(manifest.get("global_version", 0)) if manifest else 0,
        )
        global_version = previous_global_version + 1
        prior_pending = set(state.get("pending", []))
        prior_records = copy.deepcopy(state.get("images", {}))
        valid_names = {sample.name for sample in validation.samples}

        for name in list(state.get("images", {})):
            if name not in valid_names:
                state["images"].pop(name, None)
                continue
            record = state["images"][name]
            sample = next(item for item in validation.samples if item.name == name)
            if (
                record.get("config_hash") != effective_config.hash
                or record.get("content_hash") != sample.content_hash
                or record.get("truth_hash") != sample.truth_hash
            ):
                state["images"].pop(name, None)

        scorer = DeterministicScorer(effective_config)
        plans = [
            self._plan_image(
                index,
                sample,
                effective_config,
                scorer,
                state,
                prior_records.get(sample.name),
                sample.name in prior_pending,
            )
            for index, sample in enumerate(validation.samples)
        ]
        plans.sort(key=lambda plan: plan.sort_key)
        if stop_after_images is not None:
            stop_after_images = max(0, int(stop_after_images))

        failures: list[Failure] = list(load_failures)
        pending: list[str] = []
        selected: list[tuple[_ImagePlan, int]] = []
        budget_entries: list[dict[str, Any]] = []
        remaining_budget = effective_config.pixel_budget
        for plan in plans:
            estimated = plan.estimated_cost
            quota = remaining_budget
            if estimated <= quota:
                selected.append((plan, estimated))
                remaining_budget -= estimated
                budget_entries.append(
                    {
                        "estimated_pixels": estimated,
                        "image": plan.sample.name,
                        "priority": "pending" if plan.prior_pending else "normal",
                        "quota_pixels": quota,
                        "reserved_pixels": estimated,
                        "status": "reserved",
                    }
                )
            else:
                pending.append(plan.sample.name)
                budget_entries.append(
                    {
                        "estimated_pixels": estimated,
                        "image": plan.sample.name,
                        "priority": "pending" if plan.prior_pending else "normal",
                        "quota_pixels": quota,
                        "reserved_pixels": 0,
                        "status": "budget_exhausted",
                    }
                )
                failures.append(
                    Failure(
                        code="E_IMAGE_BUDGET_EXHAUSTED",
                        item=plan.sample.name,
                        stage="schedule",
                        message="image requires more mask pixels than its allocated quota",
                        details={
                            "estimated_pixels": estimated,
                            "quota_pixels": quota,
                        },
                    )
                )
        pending_plans = [plan for plan in plans if plan.sample.name in pending]
        budget_effectively_exhausted = bool(pending_plans) and all(
            plan.estimated_cost > remaining_budget for plan in pending_plans
        )
        if remaining_budget == 0 or budget_effectively_exhausted:
            already_failed = {
                failure.item
                for failure in failures
                if failure.code == "E_PIXEL_BUDGET_PENDING"
            }
            for plan in pending_plans:
                if plan.sample.name not in already_failed:
                    failures.append(
                        Failure(
                            code="E_PIXEL_BUDGET_PENDING",
                            item=plan.sample.name,
                            stage="schedule",
                            message="global pixel budget is exhausted; image remains pending",
                            details={
                                "pixel_budget": effective_config.pixel_budget,
                                "remaining_budget": remaining_budget,
                            },
                        )
                    )

        state["global_version"] = global_version
        state["config_hash"] = effective_config.hash
        state["cumulative_consumed"] = int(state.get("cumulative_consumed", 0))
        state["run_complete"] = False
        state["pending"] = sorted(
            {plan.sample.name for plan in plans},
            key=lambda name: (0 if name in prior_pending else 1, name),
        )
        completed_this_run = 0
        stopped = bool(stop_event and stop_event.is_set())
        executions: list[_ImageExecution] = []
        peak_workers = 0
        batch_consumed = 0

        if not stopped:
            with ThreadPoolExecutor(
                max_workers=effective_config.max_workers,
                thread_name_prefix="panorama",
            ) as executor:
                wave_start = 0
                while wave_start < len(selected):
                    if stop_event is not None and stop_event.is_set():
                        stopped = True
                        break
                    if (
                        stop_after_images is not None
                        and completed_this_run >= stop_after_images
                    ):
                        stopped = True
                        break
                    wave = selected[
                        wave_start : wave_start + effective_config.max_workers
                    ]
                    wave_start += len(wave)
                    futures: dict[Future[_ImageExecution], tuple[_ImagePlan, int]] = {}
                    for plan, quota in wave:
                        future = executor.submit(
                            self._execute_plan,
                            plan,
                            effective_config,
                            scorer,
                            state,
                            quota,
                            global_version,
                        )
                        futures[future] = (plan, quota)
                    pending_futures = set(futures)
                    while pending_futures:
                        done, pending_futures = wait(
                            pending_futures, return_when=FIRST_COMPLETED
                        )
                        for future in sorted(done, key=lambda item: futures[item][0].sort_key):
                            plan, quota = futures[future]
                            try:
                                execution = future.result()
                            except _ImageBudgetExceeded as exc:
                                pending.append(plan.sample.name)
                                failures.append(
                                    Failure(
                                        code="E_IMAGE_BUDGET_EXHAUSTED",
                                        item=plan.sample.name,
                                        stage="schedule",
                                        message=str(exc),
                                        details={
                                            "quota_pixels": exc.quota,
                                            "spent_pixels": exc.spent,
                                        },
                                    )
                                )
                                budget_entries.append(
                                    {
                                        "estimated_pixels": plan.estimated_cost,
                                        "image": plan.sample.name,
                                        "priority": "pending" if plan.prior_pending else "normal",
                                        "quota_pixels": quota,
                                        "reserved_pixels": quota,
                                        "status": "aborted_budget",
                                    }
                                )
                                continue
                            except Exception as exc:
                                pending.append(plan.sample.name)
                                failures.append(
                                    Failure(
                                        code="E_RUNTIME",
                                        item=plan.sample.name,
                                        stage="inference",
                                        message="image task failed",
                                        details={"error": repr(exc)},
                                    )
                                )
                                continue
                            self._commit_execution(
                                state,
                                execution,
                                effective_config,
                            )
                            executions.append(execution)
                            failures.extend(execution.failures)
                            completed_this_run += 1
                            batch_consumed += execution.spent_pixels
                            peak_workers = max(
                                peak_workers, min(len(wave), effective_config.max_workers)
                            )
                            _update_budget_entry(
                                budget_entries,
                                plan.sample.name,
                                execution.spent_pixels,
                            )
                            if (
                                completed_this_run % effective_config.snapshot_interval == 0
                            ):
                                self._checkpoint(
                                    state,
                                    effective_config,
                                    validation,
                                    sorted(
                                        set(plan.sample.name for plan in plans)
                                        - set(state.get("images", {}))
                                    ),
                                )
                            if (
                                stop_after_images is not None
                                and completed_this_run >= stop_after_images
                            ):
                                stopped = True
                    if stopped:
                        break

        for plan, quota in selected:
            if not any(execution.plan.sample.name == plan.sample.name for execution in executions) and plan.sample.name not in pending:
                pending.append(plan.sample.name)
                for entry in budget_entries:
                    if entry["image"] == plan.sample.name and entry["status"] == "reserved":
                        entry["status"] = "not_started_stop" if stopped else "not_started"
                        entry["consumed_pixels"] = 0
                        break

        for plan in plans:
            if plan.sample.name in pending:
                continue
            record = state.get("images", {}).get(plan.sample.name)
            if record is None:
                pending.append(plan.sample.name)

        pending_names = sorted(set(pending), key=lambda name: (0 if name in prior_pending else 1, name))
        if stopped:
            for name in pending_names:
                if not any(
                    failure.code == "E_IMAGE_BUDGET_EXHAUSTED" and failure.item == name
                    for failure in failures
                ):
                    break
        self._prune_and_evict(state, effective_config, global_version, pending_names)

        completed_records = {
            name: state["images"][name]
            for name in sorted(state.get("images", {}))
            if name in valid_names
        }
        eval_results: dict[str, EvaluationResult] = {}
        for execution in executions:
            eval_results[execution.plan.sample.name] = execution.evaluation
        for name, record in completed_records.items():
            if name in eval_results:
                continue
            sample = next(item for item in validation.samples if item.name == name)
            eval_key = _evaluation_key(record["instance_key"], sample.truth_hash, effective_config)
            cached = state.get("evaluation_cache", {}).get(eval_key)
            if cached:
                eval_results[name] = _evaluation_from_cache(cached)
                continue
            instance_payload = state.get("instance_cache", {}).get(record["instance_key"])
            if instance_payload is None:
                continue
            result = _evaluate_payload(sample, instance_payload, effective_config)
            state.setdefault("evaluation_cache", {})[eval_key] = _evaluation_to_cache(result)
            eval_results[name] = result
            failures.extend(result.failures)
        failures = sort_failures(_deduplicate_failures(failures))

        aggregate = _empty_cache_audit()
        for execution in executions:
            _add_cache_audit(aggregate, execution)
        aggregate["evicted_tile_count"] = state.get("last_evicted_tile_count", 0)
        aggregate["tile_cache_size"] = len(state.get("tile_cache", {}))

        report = self._build_report(
            state=state,
            manifest=None,
            config=effective_config,
            validation=validation,
            failures=failures,
            pending=pending_names,
            budget_entries=budget_entries,
            cache_audit=aggregate,
            executions=executions,
            eval_results=eval_results,
            recovery_code=recovery_code,
            global_version=global_version,
            stopped=stopped,
            command_result=command_result,
            batch_consumed=batch_consumed,
            peak_workers=peak_workers,
        )
        report["recovery"]["state_load_error"] = state_error
        report["recovery"]["manifest_load_error"] = manifest_error

        manifest_payload = {
            "batch_consumed": batch_consumed,
            "clean": not pending_names and not stopped,
            "completed_images": sorted(completed_records),
            "config_hash": effective_config.hash,
            "cumulative_consumed": int(state.get("cumulative_consumed", 0)) + batch_consumed,
            "global_version": global_version,
            "pending": pending_names,
            "schema": 1,
            "stage_hashes": {},
            "stage_versions": {},
        }
        if manifest is not None:
            manifest_payload["stage_hashes"] = dict(manifest.get("stage_hashes", {}))
            manifest_payload["stage_versions"] = dict(manifest.get("stage_versions", {}))

        stage_payloads = self._stage_payloads(
            validation,
            state,
            effective_config,
            failures,
            eval_results,
            completed_records,
            global_version,
        )
        for stage_name, payload in stage_payloads.items():
            self.snapshots.publish_stage(stage_name, payload, manifest_payload)
        manifest_payload["clean"] = not pending_names and not stopped
        manifest_payload["completed_images"] = sorted(completed_records)
        manifest_payload["pending"] = pending_names
        state["stage_versions"] = dict(manifest_payload["stage_versions"])
        state["stage_hashes"] = dict(manifest_payload["stage_hashes"])
        state["cumulative_consumed"] = int(state.get("cumulative_consumed", 0)) + batch_consumed
        state["last_batch_consumed"] = batch_consumed
        state["pending"] = pending_names
        state["completed_images"] = sorted(completed_records)
        state["run_complete"] = True
        state["clean"] = manifest_payload["clean"]
        self.store.save(state)
        self.snapshots.save_manifest(manifest_payload)

        report["manifest"] = {
            "clean": manifest_payload["clean"],
            "completed_images": manifest_payload["completed_images"],
            "config_hash": manifest_payload["config_hash"],
            "global_version": manifest_payload["global_version"],
            "pending": manifest_payload["pending"],
            "stage_versions": dict(manifest_payload["stage_versions"]),
        }
        report["state"]["completed_images"] = sorted(completed_records)
        report["state"]["consumed_cumulative"] = state["cumulative_consumed"]
        report["state"]["pending"] = pending_names
        exit_code = _exit_code(
            stopped=stopped,
            pending=pending_names,
            failures=failures,
        )
        report["run"]["exit_code"] = exit_code
        report["run"]["stopped"] = stopped
        report_payload = _stable_report(report)
        reports_dir = self.state_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_file = reports_dir / f"report-v{global_version:08d}.json"
        atomic_write_json(report_file, report_payload)
        atomic_write_json(reports_dir / "latest.json", report_payload)
        if report_path is not None:
            atomic_write_json(report_path, report_payload)

        return RunResult(
            exit_code=exit_code,
            report=report_payload,
            failures=tuple(failures),
            recovery_code=recovery_code,
            stopped=stopped,
            pending=tuple(pending_names),
        )

    def _plan_image(
        self,
        order: int,
        sample: Sample,
        config: WorkbenchConfig,
        scorer: DeterministicScorer,
        state: dict[str, Any],
        old_record: dict[str, Any] | None,
        prior_pending: bool,
    ) -> _ImagePlan:
        geometries = tile_geometries(
            sample.width,
            sample.height,
            config.window_size,
            config.window_stride,
        )
        exact_reuse = bool(
            old_record
            and old_record.get("config_hash") == config.hash
            and old_record.get("content_hash") == sample.content_hash
            and old_record.get("truth_hash") == sample.truth_hash
            and old_record.get("instance_key")
            in state.get("instance_cache", {})
        )
        changed_tile_keys: list[str] = []
        invalidated_instance_ids: list[int] = []
        old_instance_ids: list[int] = []
        old_mask_keys = (old_record or {}).get("mask_keys", {})
        old_tile_keys = (old_record or {}).get("tile_keys", {})
        old_instance_payload = None
        if old_record is not None:
            old_instance_payload = state.get("instance_cache", {}).get(
                old_record.get("instance_key", "")
            )
            if old_instance_payload:
                old_instance_ids = sorted(
                    int(instance["instance_id"])
                    for instance in old_instance_payload.get("instances", [])
                )
        candidates: list[_TileCandidate] = []
        if exact_reuse:
            cost = 0
            for geometry in geometries:
                tile = np.ascontiguousarray(
                    sample.rgb[
                        geometry.y : geometry.y + geometry.size,
                        geometry.x : geometry.x + geometry.size,
                        :,
                    ]
                )
                content_hash = tile_content_hash(tile)
                candidates.append(
                    _TileCandidate(
                        geometry=geometry,
                        content_hash=content_hash,
                        feature_key=feature_cache_key(content_hash, geometry, config),
                    )
                )
        else:
            cost = 0
            all_masks_current = True
            mask_payloads: dict[str, str] = {}
            for geometry in geometries:
                tile = np.ascontiguousarray(
                    sample.rgb[
                        geometry.y : geometry.y + geometry.size,
                        geometry.x : geometry.x + geometry.size,
                        :,
                    ]
                )
                content_hash = tile_content_hash(tile)
                feature_key = feature_cache_key(content_hash, geometry, config)
                candidate = _TileCandidate(geometry, content_hash, feature_key)
                candidates.append(candidate)
                old_feature_key = old_tile_keys.get(geometry.key)
                if old_feature_key != feature_key:
                    changed_tile_keys.append(geometry.key)
                tile_record = state.get("tile_cache", {}).get(feature_key)
                expected_mask_key = mask_cache_key(feature_key, config, geometry)
                if tile_record is None:
                    cost += geometry.size * geometry.size
                    all_masks_current = False
                elif (
                    tile_record.get("declared_mask_key") != expected_mask_key
                    or not tile_record.get("class_grid")
                ):
                    cost += geometry.size * geometry.size
                    all_masks_current = False
                else:
                    mask_payloads[geometry.key] = tile_record["mask_payload_hash"]
                if old_mask_keys.get(geometry.key) != expected_mask_key:
                    if geometry.key not in changed_tile_keys:
                        changed_tile_keys.append(geometry.key)
            instance_hit = False
            if all_masks_current:
                candidate_masks = []
                try:
                    for candidate in candidates:
                        tile_record = state.get("tile_cache", {}).get(
                            candidate.feature_key
                        )
                        if tile_record is None:
                            all_masks_current = False
                            break
                        candidate_masks.append(
                            _mask_from_record(
                                tile_record,
                                candidate.geometry,
                                config,
                            )
                        )
                except Exception:
                    all_masks_current = False
                if all_masks_current:
                    candidate_key = instance_input_key(
                        sample.width,
                        sample.height,
                        candidate_masks,
                        config,
                    )
                    instance_hit = candidate_key in state.get("instance_cache", {})
            if not instance_hit:
                cost += sum(
                    geometry.size * geometry.size for geometry in geometries
                )
        if old_instance_payload:
            if exact_reuse:
                invalidated_instance_ids = []
            else:
                merge_changed = bool(
                    old_record
                    and (
                        old_record.get("merge_hash") != config.merge_hash
                        or old_record.get("category_hash") != config.category_hash
                        or old_record.get("quant_level") != config.quant_level
                    )
                )
                changed_set = set(changed_tile_keys)
                for instance in old_instance_payload.get("instances", []):
                    if merge_changed or changed_set.intersection(instance.get("tile_keys", [])):
                        invalidated_instance_ids.append(int(instance["instance_id"]))
                invalidated_instance_ids.sort()
        return _ImagePlan(
            order=order,
            prior_pending=prior_pending,
            sample=sample,
            old_record=old_record,
            candidates=candidates,
            estimated_cost=cost,
            exact_reuse=exact_reuse,
            old_instance_ids=old_instance_ids,
            changed_tile_keys=sorted(changed_tile_keys),
            invalidated_instance_ids=invalidated_instance_ids,
        )

    def _execute_plan(
        self,
        plan: _ImagePlan,
        config: WorkbenchConfig,
        scorer: DeterministicScorer,
        state: dict[str, Any],
        quota: int,
        global_version: int,
    ) -> _ImageExecution:
        with self._active_lock:
            self._active_workers += 1
            self._peak_workers = max(self._peak_workers, self._active_workers)
        try:
            return self._execute_plan_inner(
                plan, config, scorer, state, quota, global_version
            )
        finally:
            with self._active_lock:
                self._active_workers -= 1

    def _execute_plan_inner(
        self,
        plan: _ImagePlan,
        config: WorkbenchConfig,
        scorer: DeterministicScorer,
        state: dict[str, Any],
        quota: int,
        global_version: int,
    ) -> _ImageExecution:
        sample = plan.sample
        if plan.exact_reuse and plan.old_record is not None:
            record = copy.deepcopy(plan.old_record)
            instance_payload = state["instance_cache"][record["instance_key"]]
            tile_keys = record.get("tile_keys", {})
            mask_keys = record.get("mask_keys", {})
            with self._cache_lock:
                tile_hits = sum(
                    key in state.get("tile_cache", {}) for key in tile_keys.values()
                )
                mask_hits = sum(
                    state.get("tile_cache", {})
                    .get(tile_keys.get(tile_key), {})
                    .get("declared_mask_key")
                    == mask_key
                    for tile_key, mask_key in mask_keys.items()
                )
            eval_key = _evaluation_key(record["instance_key"], sample.truth_hash, config)
            cached_eval = state.get("evaluation_cache", {}).get(eval_key)
            if cached_eval:
                evaluation = _evaluation_from_cache(cached_eval)
            else:
                evaluation = _evaluate_payload(sample, instance_payload, config)
            record["eval_key"] = eval_key
            return _ImageExecution(
                plan=plan,
                record=record,
                instance_payload=instance_payload,
                evaluation=evaluation,
                failures=list(evaluation.failures),
                spent_pixels=0,
                tile_hits=tile_hits,
                tile_misses=0,
                mask_hits=mask_hits,
                mask_recomputes=0,
                decision_reuses=mask_hits,
                instance_hits=1,
                reused_tiles=sorted(record.get("tile_keys", {})),
                recomputed_tiles=[],
                exact_reuse=True,
            )

        spent = 0
        tile_hits = 0
        tile_misses = 0
        mask_hits = 0
        mask_recomputes = 0
        decision_reuses = 0
        reused_tiles: list[str] = []
        recomputed_tiles: list[str] = []
        masks: list[TileMask] = []
        tile_keys: dict[str, str] = {}
        mask_keys: dict[str, str] = {}
        mask_payloads: dict[str, str] = {}

        for candidate in plan.candidates:
            geometry = candidate.geometry
            tile_pixels = geometry.size * geometry.size
            expected_mask_key = mask_cache_key(
                candidate.feature_key, config, geometry
            )
            with self._cache_lock:
                tile_record = state.get("tile_cache", {}).get(candidate.feature_key)
            previous_payload = None
            previous_mask_key = None
            record_for_reuse = None
            if tile_record is not None:
                try:
                    features = decode_array(tile_record["features"])
                    if tuple(features.shape) != (geometry.size, geometry.size, 14):
                        raise ValueError("feature shape mismatch")
                    inference = TileInference(
                        geometry=geometry,
                        content_hash=candidate.content_hash,
                        feature_key=candidate.feature_key,
                        features=features,
                        residual=dict(tile_record.get("residual", {})),
                    )
                    record_for_reuse = tile_record
                    tile_hits += 1
                    previous_payload = tile_record.get("mask_payload_hash")
                    previous_mask_key = tile_record.get("declared_mask_key")
                except Exception:
                    tile_record = None
            if tile_record is None:
                inference = infer_tile(sample.rgb, geometry, config, scorer)
                tile_misses += 1
                spent += tile_pixels
                if spent > quota:
                    raise _ImageBudgetExceeded(spent, quota)
            assert inference is not None
            if (
                record_for_reuse is not None
                and previous_mask_key == expected_mask_key
                and record_for_reuse.get("class_grid")
            ):
                mask_was_reused = True
                try:
                    class_grid = decode_array(record_for_reuse["class_grid"]).astype(
                        np.int32, copy=False
                    )
                    if tuple(class_grid.shape) != (geometry.size, geometry.size):
                        raise ValueError("mask shape mismatch")
                    counts_raw = record_for_reuse.get("class_counts", {})
                    counts = {int(key): int(value) for key, value in counts_raw.items()}
                    mask = TileMask(
                        geometry=geometry,
                        feature_key=candidate.feature_key,
                        mask_key=expected_mask_key,
                        class_grid=class_grid,
                        class_counts=counts,
                        payload_hash=record_for_reuse["mask_payload_hash"],
                    )
                    mask_hits += 1
                except Exception:
                    mask = classify_tile(inference, scorer, config)
                    mask_recomputes += 1
                    mask_was_reused = False
                    spent += tile_pixels
                    if spent > quota:
                        raise _ImageBudgetExceeded(spent, quota)
            else:
                mask = classify_tile(inference, scorer, config)
                if record_for_reuse is None:
                    mask_recomputes += 1
                    mask_was_reused = False
                else:
                    spent += tile_pixels
                    if spent > quota:
                        raise _ImageBudgetExceeded(spent, quota)
                    if previous_payload == mask.payload_hash:
                        mask_hits += 1
                        decision_reuses += 1
                        mask_was_reused = True
                    else:
                        mask_recomputes += 1
                        mask_was_reused = False
            tile_record_payload = {
                "class_counts": {str(key): value for key, value in mask.class_counts.items()},
                "class_grid": encode_array(mask.class_grid),
                "content_hash": candidate.content_hash,
                "declared_mask_key": mask.mask_key,
                "feature_key": candidate.feature_key,
                "features": encode_array(inference.features),
                "last_used_version": global_version,
                "mask_payload_hash": mask.payload_hash,
                "residual": dict(inference.residual),
                "tile": {
                    "col": geometry.col,
                    "row": geometry.row,
                    "size": geometry.size,
                    "x": geometry.x,
                    "y": geometry.y,
                },
            }
            with self._cache_lock:
                state.setdefault("tile_cache", {})[candidate.feature_key] = tile_record_payload
            masks.append(mask)
            tile_keys[geometry.key] = candidate.feature_key
            mask_keys[geometry.key] = mask.mask_key
            mask_payloads[geometry.key] = mask.payload_hash
            if mask_was_reused:
                reused_tiles.append(geometry.key)
            else:
                recomputed_tiles.append(geometry.key)

        input_key = instance_input_key(
            sample.width, sample.height, masks, config
        )
        instance_payload = state.get("instance_cache", {}).get(input_key)
        instance_hit = instance_payload is not None
        if instance_payload is None:
            merge = merge_masks(sample.width, sample.height, masks, config)
            spent += merge.processed_mask_pixels
            if spent > quota:
                raise _ImageBudgetExceeded(spent, quota)
            instance_payload = _merge_to_payload(merge)
        instance_key = input_key
        eval_key = _evaluation_key(instance_key, sample.truth_hash, config)
        cached_eval = state.get("evaluation_cache", {}).get(eval_key)
        if cached_eval:
            evaluation = _evaluation_from_cache(cached_eval)
        else:
            evaluation = _evaluate_payload(sample, instance_payload, config)
        record = {
            "category_hash": config.category_hash,
            "completed_global_version": global_version,
            "config_hash": config.hash,
            "content_hash": sample.content_hash,
            "eval_key": eval_key,
            "fragment_count": len(instance_payload.get("fragments", [])),
            "height": sample.height,
            "inference_hash": config.inference_hash,
            "instance_key": instance_key,
            "mask_keys": dict(sorted(mask_keys.items())),
            "mask_payloads": dict(sorted(mask_payloads.items())),
            "merge_hash": config.merge_hash,
            "name": sample.name,
            "quant_level": config.quant_level,
            "tile_count": len(tile_keys),
            "tile_keys": dict(sorted(tile_keys.items())),
            "truth_hash": sample.truth_hash,
            "width": sample.width,
        }
        old_payloads = (plan.old_record or {}).get("mask_payloads", {})
        actual_changed_tiles = sorted(
            key
            for key, value in mask_payloads.items()
            if old_payloads.get(key) != value
        )
        plan.changed_tile_keys = actual_changed_tiles
        old_instance_payload = state.get("instance_cache", {}).get(
            (plan.old_record or {}).get("instance_key", "")
        )
        if old_instance_payload:
            merge_changed = bool(
                plan.old_record
                and plan.old_record.get("merge_hash") != config.merge_hash
            )
            changed_set = set(actual_changed_tiles)
            plan.invalidated_instance_ids = sorted(
                int(instance["instance_id"])
                for instance in old_instance_payload.get("instances", [])
                if merge_changed
                or changed_set.intersection(instance.get("tile_keys", []))
            )
        else:
            plan.invalidated_instance_ids = []
        return _ImageExecution(
            plan=plan,
            record=record,
            instance_payload=instance_payload,
            evaluation=evaluation,
            failures=list(evaluation.failures),
            spent_pixels=spent,
            tile_hits=tile_hits,
            tile_misses=tile_misses,
            mask_hits=mask_hits,
            mask_recomputes=mask_recomputes,
            decision_reuses=decision_reuses,
            instance_hits=1 if instance_hit else 0,
            reused_tiles=sorted(reused_tiles),
            recomputed_tiles=sorted(recomputed_tiles),
            exact_reuse=False,
        )

    def _commit_execution(
        self,
        state: dict[str, Any],
        execution: _ImageExecution,
        config: WorkbenchConfig,
    ) -> None:
        name = execution.plan.sample.name
        state.setdefault("images", {})[name] = execution.record
        state.setdefault("instance_cache", {})[
            execution.record["instance_key"]
        ] = execution.instance_payload
        state.setdefault("evaluation_cache", {})[
            execution.record["eval_key"]
        ] = _evaluation_to_cache(execution.evaluation)
        state["pending"] = [
            pending for pending in state.get("pending", []) if pending != name
        ]

    def _checkpoint(
        self,
        state: dict[str, Any],
        config: WorkbenchConfig,
        validation: ValidationResult,
        pending: Iterable[str],
    ) -> None:
        with self._cache_lock:
            checkpoint = copy.deepcopy(state)
        checkpoint["config_hash"] = config.hash
        checkpoint["pending"] = sorted(set(pending))
        checkpoint["run_complete"] = False
        self.store.save(checkpoint)

    def _prune_and_evict(
        self,
        state: dict[str, Any],
        config: WorkbenchConfig,
        global_version: int,
        pending: list[str],
    ) -> None:
        retained_instance_keys = {
            record.get("instance_key")
            for record in state.get("images", {}).values()
            if record.get("instance_key")
        }
        state["instance_cache"] = {
            key: value
            for key, value in state.get("instance_cache", {}).items()
            if key in retained_instance_keys
        }
        retained_eval_keys = {
            record.get("eval_key")
            for record in state.get("images", {}).values()
            if record.get("eval_key")
        }
        state["evaluation_cache"] = {
            key: value
            for key, value in state.get("evaluation_cache", {}).items()
            if key in retained_eval_keys
        }
        tile_cache = state.get("tile_cache", {})
        for key, record in tile_cache.items():
            record.setdefault("last_used_version", 0)
        ordered = sorted(
            tile_cache,
            key=lambda key: (
                int(tile_cache[key].get("last_used_version", 0)), key
            ),
        )
        evicted = 0
        while len(tile_cache) > config.cache_budget and ordered:
            key = ordered.pop(0)
            tile_cache.pop(key, None)
            evicted += 1
        state["last_evicted_tile_count"] = evicted

    def _build_report(
        self,
        state: dict[str, Any],
        manifest: dict[str, Any] | None,
        config: WorkbenchConfig,
        validation: ValidationResult,
        failures: list[Failure],
        pending: list[str],
        budget_entries: list[dict[str, Any]],
        cache_audit: dict[str, Any],
        executions: list[_ImageExecution],
        eval_results: dict[str, EvaluationResult],
        recovery_code: str,
        global_version: int,
        stopped: bool,
        command_result: CommandResult,
        batch_consumed: int,
        peak_workers: int,
    ) -> dict[str, Any]:
        execution_by_name = {execution.plan.sample.name: execution for execution in executions}
        image_reports: dict[str, Any] = {}
        truncated_instances: list[dict[str, Any]] = []
        for name in sorted(state.get("images", {})):
            record = state["images"][name]
            payload = state.get("instance_cache", {}).get(record.get("instance_key", ""))
            if payload is None:
                continue
            evaluation = eval_results.get(name)
            semantic = decode_array(payload["semantic_raster"])
            instances_raster = decode_array(payload["instance_raster"])
            for item in payload.get("truncated_instances", []):
                truncated_instances.append({"image": name, **item})
            execution = execution_by_name.get(name)
            image_reports[name] = {
                "cache": {
                    "decision_reuses": execution.decision_reuses if execution else 0,
                    "exact_image_reuse": bool(execution and execution.exact_reuse),
                    "instance_hits": execution.instance_hits if execution else 1,
                    "mask_hits": execution.mask_hits if execution else len(record.get("mask_keys", {})),
                    "mask_recomputes": execution.mask_recomputes if execution else 0,
                    "recomputed_tiles": execution.recomputed_tiles if execution else [],
                    "reused_tiles": execution.reused_tiles if execution else sorted(record.get("tile_keys", {})),
                    "tile_hits": execution.tile_hits if execution else len(record.get("tile_keys", {})),
                    "tile_misses": execution.tile_misses if execution else 0,
                },
                "dropped_instances": list(payload.get("dropped_instances", [])),
                "evaluation": evaluation.as_dict() if evaluation else None,
                "instance_raster": instances_raster.reshape(-1).tolist(),
                "instances": list(payload.get("instances", [])),
                "invalidated_instance_ids": (
                    execution.plan.invalidated_instance_ids if execution else []
                ),
                "semantic_raster": semantic.reshape(-1).tolist(),
                "shape": [record["height"], record["width"]],
                "truncated_instances": list(payload.get("truncated_instances", [])),
            }
        report = {
            "budget": {
                "batch_limit": config.pixel_budget,
                "batch_consumed": batch_consumed,
                "entries": budget_entries,
                "remaining": max(0, config.pixel_budget - batch_consumed),
            },
            "cache": cache_audit,
            "commands": command_result.audit,
            "config": config.as_report(),
            "failure_list": failures_as_dicts(failures),
            "images": image_reports,
            "manifest": {},
            "pending_queue": pending,
            "recovery": {"code": recovery_code, "state_load_error": None},
            "run": {
                "exit_code": 0,
                "global_version": global_version,
                "peak_workers": self._peak_workers,
                "stopped": stopped,
            },
            "samples": validation.summary,
            "state": {
                "completed_images": sorted(state.get("images", {})),
                "consumed_cumulative": int(state.get("cumulative_consumed", 0)),
                "pending": pending,
            },
            "truncated_instances": truncated_instances,
        }
        return report

    def _stage_payloads(
        self,
        validation: ValidationResult,
        state: dict[str, Any],
        config: WorkbenchConfig,
        failures: list[Failure],
        eval_results: dict[str, EvaluationResult],
        completed_records: dict[str, Any],
        global_version: int,
    ) -> dict[str, dict[str, Any]]:
        load_payload = {
            "config_hash": config.hash,
            "failures": failures_as_dicts(
                [failure for failure in failures if failure.stage == "load"]
            ),
            "samples": [
                {
                    "content_hash": sample.content_hash,
                    "height": sample.height,
                    "name": sample.name,
                    "truth_hash": sample.truth_hash,
                    "truth_instances": [item.as_dict() for item in sample.truth_instances],
                    "width": sample.width,
                }
                for sample in validation.samples
            ],
        }
        inference_payload = {
            "config_hash": config.hash,
            "images": {
                name: {
                    "tile_count": record.get("tile_count", 0),
                    "tile_keys": record.get("tile_keys", {}),
                }
                for name, record in sorted(completed_records.items())
            },
        }
        merge_payload = {
            "config_hash": config.hash,
            "images": {},
        }
        evaluation_payload = {
            "config_hash": config.hash,
            "images": {},
        }
        for name, record in sorted(completed_records.items()):
            payload = state.get("instance_cache", {}).get(record["instance_key"], {})
            merge_payload["images"][name] = {
                "dropped_instances": payload.get("dropped_instances", []),
                "instances": payload.get("instances", []),
                "truncated_instances": payload.get("truncated_instances", []),
            }
            evaluation = eval_results.get(name)
            evaluation_payload["images"][name] = (
                evaluation.as_dict() if evaluation else None
            )
        return {
            "evaluation": evaluation_payload,
            "inference": inference_payload,
            "load": load_payload,
            "merge": merge_payload,
        }


def _empty_state() -> dict[str, Any]:
    return {
        "clean": False,
        "completed_images": [],
        "config_hash": "",
        "cumulative_consumed": 0,
        "evaluation_cache": {},
        "global_version": 0,
        "images": {},
        "instance_cache": {},
        "last_batch_consumed": 0,
        "last_evicted_tile_count": 0,
        "pending": [],
        "run_complete": False,
        "schema": 1,
        "stage_hashes": {},
        "stage_versions": {},
        "tile_cache": {},
    }


def _state_is_valid(state: dict[str, Any] | None) -> bool:
    if not isinstance(state, dict):
        return False
    if state.get("schema") != 1:
        return False
    for key in (
        "images",
        "tile_cache",
        "instance_cache",
        "evaluation_cache",
        "pending",
        "completed_images",
    ):
        if key not in state:
            return False
    return all(
        isinstance(state[key], dict)
        for key in ("images", "tile_cache", "instance_cache", "evaluation_cache")
    )


def _recovery_code(
    state: dict[str, Any] | None,
    state_error: str | None,
    manifest: dict[str, Any] | None,
    manifest_error: str | None,
    config_hash: str,
) -> str:
    if state_error or manifest_error:
        return "R_FULL_REBUILD"
    if state is None:
        if manifest is not None:
            return "R_FULL_REBUILD"
        return "R_NEW_RUN"
    state_hash = state.get("config_hash")
    manifest_hash = manifest.get("config_hash") if manifest else None
    if state_hash != config_hash or manifest_hash not in {None, config_hash}:
        return "R_CONFIG_RECONCILE"
    if state.get("run_complete") is False or (manifest and not manifest.get("clean", False)):
        return "R_RESUME_CHECKPOINT"
    return "R_STABLE_RERUN"


def _merge_to_payload(merge: MergeResult) -> dict[str, Any]:
    return {
        "conflict_pixels": merge.conflict_pixels,
        "dropped_instances": list(merge.dropped_instances),
        "fragments": [
            {
                "area": fragment.area,
                "category_code": fragment.category_code,
                "fragment_id": fragment.fragment_id,
                "tile_key": fragment.tile_key,
            }
            for fragment in merge.fragments
        ],
        "instance_raster": encode_array(merge.instance_raster),
        "instances": [instance.as_dict() for instance in merge.instances],
        "mask_pixels_processed": merge.processed_mask_pixels,
        "merge_events": list(merge.merge_events),
        "semantic_raster": encode_array(merge.semantic_raster),
        "truncated_instances": list(merge.truncated_instances),
    }


def _evaluate_payload(
    sample: Sample,
    payload: dict[str, Any],
    config: WorkbenchConfig,
) -> EvaluationResult:
    from .merging import MergedInstance

    instances = tuple(
        MergedInstance(
            instance_id=int(item["instance_id"]),
            category_code=int(item["category_code"]),
            pixels=(),
            area=int(item["area"]),
            min_row=int(item["bbox"][1]),
            min_col=int(item["bbox"][0]),
            max_row=int(item["bbox"][1] + item["bbox"][3] - 1),
            max_col=int(item["bbox"][0] + item["bbox"][2] - 1),
            fragment_ids=tuple(item.get("fragment_ids", [])),
            tile_keys=tuple(item.get("tile_keys", [])),
            representative_fragment_id=str(item.get("representative_fragment_id", "")),
        )
        for item in payload.get("instances", [])
    )
    instance_raster = decode_array(payload["instance_raster"]).astype(np.int32, copy=False)
    semantic_raster = decode_array(payload["semantic_raster"]).astype(np.int32, copy=False)
    expanded: list[MergedInstance] = []
    for instance in instances:
        pixels = tuple(np.flatnonzero(instance_raster.reshape(-1) == instance.instance_id).tolist())
        expanded.append(
            MergedInstance(
                instance_id=instance.instance_id,
                category_code=instance.category_code,
                pixels=pixels,
                area=len(pixels),
                min_row=instance.min_row,
                min_col=instance.min_col,
                max_row=instance.max_row,
                max_col=instance.max_col,
                fragment_ids=instance.fragment_ids,
                tile_keys=instance.tile_keys,
                representative_fragment_id=instance.representative_fragment_id,
            )
        )
    return evaluate_predictions(sample, tuple(expanded), semantic_raster, instance_raster, config)


def _evaluation_key(instance_key: str, truth_hash: str, config: WorkbenchConfig) -> str:
    payload = (
        f"evaluation-v1|{instance_key}|{truth_hash}|"
        f"{config.instance_match_iou}|{config.min_semantic_accuracy}|"
        f"{config.min_instance_quality}"
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _evaluation_to_cache(evaluation: EvaluationResult) -> dict[str, Any]:
    return {
        **evaluation.as_dict(),
        "failure_dicts": failures_as_dicts(list(evaluation.failures)),
    }


def _evaluation_from_cache(payload: dict[str, Any]) -> EvaluationResult:
    failures = _failures_from_dicts(payload.get("failure_dicts", []))
    return EvaluationResult(
        semantic_accuracy=float(payload["semantic_accuracy"]),
        instance_quality=float(payload["instance_quality"]),
        mean_matched_iou=float(payload["mean_matched_iou"]),
        true_positive_count=int(payload["true_positive_count"]),
        false_positive_count=int(payload["false_positive_count"]),
        false_negative_count=int(payload["false_negative_count"]),
        matched_pairs=tuple(payload.get("matched_pairs", [])),
        predicted_diagnostics=tuple(payload.get("predicted_diagnostics", [])),
        unmatched_predicted_instances=tuple(
            payload.get("unmatched_predicted_instances", [])
        ),
        missed_truth_instances=tuple(payload.get("missed_truth_instances", [])),
        failures=tuple(failures),
    )


def _failures_from_dicts(values: list[dict[str, Any]]) -> list[Failure]:
    return [
        Failure(
            code=str(value.get("code", "E_UNKNOWN")),
            item=str(value.get("item", "")),
            stage=str(value.get("stage", "evaluation")),
            message=str(value.get("message", "")),
            details=dict(value.get("details", {})),
        )
        for value in values
    ]


def _mask_from_record(
    record: dict[str, Any], geometry: TileGeometry, config: WorkbenchConfig
) -> TileMask:
    class_grid = decode_array(record["class_grid"]).astype(np.int32, copy=False)
    counts = {int(key): int(value) for key, value in record.get("class_counts", {}).items()}
    return TileMask(
        geometry=geometry,
        feature_key=record["feature_key"],
        mask_key=record.get("declared_mask_key", ""),
        class_grid=class_grid,
        class_counts=counts,
        payload_hash=record["mask_payload_hash"],
    )


def _empty_cache_audit() -> dict[str, Any]:
    return {
        "decision_reuses": 0,
        "evicted_tile_count": 0,
        "instance_hits": 0,
        "mask_hits": 0,
        "mask_recomputes": 0,
        "tile_cache_size": 0,
        "tile_cache_hit_count": 0,
        "tile_hits": 0,
        "tile_misses": 0,
    }


def _add_cache_audit(audit: dict[str, Any], execution: _ImageExecution) -> None:
    for key in (
        "decision_reuses",
        "instance_hits",
        "mask_hits",
        "mask_recomputes",
        "tile_hits",
        "tile_misses",
    ):
        audit[key] += int(getattr(execution, key))
    audit["tile_cache_hit_count"] = audit["tile_hits"]


def _update_budget_entry(
    entries: list[dict[str, Any]], image: str, consumed: int
) -> None:
    for entry in entries:
        if entry["image"] == image and entry["status"] == "reserved":
            entry["consumed_pixels"] = consumed
            entry["status"] = "completed"
            return


def _deduplicate_failures(failures: list[Failure]) -> list[Failure]:
    unique: dict[tuple[str, str, str, str], Failure] = {}
    for failure in failures:
        unique[(failure.code, failure.item, failure.stage, failure.message)] = failure
    return list(unique.values())


def _exit_code(*, stopped: bool, pending: list[str], failures: list[Failure]) -> int:
    if stopped:
        return 130
    if any(failure.code in {"E_CONFIG_INVALID", "E_IMAGE_FORMAT", "E_TRUTH_INVALID"} for failure in failures):
        return 2
    if pending:
        return 3
    if any(failure.code == "E_QUALITY_BELOW" for failure in failures):
        return 4
    if any(failure.code == "E_RUNTIME" for failure in failures):
        return 5
    return 0


def _stable_report(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _stable_report(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_stable_report(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value
