"""End-to-end staged calibration pipeline with cache and diagnostics."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .calibration import calibrate_group
from .commands import apply_commands, load_commands
from .distillation import ATTRIBUTION_PRIORITY, analyze_group, build_group_distillation
from .errors import (
    E_BATCH_BUDGET_PENDING,
    E_CALIBRATION_INVALID,
    E_CONFIG_INVALID,
    E_GROUP_INVALID,
    E_INTERNAL,
    E_QUANT_BOUNDS,
    E_QUOTA_EXCEEDED,
    E_RECEIPT_ORPHAN,
    E_VERSION_ORDER,
    E_WEIGHT_DUPLICATE,
    E_WEIGHT_INVALID,
    E_WEIGHT_SHAPE,
    FailureList,
)
from .faults import apply_faults, load_faults
from .loading import load_dataset, validate_effective_config
from .model import Config, Dataset, Layer
from .quantization import quantize_group
from .receipts import build_receipt_ledger
from .scheduler import BudgetPlan, ScheduledGroup, budget_report, plan_budget
from .snapshot import StateStore
from .utils import clean_float, stable_hash, write_json


STAGE_VERSION = 1


@dataclass
class PipelineRequest:
    weights_path: Path
    calibration_path: Path
    config_path: Path
    state_dir: Path
    receipt_paths: tuple[Path, ...] = ()
    commands_path: Path | None = None
    faults_path: Path | None = None
    report_path: Path | None = None
    full_rebuild: bool = False


@dataclass
class StageTaskResult:
    layer: str
    group_index: int
    payload: dict[str, Any] | None
    cache_key: str
    reused: bool
    computed: bool
    artifact_path: Path | None = None
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class CalibrationPipeline:
    def __init__(self, request: PipelineRequest) -> None:
        self.request = request

    def run(self) -> dict[str, Any]:
        failures = FailureList()
        dataset = load_dataset(
            weights_path=Path(self.request.weights_path),
            calibration_path=Path(self.request.calibration_path),
            config_path=Path(self.request.config_path),
            receipt_paths=[Path(path) for path in self.request.receipt_paths],
        )

        commands_applied: list[dict[str, Any]] = []
        if self.request.commands_path is not None:
            commands = load_commands(Path(self.request.commands_path), failures)
            commands_applied = apply_commands(dataset, commands)
            if any(item.get("op") in {"replace_layer", "set_layer_weights", "patch_config", "set_config"} for item in commands_applied):
                validate_effective_config(dataset)

        fault_results: list[dict[str, Any]] = []
        if self.request.faults_path is not None:
            injections = load_faults(Path(self.request.faults_path), failures)
            fault_results = apply_faults(dataset, injections)

        deleted_layers = {
            str(item.get("layer"))
            for item in commands_applied
            if item.get("op") == "delete_layer"
        }
        for receipt in dataset.receipts.receipts:
            if receipt.layer in deleted_layers:
                failures.add(
                    E_RECEIPT_ORPHAN,
                    f"receipt references layer {receipt.layer!r} deleted by command stream",
                    "commands",
                    layer=receipt.layer,
                    details={"receipt_id": receipt.receipt_id},
                )

        # Loading and command failures live in the dataset; scheduler faults are
        # added below. Keep one FailureList to make code counts authoritative.
        failures.extend(dataset.failures.items)
        dataset.failures = failures

        layer_names = {layer.name for layer in dataset.layers}
        blocked_layers: set[str] = set()
        global_block = dataset.config is None or dataset.calibration is None or not dataset.layers
        layer_block_codes = {
            E_WEIGHT_INVALID,
            E_WEIGHT_SHAPE,
            E_WEIGHT_DUPLICATE,
            E_GROUP_INVALID,
            E_QUANT_BOUNDS,
            E_VERSION_ORDER,
        }
        global_block_codes = {E_CONFIG_INVALID, E_CALIBRATION_INVALID, E_QUANT_BOUNDS, E_GROUP_INVALID}
        for failure in failures.items:
            if failure.layer in layer_names and failure.code in layer_block_codes:
                blocked_layers.add(failure.layer)  # type: ignore[arg-type]
            elif failure.layer is None and failure.code in global_block_codes:
                global_block = True

        runnable_layers = [
            layer for layer in dataset.valid_layers if layer.name not in blocked_layers
        ]
        config = dataset.config
        calibration = dataset.calibration
        plan = (
            plan_budget(runnable_layers, config)
            if config is not None and calibration is not None and not global_block
            else BudgetPlan()
        )
        for pending in plan.pending:
            failures.add(
                pending.code,
                pending.reason,
                "scheduling",
                layer=pending.layer_name,
                details={
                    "group_index": pending.group_index,
                    "start": pending.start,
                    "end": pending.end,
                    "points": pending.num_points,
                },
            )

        config_hash = config.fingerprint if config is not None else stable_hash(None)
        global_version = self._global_version(dataset, config_hash)
        request = self.request
        report: dict[str, Any]

        with StateStore(Path(request.state_dir), full_rebuild=request.full_rebuild) as store:
            previous_manifest = store.previous_manifest
            version_check = self._version_check(
                previous_manifest,
                global_version=global_version,
                config_hash=config_hash,
                dataset=dataset,
                full_rebuild=request.full_rebuild,
            )

            calibration_results: dict[tuple[str, int], StageTaskResult] = {}
            quantization_results: dict[tuple[str, int], StageTaskResult] = {}
            error_results: dict[tuple[str, int], StageTaskResult] = {}
            distillation_results: dict[tuple[str, int], StageTaskResult] = {}
            cache_events: list[dict[str, Any]] = []

            loading_key = stable_hash(
                {
                    "stage": "loading_validation",
                    "stage_version": STAGE_VERSION,
                    "global_version": global_version,
                    "config_hash": config_hash,
                }
            )
            loading_path = store.artifact_path("loading")
            cached_loading = store.read_artifact(loading_path)
            loading_reused = bool(cached_loading and cached_loading.get("cache_key") == loading_key)
            loading_payload = self._loading_payload(dataset, plan, failures)
            if not loading_reused:
                store.write_artifact(
                    loading_path,
                    self._artifact(
                        "loading",
                        loading_key,
                        loading_payload,
                        layer=None,
                        group_index=None,
                        calibration_sample_ids=calibration.sample_ids if calibration else (),
                    ),
                )
            cache_events.append(
                {
                    "stage": "loading",
                    "layer": None,
                    "group_index": None,
                    "reused": loading_reused,
                    "computed": not loading_reused,
                }
            )
            interim_manifest = self._manifest(
                dataset,
                store,
                global_version=global_version,
                config_hash=config_hash,
                plan=plan,
                completed_layers=[],
                partial_layers={},
                layer_state={},
                version_check=version_check,
                stage_completed_layers={},
            )
            store.commit(interim_manifest, "loading")

            if not global_block and config is not None and calibration is not None:
                calibration_results = self._run_calibration(
                    dataset,
                    config,
                    calibration,
                    plan,
                    store,
                    failures,
                    cache_events,
                )
                calibrated_layers = self._layers_complete_at_stage(plan, calibration_results)
                manifest = self._manifest(
                    dataset,
                    store,
                    global_version=global_version,
                    config_hash=config_hash,
                    plan=plan,
                    completed_layers=sorted(calibrated_layers),
                    partial_layers={},
                    layer_state={},
                    version_check=version_check,
                    stage_completed_layers={"calibration": sorted(calibrated_layers)},
                )
                store.commit(manifest, "calibration")

                quantization_results = self._run_quantization(
                    dataset,
                    config,
                    plan,
                    store,
                    calibration_results,
                    failures,
                    cache_events,
                    global_version,
                    config_hash,
                    version_check,
                )
                manifest = self._manifest(
                    dataset,
                    store,
                    global_version=global_version,
                    config_hash=config_hash,
                    plan=plan,
                    completed_layers=sorted(
                        set(calibrated_layers)
                        & self._layers_complete_at_stage(plan, quantization_results)
                    ),
                    partial_layers={},
                    layer_state={},
                    version_check=version_check,
                    stage_completed_layers={
                        "calibration": sorted(calibrated_layers),
                        "quantization": sorted(self._layers_complete_at_stage(plan, quantization_results)),
                    },
                )
                store.commit(manifest, "quantization")

                error_results, distillation_results = self._run_error_and_distillation(
                    dataset,
                    config,
                    calibration,
                    plan,
                    store,
                    quantization_results,
                    failures,
                    cache_events,
                )
                error_complete = self._layers_complete_at_stage(plan, error_results)
                distill_complete = self._layers_complete_at_stage(plan, distillation_results)
                manifest = self._manifest(
                    dataset,
                    store,
                    global_version=global_version,
                    config_hash=config_hash,
                    plan=plan,
                    completed_layers=sorted(calibrated_layers & error_complete & distill_complete),
                    partial_layers={},
                    layer_state={},
                    version_check=version_check,
                    stage_completed_layers={
                        "calibration": sorted(calibrated_layers),
                        "quantization": sorted(self._layers_complete_at_stage(plan, quantization_results)),
                        "error_analysis": sorted(error_complete),
                        "distillation": sorted(distill_complete),
                    },
                )
                store.commit(manifest, "error_analysis")
                store.commit(manifest, "distillation")

            (
                ledger_payload,
                ledger_key,
                ledger_reused,
                layer_state,
                completed_layers,
                partial_layers,
                invalidation,
            ) = self._build_ledger_stage(
                dataset,
                config,
                calibration,
                plan,
                calibration_results,
                quantization_results,
                error_results,
                distillation_results,
                store,
                cache_events,
                failures,
                previous_manifest,
                global_version,
                config_hash,
            )
            manifest = self._manifest(
                dataset,
                store,
                global_version=global_version,
                config_hash=config_hash,
                plan=plan,
                completed_layers=completed_layers,
                partial_layers=partial_layers,
                layer_state=layer_state,
                version_check=version_check,
                stage_completed_layers={
                    "calibration": sorted(self._layers_complete_at_stage(plan, calibration_results)),
                    "quantization": sorted(self._layers_complete_at_stage(plan, quantization_results)),
                    "error_analysis": sorted(self._layers_complete_at_stage(plan, error_results)),
                    "distillation": sorted(self._layers_complete_at_stage(plan, distillation_results)),
                    "receipt_ledger": [] if ledger_payload is None else [layer.name for layer in runnable_layers],
                },
            )
            store.commit(manifest, "receipt_ledger")

            report = self._make_report(
                dataset=dataset,
                config=config,
                calibration=calibration,
                plan=plan,
                failures=failures,
                fault_results=fault_results,
                commands_applied=commands_applied,
                global_version=global_version,
                config_hash=config_hash,
                version_check=version_check,
                invalidation=invalidation,
                cache_events=cache_events,
                calibration_results=calibration_results,
                quantization_results=quantization_results,
                error_results=error_results,
                distillation_results=distillation_results,
                ledger_payload=ledger_payload,
                ledger_key=ledger_key,
                ledger_reused=ledger_reused,
                completed_layers=completed_layers,
                partial_layers=partial_layers,
                blocked_layers=sorted(blocked_layers),
                global_block=global_block,
                store=store,
                manifest=manifest,
            )

        if request.report_path is not None:
            write_json(Path(request.report_path), report)
        return report

    def _global_version(self, dataset: Dataset, config_hash: str) -> str:
        return stable_hash(
            {
                "schema": "quantcalib-global-v1",
                "weights": [layer.content_payload() for layer in dataset.layers],
                "calibration": dataset.calibration.payload() if dataset.calibration else None,
                "config": config_hash,
                "config_raw": dataset.config.raw if dataset.config else None,
                "receipts": dataset.receipts.fingerprint_payload(),
            }
        )

    def _version_check(
        self,
        previous_manifest: dict[str, Any] | None,
        *,
        global_version: str,
        config_hash: str,
        dataset: Dataset,
        full_rebuild: bool,
    ) -> dict[str, Any]:
        if previous_manifest is None:
            return {
                "previous_global_version": None,
                "current_global_version": global_version,
                "previous_config_hash": None,
                "current_config_hash": config_hash,
                "global_version_changed": True,
                "config_hash_changed": True,
                "weight_package_changed": True,
                "calibration_set_changed": True,
                "comparison_performed_before_commit": True,
                "full_rebuild": full_rebuild,
                "previous_snapshot": None,
                "previous_completed_layers": [],
                "previous_partial_layers": {},
                "resume_from_existing_generation": False,
                "policy": "content versions and config hash are checked before stage commit; per-layer cache keys determine exact invalidation",
            }
        return {
            "previous_global_version": previous_manifest.get("global_version"),
            "current_global_version": global_version,
            "previous_config_hash": previous_manifest.get("config_hash"),
            "current_config_hash": config_hash,
            "global_version_changed": previous_manifest.get("global_version") != global_version,
            "config_hash_changed": previous_manifest.get("config_hash") != config_hash,
            "weight_package_changed": previous_manifest.get("weight_package_fingerprint")
            != dataset.weight_fingerprint,
            "calibration_set_changed": previous_manifest.get("calibration_fingerprint")
            != dataset.calibration_fingerprint,
            "comparison_performed_before_commit": True,
            "full_rebuild": full_rebuild,
            "previous_snapshot": previous_manifest.get("snapshot"),
            "previous_completed_layers": previous_manifest.get("completed_layers", []),
            "previous_partial_layers": previous_manifest.get("partial_layers", {}),
            "resume_from_existing_generation": not full_rebuild,
            "policy": "content versions and config hash are checked before stage commit; per-layer cache keys determine exact invalidation",
        }

    def _artifact(
        self,
        stage: str,
        cache_key: str,
        payload: dict[str, Any],
        *,
        layer: str | None,
        group_index: int | None,
        calibration_sample_ids: Iterable[str],
        role_fingerprint: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        artifact = {
            "schema": "quantcalib-stage-artifact-v1",
            "stage": stage,
            "stage_version": STAGE_VERSION,
            "cache_key": cache_key,
            "layer": layer,
            "group_index": group_index,
            "calibration_sample_ids": list(calibration_sample_ids),
            "role_fingerprint": role_fingerprint,
            "payload": payload,
        }
        if extra:
            artifact["extra"] = extra
        return artifact

    def _cache_lookup(
        self, store: StateStore, path: Path, cache_key: str, stage: str
    ) -> tuple[dict[str, Any] | None, bool]:
        artifact = store.read_artifact(path)
        if (
            artifact is not None
            and artifact.get("cache_key") == cache_key
            and artifact.get("stage") == stage
            and artifact.get("stage_version") == STAGE_VERSION
            and isinstance(artifact.get("payload"), dict)
        ):
            return artifact["payload"], True
        return None, False

    def _loading_payload(
        self, dataset: Dataset, plan: BudgetPlan, failures: FailureList
    ) -> dict[str, Any]:
        return {
            "schema": "quantcalib-loading-validation-v1",
            "weight_layer_count": len(dataset.layers),
            "layer_names": [layer.name for layer in dataset.layers],
            "calibration_sample_count": len(dataset.calibration.samples) if dataset.calibration else 0,
            "calibration_sample_ids": list(dataset.calibration.sample_ids) if dataset.calibration else [],
            "receipt_count": len(dataset.receipts.receipts),
            "failure_count": len(failures.items),
            "scheduled_group_count": len(plan.scheduled),
            "pending_group_count": len(plan.pending),
        }

    def _run_calibration(
        self,
        dataset: Dataset,
        config: Config,
        calibration: Any,
        plan: BudgetPlan,
        store: StateStore,
        failures: FailureList,
        cache_events: list[dict[str, Any]],
    ) -> dict[tuple[str, int], StageTaskResult]:
        layer_map = {layer.name: layer for layer in dataset.layers}

        def task(group: ScheduledGroup) -> StageTaskResult:
            layer = layer_map[group.layer_name]
            cfg = config.for_layer(layer)
            cache_key = stable_hash(
                {
                    "stage": "group_calibration",
                    "stage_version": STAGE_VERSION,
                    "layer_content": layer.content_fingerprint,
                    "group_index": group.group_index,
                    "range": [group.start, group.end],
                    "config": cfg.fingerprint_payload(),
                    "calibration": calibration.fingerprint,
                }
            )
            path = store.artifact_path("calibration", layer.name, group.group_index)
            payload, reused = self._cache_lookup(store, path, cache_key, "group_calibration")
            try:
                if payload is None:
                    payload = calibrate_group(
                        layer,
                        cfg,
                        group.group_index,
                        calibration_sample_ids=calibration.sample_ids,
                    )
                    store.write_artifact(
                        path,
                        self._artifact(
                            "group_calibration",
                            cache_key,
                            payload,
                            layer=layer.name,
                            group_index=group.group_index,
                            calibration_sample_ids=calibration.sample_ids,
                            role_fingerprint=payload["role_fingerprint"],
                        ),
                    )
                return StageTaskResult(
                    layer.name,
                    group.group_index,
                    payload,
                    cache_key,
                    reused,
                    not reused,
                    path,
                )
            except Exception as exc:  # pragma: no cover - defensive task boundary
                return StageTaskResult(
                    layer.name,
                    group.group_index,
                    None,
                    cache_key,
                    reused,
                    False,
                    path,
                    error=str(exc),
                )

        results: dict[tuple[str, int], StageTaskResult] = {}
        with ThreadPoolExecutor(max_workers=config.scheduling.thread_pool_workers) as executor:
            for result in executor.map(task, plan.scheduled):
                self._record_task(result, "calibration", failures, cache_events)
                if result.payload is not None:
                    results[(result.layer, result.group_index)] = result
        return results

    def _run_quantization(
        self,
        dataset: Dataset,
        config: Config,
        plan: BudgetPlan,
        store: StateStore,
        calibration_results: dict[tuple[str, int], StageTaskResult],
        failures: FailureList,
        cache_events: list[dict[str, Any]],
        global_version: str,
        config_hash: str,
        version_check: dict[str, Any],
    ) -> dict[tuple[str, int], StageTaskResult]:
        layer_map = {layer.name: layer for layer in dataset.layers}

        def task(group: ScheduledGroup) -> StageTaskResult:
            layer = layer_map[group.layer_name]
            cfg = config.for_layer(layer)
            calibration_result = calibration_results.get((layer.name, group.group_index))
            if calibration_result is None or calibration_result.payload is None:
                return StageTaskResult(
                    layer.name,
                    group.group_index,
                    None,
                    "",
                    False,
                    False,
                    error="calibration result unavailable",
                )
            cache_key = stable_hash(
                {
                    "stage": "group_quantization",
                    "stage_version": STAGE_VERSION,
                    "calibration_key": calibration_result.cache_key,
                    "layer_content": layer.content_fingerprint,
                    "config": cfg.fingerprint_payload(),
                    "batch_point_budget": config.scheduling.batch_quantization_point_budget,
                }
            )
            path = store.artifact_path("quantization", layer.name, group.group_index)
            payload, reused = self._cache_lookup(store, path, cache_key, "group_quantization")
            try:
                if payload is None:
                    payload = quantize_group(
                        layer,
                        cfg,
                        calibration_result.payload,
                        batch_point_budget=config.scheduling.batch_quantization_point_budget,
                        calibration_sample_ids=list(calibration_result.payload["calibration_sample_ids"]),
                    )
                    store.write_artifact(
                        path,
                        self._artifact(
                            "group_quantization",
                            cache_key,
                            payload,
                            layer=layer.name,
                            group_index=group.group_index,
                            calibration_sample_ids=payload["calibration_sample_ids"],
                            role_fingerprint=payload["role_fingerprint"],
                            extra={"role": payload["role"]},
                        ),
                    )
                return StageTaskResult(
                    layer.name,
                    group.group_index,
                    payload,
                    cache_key,
                    reused,
                    not reused,
                    path,
                    extra={"role": payload.get("role")},
                )
            except Exception as exc:  # pragma: no cover
                return StageTaskResult(
                    layer.name,
                    group.group_index,
                    None,
                    cache_key,
                    reused,
                    False,
                    path,
                    error=str(exc),
                )

        results: dict[tuple[str, int], StageTaskResult] = {}
        interval = config.scheduling.snapshot_commit_interval_batches
        with ThreadPoolExecutor(max_workers=config.scheduling.thread_pool_workers) as executor:
            for batch_index, batch in enumerate(plan.execution_batches, 1):
                for result in executor.map(task, batch):
                    self._record_task(result, "quantization", failures, cache_events)
                    if result.payload is not None:
                        results[(result.layer, result.group_index)] = result
                if batch_index % interval == 0:
                    manifest = self._manifest(
                        dataset,
                        store,
                        global_version=global_version,
                        config_hash=config_hash,
                        plan=plan,
                        completed_layers=[],
                        partial_layers={},
                        layer_state={},
                        version_check=version_check,
                        stage_completed_layers={
                            "quantization": sorted(self._layers_complete_at_stage(plan, results))
                        },
                    )
                    store.commit(manifest, f"quantization_batch_{batch_index}")
        return results

    def _run_error_and_distillation(
        self,
        dataset: Dataset,
        config: Config,
        calibration: Any,
        plan: BudgetPlan,
        store: StateStore,
        quantization_results: dict[tuple[str, int], StageTaskResult],
        failures: FailureList,
        cache_events: list[dict[str, Any]],
    ) -> tuple[dict[tuple[str, int], StageTaskResult], dict[tuple[str, int], StageTaskResult]]:
        layer_map = {layer.name: layer for layer in dataset.layers}

        def task(group: ScheduledGroup) -> tuple[StageTaskResult, StageTaskResult]:
            layer = layer_map[group.layer_name]
            cfg = config.for_layer(layer)
            quantized = quantization_results.get((layer.name, group.group_index))
            if quantized is None or quantized.payload is None:
                error_result = StageTaskResult(
                    layer.name,
                    group.group_index,
                    None,
                    "",
                    False,
                    False,
                    error="quantization result unavailable",
                )
                return error_result, StageTaskResult(
                    layer.name,
                    group.group_index,
                    None,
                    "",
                    False,
                    False,
                    error="error result unavailable",
                )
            error_key = stable_hash(
                {
                    "stage": "group_error_analysis",
                    "stage_version": STAGE_VERSION,
                    "quantization_key": quantized.cache_key,
                    "calibration": calibration.fingerprint,
                    "tolerance": config.distillation.tolerance,
                }
            )
            error_path = store.artifact_path("error_analysis", layer.name, group.group_index)
            error_payload, error_reused = self._cache_lookup(
                store, error_path, error_key, "group_error_analysis"
            )
            try:
                if error_payload is None:
                    calibration_payload = self._read_calibration_for_group(
                        store, layer, group.group_index
                    )
                    if calibration_payload is None:
                        raise ValueError("calibration artifact unavailable")
                    error_payload = analyze_group(
                        layer=layer,
                        cfg=cfg,
                        calibration=calibration,
                        calibration_group=calibration_payload,
                        quantization_group=quantized.payload,
                        distill_cfg=config.distillation,
                    )
                    store.write_artifact(
                        error_path,
                        self._artifact(
                            "group_error_analysis",
                            error_key,
                            error_payload,
                            layer=layer.name,
                            group_index=group.group_index,
                            calibration_sample_ids=calibration.sample_ids,
                            role_fingerprint=quantized.payload.get("role_fingerprint"),
                        ),
                    )
                error_result = StageTaskResult(
                    layer.name,
                    group.group_index,
                    error_payload,
                    error_key,
                    error_reused,
                    not error_reused,
                    error_path,
                )
            except Exception as exc:  # pragma: no cover
                error_result = StageTaskResult(
                    layer.name,
                    group.group_index,
                    None,
                    error_key,
                    error_reused,
                    False,
                    error_path,
                    error=str(exc),
                )
                return error_result, StageTaskResult(
                    layer.name,
                    group.group_index,
                    None,
                    "",
                    False,
                    False,
                    error="error analysis failed",
                )

            distill_key = stable_hash(
                {
                    "stage": "group_distillation",
                    "stage_version": STAGE_VERSION,
                    "error_key": error_key,
                    "calibration": calibration.fingerprint,
                    "distillation": config.distillation.fingerprint_payload(),
                }
            )
            distill_path = store.artifact_path("distillation", layer.name, group.group_index)
            distill_payload, distill_reused = self._cache_lookup(
                store, distill_path, distill_key, "group_distillation"
            )
            try:
                if distill_payload is None:
                    distill_payload = build_group_distillation(
                        layer=layer,
                        calibration=calibration,
                        error_analysis=error_payload,
                        distill_cfg=config.distillation,
                    )
                    store.write_artifact(
                        distill_path,
                        self._artifact(
                            "group_distillation",
                            distill_key,
                            distill_payload,
                            layer=layer.name,
                            group_index=group.group_index,
                            calibration_sample_ids=calibration.sample_ids,
                            role_fingerprint=stable_hash(config.distillation.fingerprint_payload()),
                        ),
                    )
                distill_result = StageTaskResult(
                    layer.name,
                    group.group_index,
                    distill_payload,
                    distill_key,
                    distill_reused,
                    not distill_reused,
                    distill_path,
                )
            except Exception as exc:  # pragma: no cover
                distill_result = StageTaskResult(
                    layer.name,
                    group.group_index,
                    None,
                    distill_key,
                    distill_reused,
                    False,
                    distill_path,
                    error=str(exc),
                )
            return error_result, distill_result

        error_results: dict[tuple[str, int], StageTaskResult] = {}
        distillation_results: dict[tuple[str, int], StageTaskResult] = {}
        with ThreadPoolExecutor(max_workers=config.scheduling.thread_pool_workers) as executor:
            for error_result, distill_result in executor.map(task, plan.scheduled):
                self._record_task(error_result, "error_analysis", failures, cache_events)
                self._record_task(distill_result, "distillation", failures, cache_events)
                if error_result.payload is not None:
                    error_results[(error_result.layer, error_result.group_index)] = error_result
                if distill_result.payload is not None:
                    distillation_results[
                        (distill_result.layer, distill_result.group_index)
                    ] = distill_result
        return error_results, distillation_results

    def _read_calibration_for_group(
        self, store: StateStore, layer: Layer, group_index: int
    ) -> dict[str, Any] | None:
        path = store.artifact_path("calibration", layer.name, group_index)
        artifact = store.read_artifact(path)
        if artifact is None or not isinstance(artifact.get("payload"), dict):
            return None
        return artifact["payload"]

    def _record_task(
        self,
        result: StageTaskResult,
        stage: str,
        failures: FailureList,
        cache_events: list[dict[str, Any]],
    ) -> None:
        if result.error is not None:
            failures.add(
                E_INTERNAL,
                f"{stage} group failed: {result.error}",
                stage,
                layer=result.layer,
                details={"group_index": result.group_index},
            )
        cache_events.append(
            {
                "stage": stage,
                "layer": result.layer,
                "group_index": result.group_index,
                "reused": result.reused,
                "computed": result.computed,
                "cache_key": result.cache_key,
                "artifact_path": str(result.artifact_path) if result.artifact_path else None,
            }
        )

    def _layers_complete_at_stage(
        self,
        plan: BudgetPlan,
        results: dict[tuple[str, int], StageTaskResult],
    ) -> set[str]:
        scheduled_by_layer = plan.by_layer()
        pending_by_layer = plan.pending_by_layer()
        complete: set[str] = set()
        for layer_name, groups in scheduled_by_layer.items():
            if layer_name not in pending_by_layer and all(
                (layer_name, group.group_index) in results for group in groups
            ):
                complete.add(layer_name)
        return complete

    def _build_ledger_stage(
        self,
        dataset: Dataset,
        config: Config | None,
        calibration: Any,
        plan: BudgetPlan,
        calibration_results: dict[tuple[str, int], StageTaskResult],
        quantization_results: dict[tuple[str, int], StageTaskResult],
        error_results: dict[tuple[str, int], StageTaskResult],
        distillation_results: dict[tuple[str, int], StageTaskResult],
        store: StateStore,
        cache_events: list[dict[str, Any]],
        failures: FailureList,
        previous_manifest: dict[str, Any] | None,
        global_version: str,
        config_hash: str,
    ) -> tuple[
        dict[str, Any] | None,
        str,
        bool,
        dict[str, Any],
        list[str],
        dict[str, str],
        dict[str, Any],
    ]:
        layer_state: dict[str, Any] = {}
        prior_layer_state = (
            {} if self.request.full_rebuild else (previous_manifest or {}).get("layer_state", {})
        )
        invalidation: dict[str, Any] = {"layers": {}, "removed_layers": [], "receipt_ledger": {}}
        scheduled_by_layer = plan.by_layer()
        for layer in dataset.valid_layers:
            groups = scheduled_by_layer.get(layer.name, [])
            group_indices = sorted(group.group_index for group in groups)
            calibration_keys = [
                calibration_results[(layer.name, index)].cache_key
                for index in group_indices
                if (layer.name, index) in calibration_results
            ]
            quantization_keys = [
                quantization_results[(layer.name, index)].cache_key
                for index in group_indices
                if (layer.name, index) in quantization_results
            ]
            error_keys = [
                error_results[(layer.name, index)].cache_key
                for index in group_indices
                if (layer.name, index) in error_results
            ]
            distillation_keys = [
                distillation_results[(layer.name, index)].cache_key
                for index in group_indices
                if (layer.name, index) in distillation_results
            ]
            calibration_refs = sorted(
                {
                    sample_id
                    for index in group_indices
                    if (layer.name, index) in calibration_results
                    for sample_id in calibration_results[(layer.name, index)].payload.get(
                        "calibration_sample_ids", []
                    )
                }
            )
            role_fingerprints = sorted(
                {
                    quantization_results[(layer.name, index)].payload.get("role_fingerprint")
                    for index in group_indices
                    if (layer.name, index) in quantization_results
                    and quantization_results[(layer.name, index)].payload
                }
            )
            dependency = {
                "layer": layer.name,
                "content_version": layer.content_version,
                "content_fingerprint": layer.content_fingerprint,
                "group_indices": group_indices,
                "group_count": layer.numel // config.for_layer(layer).group_granularity if config else None,
                "calibration_cache_keys": calibration_keys,
                "quantization_cache_keys": quantization_keys,
                "error_cache_keys": error_keys,
                "distillation_cache_keys": distillation_keys,
                "calibration_sample_ids": calibration_refs,
                "quantization_role_fingerprints": role_fingerprints,
            }
            layer_state[layer.name] = dependency
            prior = prior_layer_state.get(layer.name)
            if not isinstance(prior, dict):
                invalidation["layers"][layer.name] = {
                    "content_version_changed": True,
                    "initialized": True,
                    "invalidated_stages": [
                        "calibration",
                        "quantization",
                        "error_analysis",
                        "distillation",
                        "receipt_ledger",
                    ],
                    "reused_stages": [],
                }
                continue
            content_changed = prior.get("content_fingerprint") != layer.content_fingerprint
            stage_pairs = {
                "calibration": (prior.get("calibration_cache_keys"), calibration_keys),
                "quantization": (prior.get("quantization_cache_keys"), quantization_keys),
                "error_analysis": (prior.get("error_cache_keys"), error_keys),
                "distillation": (
                    prior.get("distillation_cache_keys"),
                    distillation_keys,
                ),
            }
            invalidated = [stage for stage, (old, new) in stage_pairs.items() if old != new]
            reused = [stage for stage, (old, new) in stage_pairs.items() if old == new]
            if content_changed and "calibration" not in invalidated:
                invalidated.insert(0, "calibration")
            if invalidated and "receipt_ledger" not in invalidated:
                invalidated.append("receipt_ledger")
            invalidation["layers"][layer.name] = {
                "content_version_changed": content_changed,
                "previous_content_version": prior.get("content_version"),
                "current_content_version": layer.content_version,
                "invalidated_stages": invalidated,
                "reused_stages": reused,
                "boundary_rule": "layer content -> calibration -> quantization -> error/distillation -> receipt ledger",
            }

        invalidation["removed_layers"] = sorted(
            set(prior_layer_state) - set(layer_state)
        )
        ledger_cache_key = stable_hash(
            {
                "stage": "receipt_ledger",
                "stage_version": STAGE_VERSION,
                "receipts": dataset.receipts.fingerprint,
                "layers": layer_state,
                "calibration": calibration.fingerprint if calibration else None,
                "config": config.fingerprint if config else None,
                "global_version": global_version,
            }
        )
        ledger_path = store.artifact_path("receipt_ledger")
        ledger_payload, ledger_reused = self._cache_lookup(
            store, ledger_path, ledger_cache_key, "receipt_ledger"
        )
        try:
            if ledger_payload is None:
                ledger_payload = build_receipt_ledger(dataset.receipts)
                for entry in ledger_payload["entries"]:
                    entry["dependency"] = layer_state.get(entry.get("layer"))
                store.write_artifact(
                    ledger_path,
                    self._artifact(
                        "receipt_ledger",
                        ledger_cache_key,
                        ledger_payload,
                        layer=None,
                        group_index=None,
                        calibration_sample_ids=calibration.sample_ids if calibration else (),
                        role_fingerprint=stable_hash(dataset.receipts.fingerprint_payload()),
                    ),
                )
            cache_events.append(
                {
                    "stage": "receipt_ledger",
                    "layer": None,
                    "group_index": None,
                    "reused": ledger_reused,
                    "computed": not ledger_reused,
                    "cache_key": ledger_cache_key,
                }
            )
            invalidation["receipt_ledger"] = {
                "cache_key": ledger_cache_key,
                "reused": ledger_reused,
                "invalidated": not ledger_reused,
                "downstream_of_layers": sorted(invalidation["layers"]),
                "removed_layers": invalidation["removed_layers"],
            }
        except Exception as exc:  # pragma: no cover
            failures.add(E_INTERNAL, f"receipt ledger failed: {exc}", "receipt_ledger")
            ledger_payload = None

        final_complete: set[str] = set()
        for layer_name, groups in scheduled_by_layer.items():
            if not groups:
                continue
            if all((layer_name, group.group_index) in distillation_results for group in groups):
                final_complete.add(layer_name)
        partial: dict[str, str] = {}
        for layer in dataset.valid_layers:
            if layer.name in blocked_layers_from_failures(failures):
                partial[layer.name] = "validation_blocked"
            elif layer.name not in final_complete:
                pending = [item for item in plan.pending if item.layer_name == layer.name]
                partial[layer.name] = pending[0].code if pending else "incomplete"
        return (
            ledger_payload,
            ledger_cache_key,
            ledger_reused,
            layer_state,
            sorted(final_complete),
            dict(sorted(partial.items())),
            invalidation,
        )

    def _manifest(
        self,
        dataset: Dataset,
        store: StateStore,
        *,
        global_version: str,
        config_hash: str,
        plan: BudgetPlan,
        completed_layers: list[str],
        partial_layers: dict[str, str],
        layer_state: dict[str, Any],
        version_check: dict[str, Any],
        stage_completed_layers: dict[str, list[str]],
    ) -> dict[str, Any]:
        return {
            "schema": "quantcalib-manifest-v1",
            "global_version": global_version,
            "config_hash": config_hash,
            "weight_package_fingerprint": dataset.weight_fingerprint,
            "calibration_fingerprint": dataset.calibration_fingerprint,
            "receipts_fingerprint": dataset.receipts.fingerprint,
            "completed_layers": completed_layers,
            "partial_layers": partial_layers,
            "stage_completed_layers": stage_completed_layers,
            "consumed_budget": {
                "global": plan.global_consumed,
                "per_layer": dict(sorted(plan.per_layer_consumed.items())),
                "batch_count": len(plan.batch_events),
            },
            "pending_items": [
                {
                    "layer": item.layer_name,
                    "group_index": item.group_index,
                    "code": item.code,
                }
                for item in plan.pending
            ],
            "layer_state": layer_state,
            "version_check": version_check,
            "snapshots_in_run": list(store.committed_snapshots),
        }

    def _make_report(
        self,
        *,
        dataset: Dataset,
        config: Config | None,
        calibration: Any,
        plan: BudgetPlan,
        failures: FailureList,
        fault_results: list[dict[str, Any]],
        commands_applied: list[dict[str, Any]],
        global_version: str,
        config_hash: str,
        version_check: dict[str, Any],
        invalidation: dict[str, Any],
        cache_events: list[dict[str, Any]],
        calibration_results: dict[tuple[str, int], StageTaskResult],
        quantization_results: dict[tuple[str, int], StageTaskResult],
        error_results: dict[tuple[str, int], StageTaskResult],
        distillation_results: dict[tuple[str, int], StageTaskResult],
        ledger_payload: dict[str, Any] | None,
        ledger_key: str,
        ledger_reused: bool,
        completed_layers: list[str],
        partial_layers: dict[str, str],
        blocked_layers: list[str],
        global_block: bool,
        store: StateStore,
        manifest: dict[str, Any],
    ) -> dict[str, Any]:
        layers_report: list[dict[str, Any]] = []
        attribution_table: list[dict[str, Any]] = []
        cache_by_layer: dict[str, dict[str, Any]] = {}
        for event in cache_events:
            if event.get("layer") is None:
                continue
            per_layer = cache_by_layer.setdefault(str(event["layer"]), {})
            stage = str(event["stage"])
            bucket = per_layer.setdefault(stage, {"computed": 0, "reused": 0, "groups": []})
            bucket["computed" if event.get("computed") else "reused"] += 1
            bucket["groups"].append(
                {
                    "group_index": event.get("group_index"),
                    "reused": bool(event.get("reused")),
                    "computed": bool(event.get("computed")),
                }
            )

        for layer in dataset.valid_layers:
            if layer.name in blocked_layers or global_block:
                layers_report.append(
                    {
                        "layer": layer.name,
                        "content_version": layer.content_version,
                        "content_fingerprint": layer.content_fingerprint,
                        "status": "validation_blocked",
                    }
                )
                continue
            cfg = config.for_layer(layer) if config is not None else None
            scheduled = plan.by_layer().get(layer.name, [])
            group_summaries: list[dict[str, Any]] = []
            total_saturation = 0
            calibration_hashes: list[str] = []
            quant_hashes: list[str] = []
            attribution_counts = {name: 0 for name in ATTRIBUTION_PRIORITY}
            pre_exceeded = 0
            post_exceeded = 0
            total_records = 0
            aggregate_outputs: dict[str, dict[str, list[float]]] = {}
            for group in scheduled:
                key = (layer.name, group.group_index)
                cal = calibration_results.get(key)
                quant = quantization_results.get(key)
                err = error_results.get(key)
                dist = distillation_results.get(key)
                summary: dict[str, Any] = {
                    "group_index": group.group_index,
                    "start": group.start,
                    "end": group.end,
                    "points": group.num_points,
                    "calibration_reused": bool(cal.reused) if cal else False,
                    "quantization_reused": bool(quant.reused) if quant else False,
                    "error_analysis_reused": bool(err.reused) if err else False,
                    "distillation_reused": bool(dist.reused) if dist else False,
                }
                if cal and cal.payload:
                    summary["calibration"] = {
                        "lower_quantile": cal.payload.get("lower_quantile"),
                        "upper_quantile": cal.payload.get("upper_quantile"),
                        "scale": cal.payload.get("scale"),
                        "zero": cal.payload.get("zero"),
                        "scale_base": cal.payload.get("scale_base"),
                        "role_fingerprint": cal.payload.get("role_fingerprint"),
                        "artifact_hash": stable_hash(cal.payload),
                        "percentile_metadata": cal.payload.get("percentile_metadata"),
                    }
                    calibration_hashes.append(stable_hash(cal.payload))
                if quant and quant.payload:
                    summary["quantization"] = {
                        "fixed_integer_hash": stable_hash(quant.payload.get("fixed_integers")),
                        "dequantized_hash": stable_hash(quant.payload.get("dequantized")),
                        "saturation_count": quant.payload.get("saturation_count"),
                        "truncations": quant.payload.get("truncations", []),
                        "role_fingerprint": quant.payload.get("role_fingerprint"),
                        "role": quant.payload.get("role"),
                        "artifact_hash": stable_hash(quant.payload),
                    }
                    total_saturation += int(quant.payload.get("saturation_count", 0))
                    quant_hashes.append(stable_hash(quant.payload))
                if err and err.payload:
                    summary["error_analysis"] = {
                        "pre_exceeded_count": err.payload.get("pre_exceeded_count", 0),
                        "post_exceeded_count": err.payload.get("post_exceeded_count", 0),
                        "attribution_counts": err.payload.get("attribution_counts", {}),
                        "zero_truth_count": err.payload.get("zero_truth_count", 0),
                        "artifact_hash": stable_hash(err.payload),
                    }
                    pre_exceeded += int(err.payload.get("pre_exceeded_count", 0))
                    post_exceeded += int(err.payload.get("post_exceeded_count", 0))
                    for source, count in err.payload.get("attribution_counts", {}).items():
                        attribution_counts[source] = attribution_counts.get(source, 0) + int(count)
                    for record in err.payload.get("records", []):
                        total_records += 1
                        attribution_table.append(
                            {
                                "layer": layer.name,
                                "group_index": group.group_index,
                                **record,
                            }
                        )
                if dist and dist.payload:
                    summary["distillation"] = {
                        "artifact_hash": stable_hash(dist.payload),
                        "samples": dist.payload.get("samples", []),
                    }
                    for sample in dist.payload.get("samples", []):
                        bucket = aggregate_outputs.setdefault(
                            sample["sample_id"],
                            {"raw": [], "quantized": [], "corrected": []},
                        )
                        if not bucket["raw"]:
                            size = len(sample["raw_output"])
                            bucket["raw"] = [0.0] * size
                            bucket["quantized"] = [0.0] * size
                            bucket["corrected"] = [0.0] * size
                        for index in range(len(bucket["raw"])):
                            bucket["raw"][index] += float(sample["raw_output"][index])
                            bucket["quantized"][index] += float(sample["quantized_output"][index])
                            bucket["corrected"][index] += float(sample["corrected_output"][index])
                group_summaries.append(summary)

            aggregate: list[dict[str, Any]] = []
            if calibration is not None:
                sample_map = {sample.sample_id: sample for sample in calibration.samples}
                for sample_id in sorted(aggregate_outputs):
                    bucket = aggregate_outputs[sample_id]
                    raw = [clean_float(value, 17) for value in bucket["raw"]]
                    quantized = [clean_float(value, 17) for value in bucket["quantized"]]
                    corrected = [clean_float(value, 17) for value in bucket["corrected"]]
                    reference = list(sample_map[sample_id].reference_output)
                    aggregate.append(
                        {
                            "sample_id": sample_id,
                            "raw_output": raw,
                            "quantized_output": quantized,
                            "corrected_output": corrected,
                            "reference_output": reference,
                            "pre_deviation": [
                                clean_float(q - r, 17) for q, r in zip(quantized, raw)
                            ],
                            "post_deviation": [
                                clean_float(c - r, 17) for c, r in zip(corrected, raw)
                            ],
                            "reference_deviation": [
                                clean_float(r - ref, 17) for r, ref in zip(raw, reference)
                            ],
                            "calibration_sample_ids": [sample_id],
                        }
                    )

            layer_cache = cache_by_layer.get(layer.name, {})
            layers_report.append(
                {
                    "layer": layer.name,
                    "content_version": layer.content_version,
                    "content_fingerprint": layer.content_fingerprint,
                    "numel": layer.numel,
                    "group_granularity": cfg.group_granularity if cfg else None,
                    "bit_width": cfg.bit_width if cfg else None,
                    "status": "complete" if layer.name in completed_layers else "partial",
                    "completed": layer.name in completed_layers,
                    "calibration": {
                        "group_count": len(group_summaries),
                        "artifact_hashes": calibration_hashes,
                            "calibration_sample_ids": self._layer_sample_refs(
                                scheduled, calibration_results, layer.name
                            ),
                    },
                    "quantization": {
                        "group_count": len(quant_hashes),
                        "total_saturation_count": total_saturation,
                        "artifact_hashes": quant_hashes,
                        "role_fingerprints": [
                            quantization_results[(layer.name, group.group_index)].payload.get(
                                "role_fingerprint"
                            )
                            for group in scheduled
                            if (layer.name, group.group_index) in quantization_results
                        ],
                    },
                    "error": {
                        "record_count": total_records,
                        "pre_exceeded_count": pre_exceeded,
                        "post_exceeded_count": post_exceeded,
                        "attribution_counts": attribution_counts,
                    },
                    "aggregate_outputs": aggregate,
                    "groups": group_summaries,
                    "cache": layer_cache,
                    "invalidation": invalidation.get("layers", {}).get(layer.name),
                    "pending_groups": [
                        {
                            "group_index": item.group_index,
                            "code": item.code,
                            "points": item.num_points,
                        }
                        for item in plan.pending
                        if item.layer_name == layer.name
                    ],
                }
            )

        role_fingerprints = {
            layer.name: {
                "calibration": [
                    calibration_results[(layer.name, group.group_index)].payload.get("role_fingerprint")
                    for group in plan.by_layer().get(layer.name, [])
                    if (layer.name, group.group_index) in calibration_results
                ],
                "quantization": [
                    quantization_results[(layer.name, group.group_index)].payload.get("role_fingerprint")
                    for group in plan.by_layer().get(layer.name, [])
                    if (layer.name, group.group_index) in quantization_results
                ],
                "quantization_mode": (
                    {
                        "bit_width": config.for_layer(layer).bit_width,
                        "fractional_bits": config.for_layer(layer).fractional_bits,
                        "quantization_levels": config.for_layer(layer).quantization_levels,
                        "scale_effective_digits": config.for_layer(layer).scale_effective_digits,
                        "scheme": config.for_layer(layer).scheme,
                    }
                    if config is not None
                    else None
                ),
            }
            for layer in dataset.valid_layers
        }

        strict_codes = {
            E_BATCH_BUDGET_PENDING,
            E_QUOTA_EXCEEDED,
        }
        has_hard_failure = any(item.code not in strict_codes for item in failures.items)
        if global_block or has_hard_failure:
            status = "failed"
        elif failures.items or partial_layers:
            status = "partial"
        else:
            status = "ok"
        cache_summary: dict[str, Any] = {}
        for event in cache_events:
            stage = str(event["stage"])
            bucket = cache_summary.setdefault(stage, {"computed": 0, "reused": 0})
            bucket["computed" if event.get("computed") else "reused"] += 1

        report = {
            "schema": "quantcalib-diagnostic-report-v1",
            "tool_version": "1.0.0",
            "status": status,
            "run_mode": "full_rebuild" if self.request.full_rebuild else "incremental_or_resume",
            "input_fingerprints": {
                "weights": dataset.weight_fingerprint,
                "calibration": dataset.calibration_fingerprint,
                "config": config_hash,
                "receipts": dataset.receipts.fingerprint,
                "global_version": global_version,
            },
            "loading": self._loading_payload(dataset, plan, failures),
            "version_check": version_check,
            "recovery": {
                "resume_from_existing_generation": version_check.get(
                    "resume_from_existing_generation", False
                ),
                "previous_snapshot": version_check.get("previous_snapshot"),
                "previous_completed_layers": version_check.get(
                    "previous_completed_layers", []
                ),
                "previous_partial_layers": version_check.get("previous_partial_layers", {}),
                "invalidated_layers": sorted(
                    layer
                    for layer, item in invalidation.get("layers", {}).items()
                    if item.get("invalidated_stages")
                ),
            },
            "commands_applied": commands_applied,
            "fault_injections": fault_results,
            "failure_list": failures.to_list(),
            "business_code_counts": failures.counts(),
            "complete": status == "ok",
            "completed_layers": completed_layers,
            "partial_layers": partial_layers,
            "blocked_layers": blocked_layers,
            "layers": layers_report,
            "error_attribution_priority": list(ATTRIBUTION_PRIORITY),
            "error_attribution": attribution_table,
            "role_fingerprints": role_fingerprints,
            "budget": budget_report(plan, config) if config is not None else None,
            "reuse": {
                "summary": dict(sorted(cache_summary.items())),
                "by_layer": cache_by_layer,
                "events": cache_events,
                "ledger_reused": ledger_reused,
                "ledger_cache_key": ledger_key,
                "exactness": "cache keys contain layer content fingerprint, per-layer parameter fingerprint, bit profile and calibration-set fingerprint",
            },
            "invalidation": invalidation,
            "receipt_ledger": ledger_payload,
            "artifacts": {
                "state_root": str(store.root),
                "current_pointer": str(store.pointer_path),
                "stage_artifact_versions": {
                    stage: store.stage_digest(stage)
                    for stage in (
                        "loading",
                        "calibration",
                        "quantization",
                        "error_analysis",
                        "distillation",
                        "receipt_ledger",
                    )
                },
                "snapshots": list(store.committed_snapshots),
            },
            "manifest": {
                "global_version": manifest["global_version"],
                "config_hash": manifest["config_hash"],
                "weight_package_fingerprint": manifest["weight_package_fingerprint"],
                "calibration_fingerprint": manifest["calibration_fingerprint"],
                "completed_layers": manifest["completed_layers"],
                "partial_layers": manifest["partial_layers"],
                "consumed_budget": manifest["consumed_budget"],
                "stage_completed_layers": manifest["stage_completed_layers"],
            },
            "determinism": {
                "percentile_rule": "linear interpolation over sorted values at position=(permille/1000)*(n-1)",
                "tie_rule": "ascending value then original index; equal boundary values are interchangeable",
                "rounding_rule": "nearest integer, ties away from zero",
                "scale_rule": "float64 derivation, effective-significant-digit crop, binary fixed-point profile",
                "attribution_tie_rule": list(ATTRIBUTION_PRIORITY),
                "worker_order_invariant": True,
            },
        }
        projection = {
            "layers": [
                {
                    "layer": layer.get("layer"),
                    "content_version": layer.get("content_version"),
                    "content_fingerprint": layer.get("content_fingerprint"),
                    "status": layer.get("status"),
                    "completed": layer.get("completed"),
                    "calibration": layer.get("calibration"),
                    "quantization": layer.get("quantization"),
                    "error": layer.get("error"),
                    "aggregate_outputs": layer.get("aggregate_outputs"),
                    "groups": [
                        {
                            key: value
                            for key, value in group.items()
                            if not key.endswith("_reused")
                        }
                        for group in layer.get("groups", [])
                    ],
                }
                for layer in layers_report
            ],
            "receipt_ledger": ledger_payload,
            "budget": {
                "global_consumed": plan.global_consumed,
                "per_layer_consumed": dict(sorted(plan.per_layer_consumed.items())),
                "consumption_sequence": plan.consumption_sequence,
                "batch_events": plan.batch_events,
                "pending_items": [
                    {
                        "layer": item.layer_name,
                        "group_index": item.group_index,
                        "code": item.code,
                    }
                    for item in plan.pending
                ],
            },
            "role_fingerprints": role_fingerprints,
            "completed_layers": completed_layers,
            "partial_layers": partial_layers,
            "business_code_counts": failures.counts(),
        }
        report["result_digest"] = stable_hash(projection)
        return report

    @staticmethod
    def _layer_sample_refs(
        scheduled: list[ScheduledGroup],
        calibration_results: dict[tuple[str, int], StageTaskResult],
        layer_name: str,
    ) -> list[str]:
        refs: set[str] = set()
        for group in scheduled:
            result = calibration_results.get((layer_name, group.group_index))
            if result is not None and result.payload is not None:
                refs.update(result.payload.get("calibration_sample_ids", []))
        return sorted(refs)


def blocked_layers_from_failures(failures: FailureList) -> set[str]:
    codes = {
        E_WEIGHT_INVALID,
        E_WEIGHT_SHAPE,
        E_WEIGHT_DUPLICATE,
        E_GROUP_INVALID,
        E_QUANT_BOUNDS,
        E_VERSION_ORDER,
    }
    return {
        item.layer
        for item in failures.items
        if item.layer is not None and item.code in codes
    }
