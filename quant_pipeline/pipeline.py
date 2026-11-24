"""End-to-end deterministic quantization, caching, scheduling, and reporting."""

from __future__ import annotations

import copy
import json
import threading
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .cache_keys import derive_cache_key, parameter_fingerprint
from .calibration import CalibrationResult, calibrate_layer
from .config import (
    ConfigIssue,
    EffectiveLayerConfig,
    PipelineConfig,
    format_effective_config_summary,
    parse_config,
    snapshot_config_hash,
)
from .distillation import distill_layer
from .error_analysis import ATTRIBUTION_PRIORITY, analyze_errors
from .errors import (
    E_CALIBRATION_INVALID,
    E_CONFIG_INVALID,
    E_INJECTION_INVALID,
    E_LAYER_CALIBRATION,
    E_LAYER_DISTILLATION,
    E_LAYER_QUANTIZATION,
    E_POINT_BUDGET_PENDING,
    E_POINT_QUOTA_EXCEEDED,
    E_RUN_INTERRUPTED,
    E_WEIGHT_PACK_INVALID,
    FailureRecord,
    PipelineError,
    failure,
)
from .inputs import (
    CalibrationSet,
    InjectionEngine,
    LayerCalibrationView,
    RawWeightRecord,
    WeightLayer,
    apply_injections,
    build_layer_calibration_views,
    load_injections,
    load_weight_records,
    normalize_calibration,
    validate_weight_pack,
    weight_pack_fingerprint,
)
from .quantization import QuantizationResult, quantize_layer
from .snapshot import SCHEMA_VERSION, STAGES, SnapshotStore, snapshot_global_version
from .util import pretty_json, sha256_json
from .yaml_lite import load_yaml


PIPELINE_VERSION = "1.0.0"


@dataclass
class InputPaths:
    weights: Path
    calibration: Path
    config: Path
    injections: Path | None


@dataclass
class PreparedInputs:
    config: PipelineConfig
    layers: dict[str, WeightLayer]
    calibration_set: CalibrationSet | None
    calibration_views: dict[str, LayerCalibrationView]
    injection: InjectionEngine
    failures: list[FailureRecord]
    identity: dict[str, Any]
    input_summary: dict[str, Any]


@dataclass
class LayerPlan:
    layer: WeightLayer
    config: EffectiveLayerConfig
    keys: dict[str, str]
    cached: dict[str, dict[str, Any] | None]
    hits: dict[str, bool]
    force_distill: bool
    cost: int
    task_sort_key: str
    distill_enabled: bool

    @property
    def needs_quantization_points(self) -> bool:
        return self.cost > 0


@dataclass
class LayerExecution:
    layer: str
    status: str
    records: dict[str, dict[str, Any]] = field(default_factory=dict)
    keys: dict[str, str] = field(default_factory=dict)
    reused_stages: list[str] = field(default_factory=list)
    recomputed_stages: list[str] = field(default_factory=list)
    failed_stage: str | None = None
    failure: FailureRecord | None = None
    forced_distill_recompute: bool = False
    task_sort_key: str = ""


class QuantizationPipeline:
    def __init__(
        self,
        sample_dir: str | Path,
        output_dir: str | Path,
        *,
        weights_path: str | Path | None = None,
        calibration_path: str | Path | None = None,
        config_path: str | Path | None = None,
        injections_path: str | Path | None = None,
        profile: str | None = None,
        fractional_bits: int | None = None,
        max_workers: int | None = None,
        point_budget: int | None = None,
        layer_point_quota: int | None = None,
        snapshot_interval: int | None = None,
        stop_event: threading.Event | None = None,
    ) -> None:
        self.sample_dir = Path(sample_dir)
        self.output_dir = Path(output_dir)
        self.weights_path = Path(weights_path) if weights_path is not None else None
        self.calibration_path = Path(calibration_path) if calibration_path is not None else None
        self.config_path = Path(config_path) if config_path is not None else None
        self.injections_path = Path(injections_path) if injections_path is not None else None
        self.profile = profile
        self.fractional_bits = fractional_bits
        self.max_workers = max_workers
        self.point_budget = point_budget
        self.layer_point_quota = layer_point_quota
        self.snapshot_interval = snapshot_interval
        self.stop_event = stop_event or threading.Event()

    # ------------------------------------------------------------------ loading
    def _resolve_paths(self) -> InputPaths:
        def first(candidates: list[Path], explicit: Path | None, label: str) -> Path:
            if explicit is not None:
                return explicit
            for candidate in candidates:
                if candidate.exists():
                    return candidate
            return candidates[0]

        weights = first(
            [
                self.sample_dir / "weights.jsonl",
                self.sample_dir / "weights_pack.jsonl",
                self.sample_dir / "weight_pack.jsonl",
                self.sample_dir / "layer_weights.jsonl",
            ],
            self.weights_path,
            "weights",
        )
        calibration = first(
            [
                self.sample_dir / "calibration.json",
                self.sample_dir / "calibration_set.json",
                self.sample_dir / "calibration_samples.json",
            ],
            self.calibration_path,
            "calibration",
        )
        config = first(
            [
                self.sample_dir / "config.yaml",
                self.sample_dir / "config.yml",
                self.sample_dir / "quant_config.yaml",
                self.sample_dir / "quantization.yaml",
                self.sample_dir / "configs" / "quantization.yaml",
            ],
            self.config_path,
            "config",
        )
        if self.injections_path is not None:
            injections: Path | None = self.injections_path
        else:
            candidates = [
                self.sample_dir / "injections.yaml",
                self.sample_dir / "injection_config.yaml",
                self.sample_dir / "fault_injections.yaml",
                self.sample_dir / "configs" / "injections_all.yaml",
            ]
            injections = next((candidate for candidate in candidates if candidate.exists()), None)
        return InputPaths(weights, calibration, config, injections)

    def _prepare_inputs(self, paths: InputPaths) -> tuple[PreparedInputs | None, list[FailureRecord]]:
        failures: list[FailureRecord] = []

        try:
            config_raw = load_yaml(paths.config)
        except Exception as exc:
            failures.append(
                failure(
                    E_CONFIG_INVALID,
                    f"cannot load quantization configuration: {exc}",
                    stage="validation",
                    exception_type=type(exc).__name__,
                )
            )
            config_raw = None
        if not isinstance(config_raw, dict):
            failures.append(
                failure(
                    E_CONFIG_INVALID,
                    "quantization configuration root must be a mapping",
                    stage="validation",
                    exception_type="ConfigError",
                )
            )
            return None, failures

        if paths.weights.exists():
            records = load_weight_records(paths.weights)
        else:
            records = []
            failures.append(
                failure(
                    E_WEIGHT_PACK_INVALID,
                    f"weight pack does not exist: {paths.weights}",
                    stage="validation",
                    exception_type="FileNotFoundError",
                )
            )

        calibration_raw: Any = {}
        if paths.calibration.exists():
            try:
                with open(paths.calibration, "r", encoding="utf-8") as handle:
                    calibration_raw = json.load(handle)
            except Exception as exc:
                failures.append(
                    failure(
                        E_CALIBRATION_INVALID,
                        f"cannot read calibration JSON: {exc}",
                        stage="validation",
                        exception_type=type(exc).__name__,
                    )
                )
                calibration_raw = {}
        else:
            failures.append(
                failure(
                    E_CALIBRATION_INVALID,
                    f"calibration set does not exist: {paths.calibration}",
                    stage="validation",
                    exception_type="FileNotFoundError",
                )
            )

        injection_items, injection_load_failures = load_injections(paths.injections)
        injection = apply_injections(
            injection_items,
            records,
            config_raw,
            calibration_raw,
        )
        injection.failures.extend(injection_load_failures)

        try:
            config, issues = parse_config(
                config_raw,
                profile=self.profile,
                fractional_bits=self.fractional_bits,
                runtime_overrides={
                    "max_workers": self.max_workers,
                    "point_budget": self.point_budget,
                    "layer_point_quota": self.layer_point_quota,
                    "snapshot_interval": self.snapshot_interval,
                },
                overrides={
                    key: value
                    for key, value in {
                        "profile": self.profile,
                        "fractional_bits": self.fractional_bits,
                        "max_workers": self.max_workers,
                        "point_budget": self.point_budget,
                        "layer_point_quota": self.layer_point_quota,
                        "snapshot_interval": self.snapshot_interval,
                    }.items()
                    if value is not None
                },
            )
        except Exception as exc:
            failures.append(
                failure(
                    E_CONFIG_INVALID,
                    f"configuration normalization failed: {exc}",
                    stage="validation",
                    exception_type=type(exc).__name__,
                )
            )
            return None, failures

        fatal_issues = config.global_fatal_issues
        if fatal_issues:
            for issue in fatal_issues:
                failures.append(
                    failure(
                        issue.code,
                        issue.message,
                        stage="validation",
                        exception_type="ConfigError",
                        details={"field": issue.field, "scope": issue.scope},
                    )
                )
            failures.extend(injection.failures)
            injection.finalize_hits(failures)
            return None, failures

        layers, weight_failures = validate_weight_pack(records, config, injection)
        failures.extend(weight_failures)
        failures.extend(injection.failures)

        try:
            calibration_set = normalize_calibration(calibration_raw)
        except PipelineError as exc:
            calibration_set = None
            failures.append(FailureRecord.from_exception(exc))
        calibration_views: dict[str, LayerCalibrationView] = {}
        if calibration_set is not None:
            calibration_views, calibration_failures = build_layer_calibration_views(
                layers, calibration_set, injection
            )
            failures.extend(calibration_failures)
            for failed_layer in calibration_failures:
                if failed_layer.layer:
                    layers.pop(failed_layer.layer, None)
                    calibration_views.pop(failed_layer.layer, None)

        if not config.distillation_valid:
            injection_ids = injection.injection_ids_for_global_distillation()
            for issue in config.issues:
                if issue.scope != "distillation":
                    continue
                failures.append(
                    failure(
                        issue.code,
                        issue.message,
                        stage="distillation",
                        exception_type="ConfigError",
                        injection_id=injection_ids[0] if injection_ids else None,
                        details={"field": issue.field},
                    )
                )

        run_layers = {
            name: layer for name, layer in layers.items() if name in calibration_views
        }
        if not run_layers and not failures:
            failures.append(
                failure(
                    E_WEIGHT_PACK_INVALID,
                    "weight pack contains no runnable layers",
                    stage="validation",
                    exception_type="WeightPackError",
                )
            )

        weight_fingerprint = weight_pack_fingerprint(layers)
        calibration_hash = calibration_set.raw_hash if calibration_set is not None else ""
        profile_name = self.profile if self.profile is not None else config.active_profile
        profile = config.profiles.get(profile_name, config.profiles[config.active_profile])
        global_fractional_bits = (
            self.fractional_bits if self.fractional_bits is not None else profile.fractional_bits
        )
        identity = {
            "config_hash": snapshot_config_hash(
                config.raw,
                profile=self.profile,
                fractional_bits=self.fractional_bits,
            ),
            "weight_pack_fingerprint": weight_fingerprint,
            "calibration_hash": calibration_hash,
            "quantization_profile": profile_name,
            "fractional_bits": global_fractional_bits,
        }
        input_summary = {
            "weights_file": paths.weights.name,
            "calibration_file": paths.calibration.name,
            "config_file": paths.config.name,
            "injections_file": paths.injections.name if paths.injections is not None else None,
            "weight_layer_count": len(run_layers),
            "calibration_sample_count": len(calibration_set.samples) if calibration_set else 0,
        }
        return (
            PreparedInputs(
                config=config,
                layers=run_layers,
                calibration_set=calibration_set,
                calibration_views=calibration_views,
                injection=injection,
                failures=failures,
                identity=identity,
                input_summary=input_summary,
            ),
            failures,
        )

    # ------------------------------------------------------------------- running
    def run(self) -> dict[str, Any]:
        paths = self._resolve_paths()
        prepared, load_failures = self._prepare_inputs(paths)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        store = SnapshotStore(self.output_dir)
        if prepared is None:
            report = {
                "schema_version": SCHEMA_VERSION,
                "pipeline_version": PIPELINE_VERSION,
                "status": "failed",
                "business_codes": sorted({record.code for record in load_failures}),
                "input": {
                    "weights_file": paths.weights.name,
                    "calibration_file": paths.calibration.name,
                    "config_file": paths.config.name,
                },
                "layers": [],
                "cache_ledger": {
                    "layer_order": [],
                    "stages": {stage: {"reused": [], "recomputed": [], "failed": [], "skipped": []} for stage in STAGES},
                    "reused": [],
                    "recomputed": [],
                },
                "pending": [],
                "failures": [record.as_dict() for record in load_failures],
                "injections": [],
                "events": [],
                "snapshot": {"status": "not_created"},
                "result_fingerprint": sha256_json([record.as_dict() for record in load_failures]),
            }
            store.write_report(report)
            return report

        config = prepared.config
        injection = prepared.injection
        failures: list[FailureRecord] = list(prepared.failures)
        identity = prepared.identity
        recovery = store.determine_recovery(identity)
        previous_manifest = store.read_manifest() if recovery.cache_allowed else None
        previous_stage_keys = (
            previous_manifest.get("layer_stage_keys", {})
            if isinstance(previous_manifest, dict)
            else {}
        )
        if not isinstance(previous_stage_keys, dict):
            previous_stage_keys = {}

        cached_stages: dict[str, dict[str, dict[str, Any]]] = {}
        for stage in STAGES:
            if recovery.cache_allowed:
                cached_stages[stage], _ = store.load_stage_results(stage)
            else:
                cached_stages[stage] = {}

        layer_order = sorted(prepared.layers)
        effective: dict[str, EffectiveLayerConfig] = {}
        for layer_name in layer_order:
            layer = prepared.layers[layer_name]
            try:
                effective[layer_name] = config.resolve_layer(
                    layer_name, layer.numel, marker_group_size=layer.marker_group_size
                )
            except PipelineError as exc:
                injection_ids = injection.injection_ids_for_config_layer(layer_name)
                failures.append(
                    failure(
                        exc.code,
                        exc.message,
                        stage="config",
                        layer=layer_name,
                        exception_type=exc.exception_type,
                        injection_id=injection_ids[0] if injection_ids else None,
                        details=exc.details,
                    )
                )

        plans: list[LayerPlan] = []
        keys_by_layer: dict[str, dict[str, str]] = {}
        for layer_name in sorted(effective):
            layer = prepared.layers[layer_name]
            layer_config = effective[layer_name]
            layer_calibration_hash = prepared.calibration_views[layer_name].fingerprint()
            calib_params = parameter_fingerprint(layer_config.calibration_parameters())
            calibration_key = derive_cache_key(
                stage="calibration",
                layer=layer_name,
                content_fingerprint=layer.content_fingerprint,
                parameter_fingerprint=calib_params,
                quantization_profile=layer_config.profile,
                calibration_hash=layer_calibration_hash,
                quant_gear=layer_config.target_bits,
                extra={
                    "fractional_bits": layer_config.fractional_bits,
                },
            )
            quant_params = parameter_fingerprint(layer_config.quantization_parameters())
            quantization_key = derive_cache_key(
                stage="quantization",
                layer=layer_name,
                content_fingerprint=layer.content_fingerprint,
                parameter_fingerprint=quant_params,
                quantization_profile=layer_config.profile,
                calibration_hash=layer_calibration_hash,
                quant_gear=layer_config.target_bits,
                extra={
                    "calibration_key": calibration_key,
                },
            )
            error_params = parameter_fingerprint(
                {
                    **config.tolerances.as_dict(),
                    "attribution_priority": list(ATTRIBUTION_PRIORITY),
                    "quantization_key": quantization_key,
                }
            )
            error_key = derive_cache_key(
                stage="error_analysis",
                layer=layer_name,
                content_fingerprint=layer.content_fingerprint,
                parameter_fingerprint=error_params,
                quantization_profile=layer_config.profile,
                calibration_hash=layer_calibration_hash,
                quant_gear=layer_config.target_bits,
                extra={
                    "quantization_key": quantization_key,
                },
            )
            distill_params = parameter_fingerprint(
                {
                    **config.distillation.as_dict(),
                    "error_key": error_key,
                    "calibration_hash": layer_calibration_hash,
                }
            )
            distill_key = derive_cache_key(
                stage="distillation",
                layer=layer_name,
                content_fingerprint=layer.content_fingerprint,
                parameter_fingerprint=distill_params,
                quantization_profile=layer_config.profile,
                calibration_hash=layer_calibration_hash,
                quant_gear=layer_config.target_bits,
                extra={
                    "error_key": error_key,
                },
            )
            keys = {
                "calibration": calibration_key,
                "quantization": quantization_key,
                "error_analysis": error_key,
                "distillation": distill_key,
            }
            keys_by_layer[layer_name] = keys

            cached = {
                "calibration": _matching_cache(cached_stages["calibration"].get(layer_name), calibration_key),
                "quantization": _matching_cache(cached_stages["quantization"].get(layer_name), quantization_key),
                "error_analysis": _matching_cache(cached_stages["error_analysis"].get(layer_name), error_key),
                "distillation": _matching_cache(cached_stages["distillation"].get(layer_name), distill_key),
            }
            hits = {stage: cached[stage] is not None for stage in keys}
            cost = 0 if hits["calibration"] and hits["quantization"] else layer.numel
            plans.append(
                LayerPlan(
                    layer=layer,
                    config=layer_config,
                    keys=keys,
                    cached=cached,
                    hits=hits,
                    force_distill=False,
                    cost=cost,
                    task_sort_key=f"{layer_name}|calibration>quantization>error_analysis>distillation",
                    distill_enabled=config.distillation_valid,
                )
            )
        plans.sort(key=lambda plan: plan.layer.name)

        dirty_layers: set[str] = set()
        for plan in plans:
            before = previous_stage_keys.get(plan.layer.name, {})
            previous_calibration = before.get("calibration") if isinstance(before, dict) else None
            previous_quantization = before.get("quantization") if isinstance(before, dict) else None
            if (
                previous_calibration != plan.keys["calibration"]
                or previous_quantization != plan.keys["quantization"]
                or not plan.hits["calibration"]
                or not plan.hits["quantization"]
            ):
                dirty_layers.add(plan.layer.name)
        dirty_positions = [layer_order.index(name) for name in dirty_layers if name in layer_order]
        first_dirty = min(dirty_positions) if dirty_positions else None
        for plan in plans:
            position = layer_order.index(plan.layer.name)
            plan.force_distill = bool(
                first_dirty is not None
                and position >= first_dirty
                and plan.hits["distillation"]
                and plan.layer.name not in dirty_layers
            ) or bool(first_dirty is not None and position >= first_dirty)
            # A dirty layer recomputes naturally because its derived key changed.
            # Unchanged downstream layers are forced as required by the ledger semantics.
            if first_dirty is not None and position >= first_dirty and plan.layer.name in dirty_layers:
                plan.force_distill = True

        budget = config.runtime.point_budget
        quota = config.runtime.layer_point_quota
        remaining = budget
        budget_blocked = False
        selected: list[LayerPlan] = []
        pending: list[dict[str, Any]] = []
        quota_failed_layers: set[str] = set()
        for plan in plans:
            if plan.cost > quota:
                failures.append(
                    failure(
                        E_POINT_QUOTA_EXCEEDED,
                        f"layer requires {plan.cost} points, exceeding the per-layer quota {quota}",
                        stage="scheduler",
                        layer=plan.layer.name,
                        exception_type="PointQuotaError",
                        details={"cost": plan.cost, "layer_point_quota": quota},
                    )
                )
                quota_failed_layers.add(plan.layer.name)
                continue
            if plan.cost > 0 and (budget_blocked or plan.cost > remaining):
                budget_blocked = True
                pending.append(
                    {
                        "layer": plan.layer.name,
                        "code": E_POINT_BUDGET_PENDING,
                        "reason": "batch point budget exhausted before the stable-order layer could run",
                        "required_points": plan.cost,
                        "remaining_points": remaining,
                    }
                )
                continue
            remaining -= plan.cost
            selected.append(plan)

        executions: dict[str, LayerExecution] = {}
        stopped_layers: set[str] = set()

        def execute(plan: LayerPlan) -> LayerExecution:
            view = prepared.calibration_views[plan.layer.name]
            return self._execute_layer(plan, view, config, injection)

        # Insert pre-validation failures into the report but not into successful stats.
        stage_results: dict[str, dict[str, dict[str, Any]]] = {
            stage: {} for stage in ("calibration", "quantization", "error_analysis", "distillation")
        }
        result_keys: dict[str, dict[str, str]] = {
            stage: {} for stage in ("calibration", "quantization", "error_analysis", "distillation")
        }
        completed_so_far: list[str] = []
        processed_count = 0
        batch_consumed = 0
        stage_versions: dict[str, str] = {}
        interval = config.runtime.snapshot_interval

        def commit_snapshot(status: str) -> None:
            nonlocal stage_versions
            parse_results: dict[str, dict[str, Any]] = {}
            parse_keys: dict[str, str] = {}
            for layer_name in layer_order:
                if layer_name not in prepared.layers:
                    continue
                layer = prepared.layers[layer_name]
                parse_calibration_hash = prepared.calibration_views[layer_name].fingerprint()
                parse_key = derive_cache_key(
                    stage="parse_validation",
                    layer=layer_name,
                    content_fingerprint=layer.content_fingerprint,
                    parameter_fingerprint=parameter_fingerprint(
                        effective[layer_name].as_dict() if layer_name in effective else {}
                    ),
                    quantization_profile=(
                        effective[layer_name].profile if layer_name in effective else identity["quantization_profile"]
                    ),
                    calibration_hash=parse_calibration_hash,
                    quant_gear=(effective[layer_name].target_bits if layer_name in effective else None),
                )
                parse_results[layer_name] = {
                    "cache_key": parse_key,
                    "data": {
                        "layer": layer_name,
                        "shape": list(layer.shape),
                        "version": layer.version,
                        "marker_group_size": layer.marker_group_size,
                        "content_fingerprint": layer.content_fingerprint,
                        "effective_config": (
                            effective[layer_name].as_dict() if layer_name in effective else None
                        ),
                    },
                }
                parse_keys[layer_name] = parse_key

            stage_version_map: dict[str, str] = {}
            stage_result_keys: dict[str, dict[str, str]] = {"parse_validation": parse_keys}
            stage_version_map["parse_validation"] = store.save_stage(
                "parse_validation",
                identity=identity,
                results=parse_results,
                result_keys=parse_keys,
            )
            for stage in ("calibration", "quantization", "error_analysis", "distillation"):
                stage_version_map[stage] = store.save_stage(
                    stage,
                    identity=identity,
                    results=stage_results[stage],
                    result_keys=result_keys[stage],
                )
                stage_result_keys[stage] = dict(result_keys[stage])
            stage_versions = store.read_stage_versions()

            completed_now = [
                layer_name
                for layer_name in layer_order
                if self._layer_complete(
                    layer_name,
                    stage_results,
                    config.distillation_valid,
                    executions,
                    quota_failed_layers,
                )
            ]
            completed_so_far[:] = completed_now
            consumed_cumulative = sum(
                prepared.layers[name].numel
                for name in stage_results["quantization"]
                if name in prepared.layers
            )
            layer_stage_keys = {
                name: {
                    stage: result_keys[stage][name]
                    for stage in ("calibration", "quantization", "error_analysis", "distillation")
                    if name in result_keys[stage]
                }
                for name in layer_order
                if any(name in result_keys[stage] for stage in result_keys)
            }
            manifest = {
                "schema_version": SCHEMA_VERSION,
                "pipeline_version": PIPELINE_VERSION,
                "global_version": snapshot_global_version(identity),
                "status": status,
                "config_hash": identity["config_hash"],
                "weight_pack_fingerprint": identity["weight_pack_fingerprint"],
                "calibration_hash": identity["calibration_hash"],
                "quantization_profile": identity["quantization_profile"],
                "fractional_bits": identity["fractional_bits"],
                "completed_layers": completed_now,
                "consumed_point_budget": consumed_cumulative,
                "batch_consumed_point_budget": batch_consumed,
                "stage_versions": stage_version_map,
                "layer_stage_keys": layer_stage_keys,
                "pending": list(pending),
                "failures": [record.as_dict() for record in failures],
                "injections": injection.reports_sorted(),
            }
            store.write_manifest(manifest)

        if selected:
            max_workers = config.runtime.max_workers
            with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="quant-layer") as executor:
                pending_futures: dict[Future[LayerExecution], LayerPlan] = {}
                submit_index = 0

                def submit_next() -> bool:
                    nonlocal submit_index
                    while submit_index < len(selected):
                        plan = selected[submit_index]
                        submit_index += 1
                        if self.stop_event.is_set():
                            return False
                        pending_futures[executor.submit(execute, plan)] = plan
                        return True
                    return False

                for _ in range(min(max_workers, len(selected))):
                    if not submit_next():
                        break
                while pending_futures:
                    done, _ = wait(list(pending_futures), return_when=FIRST_COMPLETED)
                    for future in done:
                        plan = pending_futures.pop(future)
                        try:
                            execution = future.result()
                        except Exception as exc:  # defensive isolation for unexpected thread failures
                            execution = LayerExecution(
                                layer=plan.layer.name,
                                status="failed",
                                keys=plan.keys,
                                failed_stage="runtime",
                                failure=failure(
                                    E_LAYER_QUANTIZATION,
                                    f"unexpected layer task failure: {exc}",
                                    stage="runtime",
                                    layer=plan.layer.name,
                                    exception_type=type(exc).__name__,
                                ),
                                task_sort_key=plan.task_sort_key,
                            )
                        executions[plan.layer.name] = execution
                        if plan.cost > 0:
                            batch_consumed += plan.cost
                        for stage, record in execution.records.items():
                            stage_results[stage][plan.layer.name] = record
                            result_keys[stage][plan.layer.name] = record["cache_key"]
                        if execution.failure is not None:
                            failures.append(execution.failure)
                        processed_count += 1
                        if processed_count % interval == 0:
                            commit_snapshot("running")
                        if not self.stop_event.is_set():
                            submit_next()
                if self.stop_event.is_set():
                    for index in range(submit_index, len(selected)):
                        plan = selected[index]
                        if plan.layer.name not in executions:
                            stopped_layers.add(plan.layer.name)
                            pending.append(
                                {
                                    "layer": plan.layer.name,
                                    "code": E_RUN_INTERRUPTED,
                                    "reason": "graceful stop requested; layer was not started",
                                    "required_points": plan.cost,
                                    "remaining_points": remaining,
                                }
                            )
        if self.stop_event.is_set():
            for plan in selected:
                if plan.layer.name not in executions and plan.layer.name not in stopped_layers:
                    stopped_layers.add(plan.layer.name)
                    pending.append(
                        {
                            "layer": plan.layer.name,
                            "code": E_RUN_INTERRUPTED,
                            "reason": "graceful stop requested; layer was not started",
                            "required_points": plan.cost,
                            "remaining_points": remaining,
                        }
                    )

        report_status = self._determine_report_status(
            layer_order,
            stage_results,
            executions,
            failures,
            pending,
            config.distillation_valid,
            quota_failed_layers,
        )
        snapshot_status = (
            "interrupted"
            if pending or self.stop_event.is_set()
            else ("completed" if report_status == "completed" else "failed")
        )
        injection.finalize_hits(failures)
        commit_snapshot(snapshot_status)

        report = self._build_report(
            config=config,
            prepared=prepared,
            paths=paths,
            layer_order=layer_order,
            effective=effective,
            stage_results=stage_results,
            result_keys=result_keys,
            executions=executions,
            plans=plans,
            failures=failures,
            pending=pending,
            quota_failed_layers=quota_failed_layers,
            recovery_events=recovery.events,
            recovery_status=recovery.status,
            final_snapshot_status=snapshot_status,
            previous_status=recovery.previous_status,
            stage_versions=stage_versions,
            report_status=report_status,
            batch_consumed=batch_consumed,
            completed_layers=completed_so_far,
        )
        store.write_report(report)
        return report

    def _execute_layer(
        self,
        plan: LayerPlan,
        view: LayerCalibrationView,
        config: PipelineConfig,
        injection: InjectionEngine,
    ) -> LayerExecution:
        layer = plan.layer
        execution = LayerExecution(
            layer=layer.name,
            status="ok",
            keys=dict(plan.keys),
            forced_distill_recompute=plan.force_distill,
            task_sort_key=plan.task_sort_key,
        )
        calibration: CalibrationResult | None = None
        quantization: QuantizationResult | None = None
        error_data: dict[str, Any] | None = None

        try:
            if plan.hits["calibration"] and plan.cached["calibration"] is not None:
                calibration = CalibrationResult.from_data(plan.cached["calibration"]["data"])
                record = copy.deepcopy(plan.cached["calibration"])
                execution.records["calibration"] = record
                execution.reused_stages.append("calibration")
            else:
                calibration = calibrate_layer(layer, plan.config)
                record = {"cache_key": plan.keys["calibration"], "data": calibration.data}
                execution.records["calibration"] = record
                execution.recomputed_stages.append("calibration")

            if plan.hits["quantization"] and plan.cached["quantization"] is not None:
                quantization = QuantizationResult.from_data(plan.cached["quantization"]["data"])
                execution.records["quantization"] = copy.deepcopy(plan.cached["quantization"])
                execution.reused_stages.append("quantization")
            else:
                quantization = quantize_layer(layer, calibration, plan.config)
                execution.records["quantization"] = {
                    "cache_key": plan.keys["quantization"],
                    "data": quantization.data,
                }
                execution.recomputed_stages.append("quantization")

            if plan.hits["error_analysis"] and plan.cached["error_analysis"] is not None:
                error_data = plan.cached["error_analysis"]["data"]
                execution.records["error_analysis"] = copy.deepcopy(plan.cached["error_analysis"])
                execution.reused_stages.append("error_analysis")
            else:
                error_data = analyze_errors(
                    layer,
                    quantization,
                    calibration,
                    plan.config,
                    config.tolerances,
                )
                execution.records["error_analysis"] = {
                    "cache_key": plan.keys["error_analysis"],
                    "data": error_data,
                }
                execution.recomputed_stages.append("error_analysis")

            if plan.distill_enabled:
                if (
                    plan.hits["distillation"]
                    and plan.cached["distillation"] is not None
                    and not plan.force_distill
                ):
                    execution.records["distillation"] = copy.deepcopy(plan.cached["distillation"])
                    execution.reused_stages.append("distillation")
                else:
                    distill_data = distill_layer(
                        layer,
                        quantization.dequantized,
                        view,
                        config.distillation,
                    )
                    execution.records["distillation"] = {
                        "cache_key": plan.keys["distillation"],
                        "data": distill_data,
                    }
                    execution.recomputed_stages.append("distillation")
        except PipelineError as exc:
            injection_ids = self._injection_ids_for_stage(injection, layer.name, exc.stage)
            record = failure(
                exc.code,
                exc.message,
                stage=exc.stage,
                layer=layer.name,
                exception_type=exc.exception_type,
                injection_id=injection_ids[0] if injection_ids else exc.injection_id,
                details=exc.details,
            )
            execution.status = "failed"
            execution.failed_stage = exc.stage
            execution.failure = record
            execution.records = {
                stage: execution.records[stage]
                for stage in execution.records
                if stage != exc.stage
            }
        except Exception as exc:
            stage = "distillation" if not plan.distill_enabled else "quantization"
            record = failure(
                E_LAYER_QUANTIZATION if stage != "distillation" else E_LAYER_DISTILLATION,
                f"unexpected {stage} failure: {exc}",
                stage=stage,
                layer=layer.name,
                exception_type=type(exc).__name__,
            )
            execution.status = "failed"
            execution.failed_stage = stage
            execution.failure = record
        return execution

    @staticmethod
    def _injection_ids_for_stage(
        injection: InjectionEngine, layer: str, stage: str | None
    ) -> list[str]:
        ids: list[str] = []
        if stage == "calibration":
            ids.extend(injection.injection_ids_for_calibration(layer))
        if stage == "distillation":
            ids.extend(injection.injection_ids_for_global_distillation())
        ids.extend(injection.injection_ids_for_config_layer(layer))
        return ids

    @staticmethod
    def _layer_complete(
        layer_name: str,
        stage_results: Mapping[str, Mapping[str, Any]],
        distill_valid: bool,
        executions: Mapping[str, LayerExecution],
        quota_failed_layers: set[str],
    ) -> bool:
        if layer_name in quota_failed_layers:
            return False
        execution = executions.get(layer_name)
        if execution is not None and execution.status != "ok":
            return False
        required = ["calibration", "quantization", "error_analysis"]
        if distill_valid:
            required.append("distillation")
        else:
            return False
        return all(layer_name in stage_results[stage] for stage in required)

    @staticmethod
    def _determine_report_status(
        layer_order: list[str],
        stage_results: Mapping[str, Mapping[str, Any]],
        executions: Mapping[str, LayerExecution],
        failures: list[FailureRecord],
        pending: list[dict[str, Any]],
        distill_valid: bool,
        quota_failed_layers: set[str],
    ) -> str:
        complete = [
            name
            for name in layer_order
            if QuantizationPipeline._layer_complete(
                name, stage_results, distill_valid, executions, quota_failed_layers
            )
        ]
        if pending:
            return "partial"
        if quota_failed_layers or any(
            execution.status != "ok" for execution in executions.values()
        ) or failures:
            return "partial" if complete else "failed"
        if len(complete) != len(layer_order):
            return "partial" if complete else "failed"
        return "completed"

    # -------------------------------------------------------------------- report
    def _build_report(
        self,
        *,
        config: PipelineConfig,
        prepared: PreparedInputs,
        paths: InputPaths,
        layer_order: list[str],
        effective: dict[str, EffectiveLayerConfig],
        stage_results: dict[str, dict[str, dict[str, Any]]],
        result_keys: dict[str, dict[str, str]],
        executions: dict[str, LayerExecution],
        plans: list[LayerPlan],
        failures: list[FailureRecord],
        pending: list[dict[str, Any]],
        quota_failed_layers: set[str],
        recovery_events: list[dict[str, Any]],
        recovery_status: str,
        final_snapshot_status: str,
        previous_status: str | None,
        stage_versions: dict[str, str],
        report_status: str,
        batch_consumed: int,
        completed_layers: list[str],
    ) -> dict[str, Any]:
        failure_by_layer: dict[str, list[FailureRecord]] = {}
        for record in failures:
            if record.layer:
                failure_by_layer.setdefault(record.layer, []).append(record)
        pending_by_layer = {entry["layer"]: entry for entry in pending}
        all_layer_names = sorted(
            set(layer_order)
            | set(failure_by_layer)
            | set(pending_by_layer)
            | set(quota_failed_layers)
        )
        report_layers: list[dict[str, Any]] = []
        ledger_stages = {
            stage: {"reused": [], "recomputed": [], "failed": [], "skipped": []}
            for stage in ("calibration", "quantization", "error_analysis", "distillation")
        }

        for layer_name in all_layer_names:
            execution = executions.get(layer_name)
            pending_entry = pending_by_layer.get(layer_name)
            layer_failures = failure_by_layer.get(layer_name, [])
            weight_layer = prepared.layers.get(layer_name)
            base: dict[str, Any] = {
                "name": layer_name,
                "status": "pending" if pending_entry else ("failed" if layer_failures else "ok"),
                "cache_keys": dict(result_keys_for(result_keys, layer_name)),
            }
            if weight_layer is not None:
                base.update(
                    {
                        "shape": list(weight_layer.shape),
                        "version": weight_layer.version,
                        "content_fingerprint": weight_layer.content_fingerprint,
                        "marker_group_size": weight_layer.marker_group_size,
                    }
                )
            if layer_name in effective:
                base["effective_config"] = effective[layer_name].as_dict()
            if prepared.calibration_set is not None:
                base["calibration_samples"] = list(prepared.calibration_set.sample_ids)

            if pending_entry:
                base["pending"] = pending_entry
                base["status"] = "pending"
                for stage in ledger_stages:
                    ledger_stages[stage]["skipped"].append(layer_name)
            elif layer_failures and (execution is None or execution.status == "failed"):
                base["failures"] = [record.as_dict() for record in layer_failures]
                base["status"] = "failed"
                if execution is not None:
                    base["failed_stage"] = execution.failed_stage
                    for stage in ("calibration", "quantization", "error_analysis", "distillation"):
                        if stage in execution.reused_stages:
                            ledger_stages[stage]["reused"].append(layer_name)
                        elif stage in execution.recomputed_stages:
                            ledger_stages[stage]["recomputed"].append(layer_name)
                        if stage == execution.failed_stage:
                            ledger_stages[stage]["failed"].append(layer_name)
                else:
                    for stage in ledger_stages:
                        ledger_stages[stage]["failed"].append(layer_name)

            for stage, public_stage in (
                ("calibration", "calibration"),
                ("quantization", "quantization"),
                ("error_analysis", "error"),
                ("distillation", "distillation"),
            ):
                if layer_name in stage_results[stage]:
                    record = stage_results[stage][layer_name]
                    base[public_stage] = record["data"]
                    base["cache_keys"][stage] = record["cache_key"]
                    if execution is not None and execution.status == "ok":
                        if stage in execution.reused_stages:
                            ledger_stages[stage]["reused"].append(layer_name)
                        elif stage in execution.recomputed_stages:
                            ledger_stages[stage]["recomputed"].append(layer_name)
                elif (
                    not pending_entry
                    and not layer_failures
                    and stage == "distillation"
                    and not config.distillation_valid
                ):
                    ledger_stages[stage]["skipped"].append(layer_name)

            if (
                layer_name not in completed_layers
                and not pending_entry
                and not layer_failures
                and layer_name in stage_results["quantization"]
                and not config.distillation_valid
            ):
                base["status"] = "quantized_only"
            if layer_name in completed_layers:
                base["status"] = "ok"
            if base["status"] == "failed":
                # A failed layer must not contribute an incomplete error statistic
                # to the report's successful result set; stage artifacts remain
                # available for forensic inspection.
                base.pop("error", None)
                base.pop("partial_error_statistics", None)
            report_layers.append(base)

        for stage in ledger_stages:
            for key in ("reused", "recomputed", "failed", "skipped"):
                ledger_stages[stage][key] = sorted(set(ledger_stages[stage][key]))
        reused_layers = sorted(
            {
                layer
                for stage in ledger_stages.values()
                for layer in stage["reused"]
            }
        )
        recomputed_layers = sorted(
            {
                layer
                for stage in ledger_stages.values()
                for layer in stage["recomputed"]
            }
        )
        downstream_invalidated = sorted(
            layer
            for layer, execution in executions.items()
            if execution.forced_distill_recompute
            and "distillation" in execution.recomputed_stages
        )
        ledger = {
            "layer_order": layer_order,
            "stages": ledger_stages,
            "reused": reused_layers,
            "recomputed": recomputed_layers,
            "downstream_distillation_invalidated": downstream_invalidated,
        }
        schedule_tasks: list[dict[str, Any]] = []
        for plan in sorted(plans, key=lambda item: item.task_sort_key):
            execution = executions.get(plan.layer.name)
            pending_entry = pending_by_layer.get(plan.layer.name)
            if pending_entry is not None:
                task_status = "pending"
            elif plan.layer.name in quota_failed_layers:
                task_status = "failed"
            elif execution is not None:
                task_status = execution.status
            else:
                task_status = "not_started"
            schedule_tasks.append(
                {
                    "sort_key": plan.task_sort_key,
                    "layer": plan.layer.name,
                    "status": task_status,
                    "budget_cost": plan.cost,
                    "cache_hits": dict(plan.hits),
                    "force_downstream_distillation": plan.force_distill,
                }
            )
        business_codes = sorted(
            {record.code for record in failures}
            | {entry["code"] for entry in pending}
            | {event["business_code"] for event in recovery_events}
        )
        success_business = {}
        for report_layer in report_layers:
            if report_layer.get("status") not in {"ok", "quantized_only"}:
                continue
            success_business[report_layer["name"]] = {
                public_stage: report_layer[public_stage]
                for public_stage in ("calibration", "quantization", "error", "distillation")
                if public_stage in report_layer
            }
        result_payload = {
            "identity": prepared.identity,
            "effective_layer_configs": {
                name: effective[name].as_dict() for name in sorted(effective)
            },
            "business_results": success_business,
            "failures": sorted(
                (record.as_dict() for record in failures),
                key=lambda item: (item.get("layer") or "", item.get("stage") or "", item.get("business_code") or ""),
            ),
            "pending": sorted(pending, key=lambda item: (item["layer"], item["code"])),
            "injections": prepared.injection.reports_sorted(),
        }
        report = {
            "schema_version": SCHEMA_VERSION,
            "pipeline_version": PIPELINE_VERSION,
            "status": report_status,
            "business_codes": business_codes,
            "input": prepared.input_summary,
            "effective_config": format_effective_config_summary(config),
            "quantization_gear": {
                "profile": prepared.identity["quantization_profile"],
                "fractional_bits": prepared.identity["fractional_bits"],
                "layer_target_bits": {
                    name: effective[name].target_bits for name in sorted(effective)
                },
                "layer_fractional_bits": {
                    name: effective[name].fractional_bits for name in sorted(effective)
                },
            },
            "fingerprints": {
                "audit_config_hash": config.config_hash,
                "model_config_hash": prepared.identity["config_hash"],
                "weight_pack_fingerprint": prepared.identity["weight_pack_fingerprint"],
                "calibration_set_hash": prepared.identity["calibration_hash"],
            },
            "runtime": {
                **config.runtime.as_dict(),
                "batch_consumed_point_budget": batch_consumed,
                "completed_layer_count": len(completed_layers),
                "scheduled_layer_count": len(executions),
            },
            "snapshot": {
                "status": final_snapshot_status,
                "recovery": recovery_status,
                "previous_status": previous_status,
                "global_version": snapshot_global_version(prepared.identity),
                "stage_versions": stage_versions,
                "completed_layers": list(completed_layers),
                "events": list(recovery_events),
            },
            "layers": report_layers,
            "cache_ledger": ledger,
            "schedule": {
                "layer_order": layer_order,
                "max_workers": config.runtime.max_workers,
                "tasks": schedule_tasks,
            },
            "pending": sorted(pending, key=lambda item: (item["layer"], item["code"])),
            "failures": [
                record.as_dict()
                for record in sorted(
                    failures,
                    key=lambda item: (
                        item.layer or "",
                        item.stage or "",
                        item.code,
                        item.injection_id or "",
                    ),
                )
            ],
            "injections": prepared.injection.reports_sorted(),
            "events": list(recovery_events),
            "result_fingerprint": sha256_json(result_payload),
        }
        return report


def _matching_cache(
    record: dict[str, Any] | None,
    cache_key: str,
) -> dict[str, Any] | None:
    if not isinstance(record, dict):
        return None
    if record.get("cache_key") != cache_key:
        return None
    if not isinstance(record.get("data"), dict):
        return None
    return record


def result_keys_for(result_keys: Mapping[str, Mapping[str, str]], layer: str) -> dict[str, str]:
    return {
        stage: result_keys[stage][layer]
        for stage in result_keys
        if layer in result_keys[stage]
    }


def run_pipeline(
    sample_dir: str | Path,
    output_dir: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    return QuantizationPipeline(sample_dir, output_dir, **kwargs).run()
