"""Incremental workbench engine, command semantics and interactive summaries."""

from __future__ import annotations

import copy
import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .calibration import deployment_entry, evaluate_layer
from .errors import (
    E_COMMAND_INVALID,
    E_CONFIG_INVALID,
    E_LAYER_INVALID,
    E_POINT_BUDGET_PENDING,
    E_RESULT_STALE,
    E_SHAPE_MISMATCH,
    Issue,
    WorkbenchError,
    issue_sort_key,
)
from .loading import (
    load_calibration,
    normalize_scheme,
    parse_calibration_payload,
    validate_replace_weight_payload,
)
from .models import (
    CalibrationSample,
    CalibrationSet,
    Command,
    EngineState,
    LayerConfig,
    WeightLayer,
    WorkbenchConfig,
)
from .quantization import DeadlineExceeded, calibrate_layer, requantize_layer
from .scheduler import (
    BatchExecution,
    BudgetPlan,
    build_budget_plan,
    execute_batch,
    point_budget_pending_issues,
)
from .snapshot import SnapshotStore
from .util import clean_float, fingerprint, stage_cache_key


@dataclass(slots=True)
class WorkerResult:
    layer: str
    calibration: dict[str, Any] | None
    quantization: dict[str, Any] | None
    evaluation: dict[str, Any] | None
    deployment: dict[str, Any] | None
    actions: dict[str, str]
    issues: list[Issue]
    cache_hits: list[str]
    cache_misses: list[str]
    elapsed_ms: float


@dataclass(slots=True)
class CommandOutcome:
    changed: bool
    affected_layers: list[str] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)
    reason: str = ""


class WorkbenchEngine:
    """A reproducible incremental compression workbench.

    The stable payload intentionally excludes generations and wall-clock values;
    those are operational metadata emitted in interactive summaries and reports.
    """

    def __init__(
        self,
        config: WorkbenchConfig,
        weights: dict[str, WeightLayer],
        calibration: CalibrationSet,
        *,
        snapshot_store: SnapshotStore | None = None,
        resume: bool = True,
        expected_version: int | None = None,
        base_issues: list[Issue] | None = None,
        command_base_dir: str | Path | None = None,
    ) -> None:
        self.config = config
        self.state = EngineState(
            weights=weights,
            config=config,
            calibration=calibration,
            generation=0,
            command_count=0,
        )
        self.snapshot_store = snapshot_store
        self.command_base_dir = Path(command_base_dir) if command_base_dir else None
        self._base_config_hash = config.file_hash or config.config_fingerprint()
        self._base_issues = list(base_issues or [])
        for unknown_layer in sorted(set(config.layers) - set(weights)):
            self._base_issues.append(
                Issue(
                    E_LAYER_INVALID,
                    f"configuration references unknown layer {unknown_layer!r}",
                    layer=unknown_layer,
                )
            )
        self.failures: list[Issue] = list(self._base_issues)
        self.failures.sort(key=lambda item: (issue_sort_key(item), item.message))
        self._cache: dict[str, dict[str, dict[str, Any]]] = {
            "calibration": {},
            "quantization": {},
            "evaluation": {},
            "deployment": {},
        }
        self._invalidation_ledger: list[dict[str, Any]] = []
        self._batch_diagnostics: list[dict[str, Any]] = []
        self._cache_stats = {
            "hits": [],
            "misses": [],
            "computed_points": 0,
            "reused_points": 0,
        }
        self._last_complete_generation = 0
        self._latest_stable_payload: dict[str, Any] | None = None
        self._last_affected_layers: list[str] = []
        self._last_execution: dict[str, Any] | None = None
        self._budget_pending_layers: list[str] = []
        self._blocked_layers: set[str] = set()
        self._last_command_elapsed_ms = 0.0
        self._lock = threading.RLock()

        if snapshot_store is not None and resume:
            manifest = snapshot_store.open(
                global_version=config.snapshot_version,
                config_hash=self._base_config_hash,
                expected_version=expected_version,
            )
            if manifest is not None:
                self._restore_snapshot(snapshot_store, manifest)

        self._executor = ThreadPoolExecutor(
            max_workers=config.max_threads,
            thread_name_prefix="quant-layer",
        )
        # Pay thread-start cost outside the first interactive command.
        list(self._executor.map(lambda _: None, range(config.max_threads)))
        if not self._needs_work_layers():
            self._mark_complete_if_ready()

    # ------------------------------------------------------------------
    # Construction / resume helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_config_files(
        cls,
        config: WorkbenchConfig,
        weight_path: str | Path,
        calibration_path: str | Path,
        *,
        snapshot_dir: str | Path | None = None,
        resume: bool = True,
        expected_version: int | None = None,
        command_base_dir: str | Path | None = None,
    ) -> "WorkbenchEngine":
        from .loading import load_weight_pack

        weights, weight_issues = load_weight_pack(weight_path)
        calibration, calibration_issues = load_calibration(calibration_path)
        store = SnapshotStore(snapshot_dir) if snapshot_dir else None
        return cls(
            config,
            weights,
            calibration,
            snapshot_store=store,
            resume=resume,
            expected_version=expected_version,
            base_issues=weight_issues + calibration_issues,
            command_base_dir=command_base_dir,
        )

    @classmethod
    def from_state(
        cls,
        config: WorkbenchConfig,
        weights: dict[str, WeightLayer],
        calibration: CalibrationSet,
        *,
        base_issues: list[Issue] | None = None,
    ) -> "WorkbenchEngine":
        cloned_config = _clone_config(config)
        cloned_weights = {name: weight.copy() for name, weight in weights.items()}
        return cls(
            cloned_config,
            cloned_weights,
            calibration.copy(),
            snapshot_store=None,
            resume=False,
            base_issues=list(base_issues or []),
        )

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def __enter__(self) -> "WorkbenchEngine":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _restore_snapshot(
        self, store: SnapshotStore, manifest: dict[str, Any]
    ) -> None:
        artifacts = store.load_artifacts()
        for stage in ("quantization", "evaluation", "deployment"):
            self._cache[stage] = dict(artifacts.get(stage, {}))
        calibration_cache = artifacts.get("calibration", {})
        for layer, payload in calibration_cache.items():
            self._cache["calibration"][layer] = payload

        runtime = store.load_runtime_state()
        if runtime:
            config_payload = runtime.get("config")
            if isinstance(config_payload, dict):
                layers_raw = config_payload.get("layers", {})
                if isinstance(layers_raw, dict):
                    restored: dict[str, LayerConfig] = {}
                    for layer, value in layers_raw.items():
                        if isinstance(value, dict):
                            restored[layer] = LayerConfig(
                                bits=int(value["bits"]),
                                group_size=int(value["group_size"]),
                                fractional_bits=int(value["fractional_bits"]),
                                scheme=str(value["scheme"]),
                            )
                    self.config.layers = restored
                defaults_raw = config_payload.get("defaults")
                if isinstance(defaults_raw, dict):
                    self.config.defaults = LayerConfig(
                        bits=int(defaults_raw["bits"]),
                        group_size=int(defaults_raw["group_size"]),
                        fractional_bits=int(defaults_raw["fractional_bits"]),
                        scheme=str(defaults_raw["scheme"]),
                    )
                if "distillation_temperature" in config_payload:
                    self.config.distillation_temperature = float(
                        config_payload["distillation_temperature"]
                    )
                if "significant_digits" in config_payload:
                    self.config.significant_digits = int(config_payload["significant_digits"])
                if "point_batch_budget" in config_payload:
                    self.config.point_batch_budget = int(config_payload["point_batch_budget"])
                if "max_threads" in config_payload:
                    restored_threads = int(config_payload["max_threads"])
                    if restored_threads > self.config.thread_limit:
                        raise WorkbenchError(
                            E_CONFIG_INVALID,
                            "snapshot max_threads exceeds configured thread limit",
                        )
                    self.config.max_threads = restored_threads
            weights_payload = runtime.get("weights")
            if isinstance(weights_payload, list):
                restored_weights: dict[str, WeightLayer] = {}
                for record in weights_payload:
                    if not isinstance(record, dict):
                        continue
                    layer = str(record["layer"])
                    shape = tuple(int(item) for item in record["shape"])
                    array = np.asarray(record["weights"], dtype=np.float64).reshape(shape)
                    restored_weights[layer] = WeightLayer(
                        layer=layer,
                        version=int(record["version"]),
                        shape=shape,
                        weights=array,
                        order=int(record.get("order", len(restored_weights))),
                        line_number=record.get("line_number"),
                    )
                if restored_weights:
                    self.state.weights = restored_weights
            calibration_payload = runtime.get("calibration")
            if isinstance(calibration_payload, dict):
                samples: list[CalibrationSample] = []
                for sample in calibration_payload.get("samples", []):
                    samples.append(
                        CalibrationSample(
                            sample_id=str(sample["sample_id"]),
                            activations={
                                str(layer): np.asarray(values, dtype=np.float64).reshape(-1)
                                for layer, values in sample.get("activations", {}).items()
                            },
                        )
                    )
                if samples:
                    self.state.calibration = CalibrationSet(samples=samples)

        self.state.generation = int(
            runtime.get("generation", manifest.get("desired_generation", manifest.get("generation", 0)))
            if runtime
            else manifest.get("desired_generation", manifest.get("generation", 0))
        )
        self.state.command_count = int(
            runtime.get("command_count", manifest.get("command_count", 0))
            if runtime
            else manifest.get("command_count", 0)
        )
        self._last_complete_generation = int(manifest.get("generation", 0))
        self._invalidation_ledger = list((runtime or {}).get("invalidation_ledger", []))
        self._blocked_layers = set((runtime or {}).get("blocked_layers", []))
        restored_failures: list[Issue] = []
        for failure in (runtime or {}).get("failures", []):
            if not isinstance(failure, dict):
                continue
            restored_failures.append(
                Issue(
                    code=str(failure.get("code", "E_INTERNAL")),
                    message=str(failure.get("message", "")),
                    command_index=failure.get("command_index"),
                    line_number=failure.get("line_number"),
                    layer=failure.get("layer"),
                    sample_id=failure.get("sample_id"),
                    details=failure.get("details") or {},
                )
            )
        self._record_issues(restored_failures)
        if manifest.get("calibration_hash") and manifest.get("calibration_hash") != self.state.calibration.content_fingerprint:
            # Calibration content changed outside the snapshot: evaluation and
            # deployment keys will naturally miss while quantization remains reusable.
            self._cache["evaluation"] = {}
            self._cache["deployment"] = {}

    # ------------------------------------------------------------------
    # Fingerprints and cache validity
    # ------------------------------------------------------------------
    def _layer_config(self, layer: str) -> LayerConfig:
        return self.config.layer_config(layer)

    def _calibration_cache_key(self, layer: str) -> str:
        weight = self.state.weights[layer]
        config = self._layer_config(layer)
        return stage_cache_key(
            "calibration",
            weight.content_fingerprint,
            config.bits,
            config.group_size,
            self.config.significant_digits,
        )

    def _quantization_cache_key(self, layer: str) -> str:
        return stage_cache_key(
            "quantization",
            self._calibration_cache_key(layer),
            self._layer_config(layer).fractional_bits,
            self._layer_config(layer).scheme,
        )

    def _evaluation_cache_key(self, layer: str) -> str:
        return stage_cache_key(
            "evaluation",
            self._quantization_cache_key(layer),
            self.state.calibration.content_fingerprint,
            self.config.distillation_temperature,
        )

    def _deployment_cache_key(self, layer: str) -> str:
        return stage_cache_key(
            "deployment",
            self._quantization_cache_key(layer),
            self._evaluation_cache_key(layer),
            self.state.weights[layer].content_fingerprint,
        )

    def _cache_hit(self, stage: str, layer: str, expected_key: str) -> dict[str, Any] | None:
        payload = self._cache[stage].get(layer)
        if payload is not None and payload.get("cache_key") == expected_key:
            return payload
        return None

    def _needs_work(self, layer: str) -> bool:
        if self._cache_hit("quantization", layer, self._quantization_cache_key(layer)) is None:
            return True
        if self._cache_hit("evaluation", layer, self._evaluation_cache_key(layer)) is None:
            return True
        if self._cache_hit("deployment", layer, self._deployment_cache_key(layer)) is None:
            return True
        return False

    def _needs_work_layers(self) -> list[str]:
        return [layer for layer in sorted(self.state.weights) if self._needs_work(layer)]

    def _current_complete_layers(self) -> list[str]:
        return [
            layer
            for layer in sorted(self.state.weights)
            if not self._needs_work(layer)
        ]

    # ------------------------------------------------------------------
    # Worker and stage computation
    # ------------------------------------------------------------------
    def _compute_layer(
        self,
        layer: str,
        deadline: float | None,
        cancel_event: threading.Event,
        *,
        command_index: int | None,
    ) -> WorkerResult:
        started = time.perf_counter()
        weight = self.state.weights[layer]
        config = self._layer_config(layer)
        calibration_key = self._calibration_cache_key(layer)
        quantization_key = self._quantization_cache_key(layer)
        evaluation_key = self._evaluation_cache_key(layer)
        deployment_key = self._deployment_cache_key(layer)
        hits: list[str] = []
        misses: list[str] = []
        actions: dict[str, str] = {}
        issues: list[Issue] = []

        calibration_payload = self._cache_hit("calibration", layer, calibration_key)
        quantized = self._cache_hit("quantization", layer, quantization_key)
        if quantized is None:
            if calibration_payload is None:
                misses.append("calibration")
                calibration_payload = calibrate_layer(
                    weight,
                    config,
                    significant_digits=self.config.significant_digits,
                    deadline=deadline,
                    cancel_event=cancel_event,
                )
                actions["calibration"] = "recompute"
            else:
                hits.append("calibration")
                actions["calibration"] = "reuse"
            misses.append("quantization")
            quantized = requantize_layer(
                weight,
                calibration_payload,
                config,
                deadline=deadline,
                cancel_event=cancel_event,
            )
            actions["quantization"] = "recompute"
        else:
            hits.append("quantization")
            actions["quantization"] = "reuse"
            if calibration_payload is None:
                calibration_payload = self._calibration_stage_payload(quantized)
                actions["calibration"] = "reuse"
            else:
                hits.append("calibration")
                actions["calibration"] = "reuse"

        evaluation = self._cache_hit("evaluation", layer, evaluation_key)
        if evaluation is None:
            misses.append("evaluation")
            evaluation, eval_issues = evaluate_layer(
                weight,
                quantized,
                self.state.calibration,
                temperature=self.config.distillation_temperature,
                command_index=command_index,
                deadline=deadline,
                cancel_event=cancel_event,
            )
            issues.extend(eval_issues)
            actions["evaluation"] = "recompute"
        else:
            hits.append("evaluation")
            actions["evaluation"] = "reuse"

        deployment = None
        if evaluation is not None:
            deployment = self._cache_hit("deployment", layer, deployment_key)
            if deployment is None:
                misses.append("deployment")
                deployment = deployment_entry(weight, quantized, evaluation)
                actions["deployment"] = "recompute"
            else:
                hits.append("deployment")
                actions["deployment"] = "reuse"
        else:
            actions["deployment"] = "blocked_by_evaluation"

        return WorkerResult(
            layer=layer,
            calibration=calibration_payload,
            quantization=quantized,
            evaluation=evaluation,
            deployment=deployment,
            actions=actions,
            issues=issues,
            cache_hits=hits,
            cache_misses=misses,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )

    def _calibration_stage_payload(self, quantized: dict[str, Any]) -> dict[str, Any]:
        layer = quantized["layer"]
        return {
            "schema": "quant-workbench-calibration-v1",
            "layer": layer,
            "version": quantized["version"],
            "shape": quantized["shape"],
            "order": quantized["order"],
            "content_fingerprint": quantized["content_fingerprint"],
            "calibration_cache_key": quantized["calibration_cache_key"],
            "cache_key": quantized["calibration_cache_key"],
            "config_fields": quantized["calibration_config_fields"],
            "calibration_sample_ids": [],
            "parameters": {
                "bits": quantized["parameters"]["bits"],
                "group_size": quantized["parameters"]["group_size"],
                "significant_digits": quantized["parameters"]["significant_digits"],
            },
            "groups": quantized["groups"],
            "symmetric_scales": quantized["symmetric_scales"],
            "symmetric_zero_points": quantized.get(
                "symmetric_zero_points", [0] * len(quantized["symmetric_scales"])
            ),
            "asymmetric_scales": quantized["asymmetric_scales"],
            "asymmetric_zero_points": quantized["asymmetric_zero_points"],
            "group_ids": quantized["group_ids"],
            "point_count": quantized["point_count"],
        }

    def _apply_worker_result(self, result: WorkerResult) -> None:
        layer = result.layer
        if result.calibration is not None:
            self._cache["calibration"][layer] = result.calibration
        if result.quantization is not None:
            self._cache["quantization"][layer] = result.quantization
        if result.evaluation is not None:
            self._cache["evaluation"][layer] = result.evaluation
            self._blocked_layers.discard(layer)
        elif result.evaluation is None:
            self._blocked_layers.add(layer)
        if result.deployment is not None:
            self._cache["deployment"][layer] = result.deployment
        for stage in result.cache_hits:
            self._cache_stats["hits"].append({"stage": stage, "layer": layer})
            if stage == "quantization":
                self._cache_stats["reused_points"] += self.state.weights[layer].point_count
        for stage in result.cache_misses:
            self._cache_stats["misses"].append({"stage": stage, "layer": layer})
            if stage == "quantization":
                self._cache_stats["computed_points"] += self.state.weights[layer].point_count
        self._record_issues(result.issues)

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------
    def _record_issues(self, issues: list[Issue]) -> None:
        existing = {(issue_sort_key(item), item.message) for item in self.failures}
        for issue in issues:
            key = (issue_sort_key(issue), issue.message)
            if key not in existing:
                self.failures.append(issue)
                existing.add(key)
        self.failures.sort(key=lambda item: (issue_sort_key(item), item.message))

    def _issue(
        self,
        command: Command,
        code: str,
        message: str,
        *,
        layer: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> Issue:
        return Issue(
            code,
            message,
            command_index=command.command_index,
            line_number=command.line_number,
            layer=layer,
            details=details or {},
        )

    def _all_layers_in_order(self) -> list[str]:
        return sorted(self.state.weights)

    def _downstream_closure(self, layers: set[str]) -> set[str]:
        order = self.state.layer_order()
        positions = {layer: index for index, layer in enumerate(order)}
        affected: set[str] = set()
        for layer in layers:
            if layer not in positions:
                continue
            affected.update(order[positions[layer] :])
        return affected

    def _resolve_target_layers(
        self, command: Command, *, wildcard_allowed: bool = True
    ) -> tuple[list[str] | None, list[Issue]]:
        layer = command.layer
        if layer in {"*", "all", "__all__"}:
            if wildcard_allowed:
                return self._all_layers_in_order(), []
            return None, [
                self._issue(command, E_LAYER_INVALID, "wildcard layer is not allowed")
            ]
        if not layer:
            return None, [
                self._issue(command, E_LAYER_INVALID, "command requires a layer")
            ]
        if layer not in self.state.weights:
            return None, [
                self._issue(command, E_LAYER_INVALID, f"unknown layer {layer!r}", layer=layer)
            ]
        return [layer], []

    def _value_from(self, command: Command, *names: str) -> Any:
        for name in names:
            if name in command.payload:
                return command.payload[name]
        return None

    def _apply_command(self, command: Command) -> CommandOutcome:
        op = command.op
        issues: list[Issue] = []
        changed_layers: set[str] = set()
        affected_layers: set[str] = set()
        reason = op

        if op == "set_temperature":
            value = self._value_from(
                command,
                "temperature",
                "value",
                "new_temperature",
                "distillation_temperature",
            )
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return CommandOutcome(
                    False,
                    issues=[self._issue(command, E_CONFIG_INVALID, "temperature must be a number")],
                )
            value = float(value)
            if not math.isfinite(value) or value <= 0:
                return CommandOutcome(
                    False,
                    issues=[self._issue(command, E_CONFIG_INVALID, "temperature must be positive and finite")],
                )
            self.config.distillation_temperature = value
            affected_layers = set(self.state.weights)
            for layer in affected_layers:
                self._cache["evaluation"].pop(layer, None)
                self._cache["deployment"].pop(layer, None)
            changed_layers = set(affected_layers)
            self.state.config_revision += 1

        elif op == "replace_calibration":
            new_calibration: CalibrationSet | None = None
            path_value = self._value_from(command, "path", "calibration_path")
            if path_value is not None:
                if not isinstance(path_value, str) or not path_value:
                    return CommandOutcome(
                        False,
                        issues=[self._issue(command, E_CONFIG_INVALID, "calibration path must be a non-empty string")],
                    )
                candidates = [Path(path_value)]
                if not Path(path_value).is_absolute():
                    if self.command_base_dir is not None:
                        candidates.append(self.command_base_dir / path_value)
                    if self.config.source_path:
                        candidates.append(Path(self.config.source_path).parent / path_value)
                source_path = next((candidate for candidate in candidates if candidate.exists()), None)
                if source_path is None:
                    return CommandOutcome(
                        False,
                        issues=[
                            self._issue(
                                command,
                                "E_IO",
                                f"calibration replacement file not found: {path_value}",
                            )
                        ],
                    )
                new_calibration, calibration_issues = load_calibration(
                    source_path, command_index=command.command_index
                )
                issues.extend(calibration_issues)
            else:
                samples_value = self._value_from(
                    command, "samples", "calibration_samples", "calibration"
                )
                new_calibration, calibration_issues = parse_calibration_payload(
                    samples_value,
                    command_index=command.command_index,
                    line_number=command.line_number,
                )
                issues.extend(calibration_issues)
            if issues or new_calibration is None or not new_calibration.samples:
                if not issues:
                    issues.append(
                        self._issue(command, E_CONFIG_INVALID, "calibration replacement is empty")
                    )
                return CommandOutcome(False, issues=issues)
            self.state.calibration = new_calibration
            affected_layers = set(self.state.weights)
            for layer in affected_layers:
                self._cache["evaluation"].pop(layer, None)
                self._cache["deployment"].pop(layer, None)
            changed_layers = set(affected_layers)
            self.state.config_revision += 1

        elif op == "replace_weight":
            layer = command.layer
            if not layer or layer in {"*", "all", "__all__"}:
                return CommandOutcome(
                    False,
                    issues=[self._issue(command, E_LAYER_INVALID, "replace_weight requires one layer")],
                )
            existing = self.state.weights.get(layer)
            replacement, replacement_issues = validate_replace_weight_payload(command, existing)
            issues.extend(replacement_issues)
            if replacement is None:
                return CommandOutcome(False, issues=issues)
            if existing is not None and replacement.version <= existing.version:
                return CommandOutcome(
                    False,
                    issues=[
                        self._issue(
                            command,
                            E_LAYER_INVALID,
                            f"replacement version {replacement.version} must be greater than {existing.version}",
                            layer=layer,
                        )
                    ],
                )
            replacement.order = existing.order if existing else len(self.state.weights)
            self.state.weights[layer] = replacement
            affected_layers = self._downstream_closure({layer})
            for name in affected_layers:
                for stage in ("calibration", "quantization", "evaluation", "deployment"):
                    self._cache[stage].pop(name, None)
            changed_layers = {layer}
            self.state.config_revision += 1

        elif op in {"set_bits", "set_group_size", "set_fractional_bits", "set_scheme"}:
            targets, target_issues = self._resolve_target_layers(command)
            if target_issues:
                return CommandOutcome(False, issues=target_issues)
            assert targets is not None
            if op == "set_bits":
                value = self._value_from(
                    command, "bits", "value", "target_bits", "new_bits", "bit_width"
                )
                if isinstance(value, bool) or not isinstance(value, int):
                    return CommandOutcome(False, issues=[self._issue(command, E_CONFIG_INVALID, "bits must be an integer")])
                if value not in self.config.allowed_bit_widths:
                    return CommandOutcome(
                        False,
                        issues=[
                            self._issue(
                                command,
                                E_CONFIG_INVALID,
                                f"bits={value} is not in allowed bit widths {list(self.config.allowed_bit_widths)}",
                            )
                        ],
                    )
                for layer in targets:
                    old = self._layer_config(layer)
                    self.config.layers[layer] = LayerConfig(
                        value, old.group_size, old.fractional_bits, old.scheme
                    )
            elif op == "set_group_size":
                value = self._value_from(
                    command,
                    "group_size",
                    "value",
                    "granularity",
                    "new_group_size",
                )
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    return CommandOutcome(False, issues=[self._issue(command, E_CONFIG_INVALID, "group_size must be a positive integer")])
                for layer in targets:
                    old = self._layer_config(layer)
                    self.config.layers[layer] = LayerConfig(
                        old.bits, value, old.fractional_bits, old.scheme
                    )
            elif op == "set_fractional_bits":
                value = self._value_from(
                    command,
                    "fractional_bits",
                    "value",
                    "fixed_fractional_bits",
                    "new_fractional_bits",
                )
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    return CommandOutcome(False, issues=[self._issue(command, E_CONFIG_INVALID, "fractional_bits must be a positive integer")])
                for layer in targets:
                    old = self._layer_config(layer)
                    self.config.layers[layer] = LayerConfig(
                        old.bits, old.group_size, value, old.scheme
                    )
            else:
                value = self._value_from(
                    command,
                    "scheme",
                    "value",
                    "new_scheme",
                    "quantization_scheme",
                )
                if not isinstance(value, str):
                    return CommandOutcome(False, issues=[self._issue(command, E_CONFIG_INVALID, "scheme must be symmetric or asymmetric")])
                value = normalize_scheme(value)
                if value not in {"symmetric", "asymmetric"}:
                    return CommandOutcome(False, issues=[self._issue(command, E_CONFIG_INVALID, "scheme must be symmetric or asymmetric")])
                for layer in targets:
                    old = self._layer_config(layer)
                    self.config.layers[layer] = LayerConfig(
                        old.bits, old.group_size, old.fractional_bits, value
                    )
            affected_layers = self._downstream_closure(set(targets))
            for name in affected_layers:
                if op in {"set_bits", "set_group_size"}:
                    for stage in ("calibration", "quantization", "evaluation", "deployment"):
                        self._cache[stage].pop(name, None)
                else:
                    for stage in ("quantization", "evaluation", "deployment"):
                        self._cache[stage].pop(name, None)
            changed_layers = set(targets)
            self.state.config_revision += 1
        else:
            return CommandOutcome(
                False,
                issues=[
                    self._issue(
                        command,
                        E_COMMAND_INVALID,
                        f"unsupported command op {command.op!r}",
                    )
                ],
            )

        for layer in affected_layers:
            self._blocked_layers.discard(layer)
        self._record_invalidation(command, changed_layers, affected_layers, reason)
        return CommandOutcome(True, affected_layers=sorted(affected_layers), reason=reason)

    def _record_invalidation(
        self,
        command: Command,
        changed_layers: set[str],
        affected_layers: set[str],
        reason: str,
    ) -> None:
        calibration_ids = self.state.calibration.sample_ids
        all_layers = sorted(self.state.weights)
        for layer in all_layers:
            if layer in affected_layers:
                action = "recompute"
                layer_reason = reason if layer in changed_layers else f"downstream_of:{reason}"
            else:
                action = "reuse"
                layer_reason = "fingerprint_unchanged"
            if reason in {"set_temperature", "replace_calibration"}:
                stages = {
                    "calibration": "reuse",
                    "quantization": "reuse",
                    "evaluation": "recompute" if action == "recompute" else "reuse",
                    "deployment": "recompute" if action == "recompute" else "reuse",
                }
                config_fields = (
                    ["distillation_temperature"]
                    if reason == "set_temperature"
                    else ["calibration_samples"]
                )
                sample_ids = calibration_ids if action == "recompute" else []
            elif reason in {"set_fractional_bits", "set_scheme"}:
                stages = {
                    "calibration": "reuse",
                    "quantization": "recompute" if action == "recompute" else "reuse",
                    "evaluation": "recompute" if action == "recompute" else "reuse",
                    "deployment": "recompute" if action == "recompute" else "reuse",
                }
                config_fields = (
                    ["fractional_bits"] if reason == "set_fractional_bits" else ["scheme"]
                )
                sample_ids = calibration_ids if action == "recompute" else []
            else:
                stages = {
                    stage: ("recompute" if action == "recompute" else "reuse")
                    for stage in ("calibration", "quantization", "evaluation", "deployment")
                }
                config_fields = {
                    "set_bits": ["bits"],
                    "set_group_size": ["group_size"],
                    "replace_weight": ["weights", "version"],
                }.get(reason, [])
                sample_ids = calibration_ids if action == "recompute" else []
            self._invalidation_ledger.append(
                {
                    "command_index": command.command_index,
                    "line_number": command.line_number,
                    "reason": layer_reason,
                    "layer": layer,
                    "action": action,
                    "stages": stages,
                    "config_fields": config_fields,
                    "calibration_sample_ids": sample_ids,
                }
            )

    # ------------------------------------------------------------------
    # Batch execution and summaries
    # ------------------------------------------------------------------
    def _point_counts(self, layers: list[str]) -> dict[str, int]:
        return {layer: self.state.weights[layer].point_count for layer in layers}

    def _execute_one_batch(
        self,
        *,
        deadline: float | None,
        command_index: int | None,
        interactive: bool,
    ) -> dict[str, Any]:
        pending_before = self._needs_work_layers()
        if not pending_before:
            return {
                "pending_before": [],
                "pending_after": [],
                "plan": None,
                "execution": None,
                "budget_pending": [],
                "progress": True,
            }
        schedulable = [layer for layer in pending_before if layer not in self._blocked_layers]
        if not schedulable:
            diagnostic = {
                "pending_before": pending_before,
                "pending_after": pending_before,
                "plan": None,
                "execution": None,
                "budget_pending": [],
                "progress": False,
                "blocked_layers": sorted(self._blocked_layers),
            }
            self._batch_diagnostics.append(diagnostic)
            return diagnostic
        point_counts = self._point_counts(schedulable)
        plan = build_budget_plan(point_counts, self.config.point_batch_budget)
        budget_issues = point_budget_pending_issues(plan, command_index=command_index)
        if not plan.batches:
            self._budget_pending_layers = list(plan.pending_layers)
            diagnostic = {
                "pending_before": pending_before,
                "pending_after": pending_before,
                "plan": plan.to_dict(),
                "execution": None,
                "budget_pending": [issue.to_dict() for issue in budget_issues],
                "progress": False,
            }
            self._batch_diagnostics.append(diagnostic)
            return diagnostic

        batch = list(plan.batches[0])
        execution = execute_batch(
            layers=batch,
            worker=lambda layer, task_deadline, cancel_event: self._compute_layer(
                layer,
                task_deadline,
                cancel_event,
                command_index=command_index,
            ),
            executor=self._executor,
            deadline=deadline,
            worker_delay_ms=self.config.worker_delay_ms,
        )
        for layer in sorted(execution.completed):
            self._apply_worker_result(execution.completed[layer])
        self._record_issues(execution.failures)
        pending_after = self._needs_work_layers()
        queue_after_first_batch = list(plan.pending_layers)
        if len(plan.batches) > 1:
            planned_remaining = [
                layer for later in plan.batches[1:] for layer in later
            ]
            queue_after_first_batch.extend(planned_remaining)
        if execution.pending_layers:
            queue_after_first_batch.extend(execution.pending_layers)
        # E_POINT_BUDGET_PENDING describes budget pressure, not an ordinary
        # wall-clock timeout inside a single batch.
        self._budget_pending_layers = sorted(
            set(queue_after_first_batch + plan.pending_layers)
        )
        if not self._budget_pending_layers and execution.timed_out:
            self._budget_pending_layers = []
        diagnostic = {
            "pending_before": pending_before,
            "pending_after": pending_after,
            "plan": plan.to_dict(),
            "execution": execution.to_dict(),
            "budget_pending": [issue.to_dict() for issue in budget_issues],
            "progress": len(pending_after) < len(pending_before)
            or bool(execution.completed),
            "blocked_layers": sorted(self._blocked_layers),
            "interactive": interactive,
        }
        self._batch_diagnostics.append(diagnostic)
        return diagnostic

    def _aggregate_summary(self) -> dict[str, Any]:
        layers = self._current_complete_layers()
        max_abs_parts: list[float] = []
        mse_weighted_sum = 0.0
        mse_weight = 0
        layer_errors: list[dict[str, Any]] = []
        for layer in layers:
            evaluation = self._cache["evaluation"].get(layer)
            if not evaluation:
                continue
            layer_max = float(evaluation["max_abs_error"])
            layer_mse = float(evaluation["mse"])
            weight = int(evaluation.get("sample_count", 0))
            sample_output_counts = sum(len(sample["pre_deviation"]) for sample in evaluation["samples"])
            weight = max(1, sample_output_counts)
            max_abs_parts.append(layer_max)
            mse_weighted_sum += layer_mse * weight
            mse_weight += weight
            layer_errors.append(
                {
                    "layer": layer,
                    "max_abs_error": clean_float(layer_max),
                    "mse": clean_float(layer_mse),
                    "corrected_max_abs_error": clean_float(
                        evaluation["corrected_max_abs_error"]
                    ),
                    "corrected_mse": clean_float(evaluation["corrected_mse"]),
                    "sample_count": int(evaluation["sample_count"]),
                }
            )
        return {
            "completed_layers": len(layers),
            "total_layers": len(self.state.weights),
            "max_abs_error": clean_float(max(max_abs_parts)) if max_abs_parts else None,
            "mse": clean_float(mse_weighted_sum / mse_weight) if mse_weight else None,
            "layer_errors": layer_errors,
        }

    def _build_stable_payload(self) -> dict[str, Any]:
        complete_layers = self._current_complete_layers()
        layer_payloads: list[dict[str, Any]] = []
        for layer in complete_layers:
            quantized = self._cache["quantization"][layer]
            evaluation = self._cache["evaluation"][layer]
            deployment = self._cache["deployment"][layer]
            layer_payloads.append(
                {
                    "layer": layer,
                    "order": self.state.weights[layer].order,
                    "quantization": quantized,
                    "evaluation": evaluation,
                    "deployment": deployment,
                }
            )
        deployment_packages = [
            payload["deployment"]
            for payload in sorted(layer_payloads, key=lambda item: item["layer"])
        ]
        all_point_counts = {
            layer: self.state.weights[layer].point_count
            for layer in sorted(self.state.weights)
        }
        plan = build_budget_plan(all_point_counts, self.config.point_batch_budget)
        return {
            "schema": "quant-workbench-result-v1",
            "summary": self._aggregate_summary(),
            "layers": layer_payloads,
            "deployment": {
                "packages": deployment_packages,
                "layer_order": [item["layer"] for item in deployment_packages],
                "package_count": len(deployment_packages),
                "total_points": sum(item["point_count"] for item in deployment_packages),
                "calibration_hash": self.state.calibration.content_fingerprint,
                "config_hash": fingerprint(
                    {
                        "defaults": self.config.defaults.to_dict(),
                        "layers": {
                            layer: self.config.layer_config(layer).to_dict()
                            for layer in sorted(self.state.weights)
                        },
                        "temperature": self.config.distillation_temperature,
                        "significant_digits": self.config.significant_digits,
                    }
                ),
            },
            "scheduling": plan.to_dict(),
            "config": {
                "defaults": self.config.defaults.to_dict(),
                "layers": {
                    layer: self.config.layer_config(layer).to_dict()
                    for layer in sorted(self.state.weights)
                },
                "allowed_bit_widths": list(self.config.allowed_bit_widths),
                "distillation_temperature": self.config.distillation_temperature,
                "significant_digits": self.config.significant_digits,
                "point_batch_budget": self.config.point_batch_budget,
                "max_threads": self.config.max_threads,
            },
            "calibration": {
                "fingerprint": self.state.calibration.content_fingerprint,
                "sample_ids": self.state.calibration.sample_ids,
                "sample_count": len(self.state.calibration.samples),
            },
            "weights": [
                self.state.weights[layer].summary_dict()
                for layer in sorted(self.state.weights)
            ],
            "failures": [issue.to_dict() for issue in self.failures],
        }

    def _mark_complete_if_ready(self) -> bool:
        if self._needs_work_layers():
            return False
        self._latest_stable_payload = self._build_stable_payload()
        self._last_complete_generation = self.state.generation
        self._budget_pending_layers = []
        return True

    def _metric_source(self) -> dict[str, Any] | None:
        if self._latest_stable_payload is None:
            return None
        return self._latest_stable_payload.get("summary")

    def _interactive_summary(
        self,
        *,
        command: Command | None,
        affected_layers: list[str],
        started: float,
        budget_ms: float,
        stale: bool,
        pending_layers: list[str],
    ) -> dict[str, Any]:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        total_layers = len(self.state.weights)
        valid_layers = self._current_complete_layers()
        latest = self._metric_source()
        complete = not stale and latest is not None and len(valid_layers) == total_layers
        if complete:
            self._last_complete_generation = self.state.generation
        codes: list[str] = []
        if not complete:
            codes.append(E_RESULT_STALE)
            if self._budget_pending_layers:
                codes.append(E_POINT_BUDGET_PENDING)
        affected_progress_total = len(affected_layers) if affected_layers else total_layers
        affected_progress_done = len(
            [layer for layer in affected_layers if layer in valid_layers]
        ) if affected_layers else len(valid_layers)
        summary: dict[str, Any] = {
            "status": "complete" if complete else "stale",
            "is_latest": complete,
            "codes": codes,
            "requested_generation": self.state.generation,
            "result_generation": self._last_complete_generation if not complete else self.state.generation,
            "result_corresponds_to_generation": self._last_complete_generation if not complete else self.state.generation,
            "completed_layers": int(latest["completed_layers"]) if latest else 0,
            "total_layers": total_layers,
            "current_completed_layers": len(valid_layers),
            "max_abs_error": latest["max_abs_error"] if latest else None,
            "mse": latest["mse"] if latest else None,
            "affected_layers": list(affected_layers),
            "pending_layers": list(pending_layers),
            "progress": {
                "completed": affected_progress_done,
                "total": affected_progress_total,
                "ratio": (
                    1.0
                    if affected_progress_total == 0
                    else affected_progress_done / affected_progress_total
                ),
            },
            "command_index": command.command_index if command else None,
            "latency_budget_ms": budget_ms,
            "elapsed_ms": elapsed_ms,
            "within_budget": elapsed_ms <= budget_ms,
        }
        return summary

    def process_command(self, command: Command) -> dict[str, Any]:
        """Apply one edit and return a summary within the configured latency budget.

        At most one point-budget batch is run per interaction.  Remaining work
        stays queued and the returned result is explicitly marked stale.
        """

        started = time.perf_counter()
        budget_ms = float(self.config.latency_budget_ms)
        # Reserve a small amount for applying results and constructing the
        # response; compute loops are cooperative but cannot be pre-empted by
        # the Python runtime.
        reserve_ms = min(2.0, max(0.2, budget_ms * 0.15))
        deadline = started + max(0.0, budget_ms - reserve_ms) / 1000.0
        command_command_count = self.state.command_count + 1
        self.state.command_count = command_command_count
        outcome = self._apply_command(command)
        self._record_issues(outcome.issues)
        if outcome.changed:
            self.state.generation += 1
            self._last_affected_layers = list(outcome.affected_layers)
        else:
            self._last_affected_layers = []

        execution = self._execute_one_batch(
            deadline=deadline,
            command_index=command.command_index,
            interactive=True,
        )
        self._last_execution = execution
        pending = self._needs_work_layers()
        complete = not pending
        if complete:
            self._mark_complete_if_ready()
        stale = not complete
        summary = self._interactive_summary(
            command=command,
            affected_layers=self._last_affected_layers,
            started=started,
            budget_ms=budget_ms,
            stale=stale,
            pending_layers=pending,
        )
        self._last_command_elapsed_ms = summary["elapsed_ms"]
        return summary

    def drain_all(self, *, max_iterations: int = 10_000) -> dict[str, Any]:
        """Finish all queued batches outside the interactive deadline."""

        iterations = 0
        while self._needs_work_layers():
            pending_before = self._needs_work_layers()
            execution = self._execute_one_batch(
                deadline=None,
                command_index=None,
                interactive=False,
            )
            iterations += 1
            if iterations >= max_iterations:
                break
            if execution["plan"] is None:
                break
            if not execution["progress"]:
                break
        self._mark_complete_if_ready()
        return self.summary(current_command=None)

    def summary(self, *, current_command: Command | None = None) -> dict[str, Any]:
        latest = self._latest_stable_payload is not None
        pending = self._needs_work_layers()
        complete = latest and not pending
        return self._interactive_summary(
            command=current_command,
            affected_layers=self._last_affected_layers,
            started=time.perf_counter(),
            budget_ms=float(self.config.latency_budget_ms),
            stale=not complete,
            pending_layers=pending,
        )

    # ------------------------------------------------------------------
    # Results and reports
    # ------------------------------------------------------------------
    def result_payload(self) -> dict[str, Any]:
        """Stable, generation-free output for incremental/full equality checks."""

        if self._needs_work_layers():
            return self._build_stable_payload()
        if self._latest_stable_payload is None:
            self._mark_complete_if_ready()
        assert self._latest_stable_payload is not None
        return copy.deepcopy(self._latest_stable_payload)

    def full_rebuild(self) -> dict[str, Any]:
        """Clear all stage caches and rebuild the current state from zero."""

        self._cache = {
            "calibration": {},
            "quantization": {},
            "evaluation": {},
            "deployment": {},
        }
        self._latest_stable_payload = None
        self._last_complete_generation = 0
        self._batch_diagnostics = []
        self._blocked_layers = set()
        self._cache_stats = {
            "hits": [],
            "misses": [],
            "computed_points": 0,
            "reused_points": 0,
        }
        self.state.generation += 1
        self.drain_all()
        return self.result_payload()

    def diagnostics(self) -> dict[str, Any]:
        pending = self._needs_work_layers()
        budget_pending = list(self._budget_pending_layers)
        return {
            "schema": "quant-workbench-diagnostics-v1",
            "generation": self.state.generation,
            "latest_complete_generation": self._last_complete_generation,
            "command_count": self.state.command_count,
            "completed_layers": self._current_complete_layers(),
            "pending_layers": pending,
            "budget_pending_layers": budget_pending,
            "blocked_layers": sorted(self._blocked_layers),
            "point_budget_pending": [
                issue.to_dict()
                for issue in point_budget_pending_issues(
                    build_budget_plan(
                        self._point_counts(pending), self.config.point_batch_budget
                    )
                )
            ] if pending else [],
            "failure_count": len(self.failures),
            "failures": [issue.to_dict() for issue in self.failures],
            "invalidation_reuse_ledger": self._invalidation_ledger,
            "cache_stats": {
                "hits": list(self._cache_stats["hits"]),
                "misses": list(self._cache_stats["misses"]),
                "computed_points": self._cache_stats["computed_points"],
                "reused_points": self._cache_stats["reused_points"],
            },
            "batch_diagnostics": self._batch_diagnostics,
            "summary": self._metric_source(),
            "scheduling": self._build_stable_payload()["scheduling"],
            "last_command_elapsed_ms": self._last_command_elapsed_ms,
        }

    def _current_artifacts(self) -> dict[str, dict[str, dict[str, Any]]]:
        artifacts: dict[str, dict[str, dict[str, Any]]] = {
            "calibration": {},
            "quantization": {},
            "evaluation": {},
            "deployment": {},
        }
        for layer in sorted(self.state.weights):
            calibration = self._cache_hit(
                "calibration", layer, self._calibration_cache_key(layer)
            )
            quantization = self._cache_hit(
                "quantization", layer, self._quantization_cache_key(layer)
            )
            evaluation = self._cache_hit(
                "evaluation", layer, self._evaluation_cache_key(layer)
            )
            deployment = self._cache_hit(
                "deployment", layer, self._deployment_cache_key(layer)
            )
            if calibration is not None:
                artifacts["calibration"][layer] = calibration
            if quantization is not None:
                artifacts["quantization"][layer] = quantization
            if evaluation is not None:
                artifacts["evaluation"][layer] = evaluation
            if deployment is not None:
                artifacts["deployment"][layer] = deployment
        return artifacts

    def _runtime_state(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "weights": [
                {
                    "layer": layer,
                    "version": weight.version,
                    "shape": list(weight.shape),
                    "order": weight.order,
                    "line_number": weight.line_number,
                    "weights": weight.weights.tolist(),
                }
                for layer, weight in sorted(self.state.weights.items())
            ],
            "calibration": {
                "fingerprint": self.state.calibration.content_fingerprint,
                "samples": [
                    {
                        "sample_id": sample.sample_id,
                        "activations": {
                            layer: values.tolist()
                            for layer, values in sorted(sample.activations.items())
                        },
                    }
                    for sample in self.state.calibration.samples
                ],
            },
            "generation": self.state.generation,
            "command_count": self.state.command_count,
            "failures": [issue.to_dict() for issue in self.failures],
            "blocked_layers": sorted(self._blocked_layers),
            "invalidation_ledger": self._invalidation_ledger,
        }

    def commit_snapshot(self, *, diagnostics: dict[str, Any] | None = None) -> dict[str, Any] | None:
        if self.snapshot_store is None:
            return None
        artifacts = self._current_artifacts()
        ingest = {
            "layers": [
                self.state.weights[layer].summary_dict()
                for layer in sorted(self.state.weights)
            ],
            "config_hash": self._base_config_hash,
            "calibration_hash": self.state.calibration.content_fingerprint,
            "initial_issues": [issue.to_dict() for issue in self._base_issues],
            "completed_layers": self._current_complete_layers(),
        }
        report = diagnostics if diagnostics is not None else self.diagnostics()
        manifest = self.snapshot_store.commit(
            global_version=self.config.snapshot_version,
            config_hash=self._base_config_hash,
            calibration_hash=self.state.calibration.content_fingerprint,
            completed_layers=self._current_complete_layers(),
            generation=self._last_complete_generation,
            command_count=self.state.command_count,
            artifacts=artifacts,
            ingest=ingest,
            runtime_state=self._runtime_state(),
            diagnostics=report,
        )
        return manifest

    def maybe_commit(self) -> dict[str, Any] | None:
        if self.state.command_count % self.config.snapshot_commit_interval != 0:
            return None
        return self.commit_snapshot()

    def diagnostic_report(self) -> dict[str, Any]:
        stable = self.result_payload()
        return {
            **self.diagnostics(),
            "stable_result": stable,
            "snapshot_manifest": self.snapshot_store.manifest if self.snapshot_store else None,
            "config": self.config.to_dict(),
            "calibration": stable["calibration"],
        }


def _clone_config(config: WorkbenchConfig) -> WorkbenchConfig:
    return WorkbenchConfig(
        snapshot_version=config.snapshot_version,
        allowed_bit_widths=tuple(config.allowed_bit_widths),
        max_threads=config.max_threads,
        thread_limit=config.thread_limit,
        latency_budget_ms=config.latency_budget_ms,
        point_batch_budget=config.point_batch_budget,
        snapshot_commit_interval=config.snapshot_commit_interval,
        distillation_temperature=config.distillation_temperature,
        significant_digits=config.significant_digits,
        defaults=LayerConfig(**config.defaults.to_dict()),
        layers={
            layer: LayerConfig(**cfg.to_dict()) for layer, cfg in config.layers.items()
        },
        source_path=config.source_path,
        raw=copy.deepcopy(config.raw),
        file_hash=config.file_hash,
        worker_delay_ms=config.worker_delay_ms,
    )
