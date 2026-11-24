"""End-to-end staged execution engine."""

from __future__ import annotations

import copy
import json
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from .commands import apply_command
from .config import LabConfig, copy_state, state_hash, validate_state
from .conversion import conversion_artifact, conversion_artifact_key
from .errors import (
    E_INTERNAL,
    E_NONFINITE,
    E_PARAM_MISSING,
    E_SHAPE,
    LabError,
    make_failure,
    sorted_failures,
)
from .fingerprints import (
    activation_row_fingerprint,
    layernorm_output_cache_key,
    layernorm_stats_cache_key,
    parameter_set_fingerprint,
    softmax_cache_key,
    stage_content_hash,
)
from .fixed import FixedContext
from .loaders import load_all_sources, read_source_bundle, source_fingerprint
from .models import LayerTensor, LoadedSources, SourceBundle, WeightRow
from .operators import (
    layernorm_apply_from_stats,
    layernorm_statistics,
    softmax_row,
)
from .scheduler import build_schedule, execute_schedule
from .snapshots import SnapshotStore, load_state, save_state
from .util import (
    canonical_json,
    is_finite_number,
    round_half_away,
    sha256_obj,
    sha256_text,
    write_json,
)


SCHEMA_REPORT = "opexlab.report.v1"


def _row_refs(items: Iterable[Tuple[str, int]]) -> list[Dict[str, Any]]:
    return [
        {"layer": layer, "row": row}
        for layer, row in sorted(set(items), key=lambda item: (item[0], item[1]))
    ]


def _safe_source_fingerprint(loaded: LoadedSources) -> str:
    try:
        return source_fingerprint(loaded)
    except Exception:
        return sha256_text(
            json.dumps(
                loaded.load_fingerprint_parts,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=True,
            )
        )


def _layer_public(layer: LayerTensor) -> Dict[str, Any]:
    return {
        "layer": layer.name,
        "seed": layer.seed,
        "dim": layer.dim,
        "rows": [
            {"row": row, "values": list(values)}
            for row, values in layer.ordered_rows()
        ],
        "reference_scale": list(layer.reference_scale),
        "reference_bias": list(layer.reference_bias),
    }


def _weight_public(weights: Dict[Tuple[str, int], WeightRow]) -> list[Dict[str, Any]]:
    return [weights[key].source_payload() for key in sorted(weights)]


def _cache_record_valid(record: Any) -> bool:
    if not isinstance(record, dict):
        return False
    if not isinstance(record.get("cache_key"), str):
        return False
    if "content_hash" not in record:
        return False
    payload = record.get("result", record.get("artifact"))
    if not isinstance(payload, dict):
        return False
    dependencies = record.get("dependencies")
    if not isinstance(dependencies, dict):
        return False
    try:
        expected = stage_content_hash(result=payload, dependencies=dependencies)
    except Exception:
        return False
    return expected == record.get("content_hash")


def _cache_status_store() -> Dict[str, list[Dict[str, Any]]]:
    return {
        "version_conversion": [],
        "softmax": [],
        "layernorm_stats": [],
        "layernorm_output": [],
    }


class ExecutionEngine:
    def __init__(
        self,
        config: LabConfig,
        *,
        runtime_dir: Optional[Path] = None,
        with_injections: bool = False,
    ):
        self.config = config
        self.runtime_dir = (runtime_dir or config.runtime_dir).resolve()
        self.store = SnapshotStore(self.runtime_dir)
        self.state_path = self.runtime_dir / "state.json"
        self.with_injections = with_injections
        self.last_metrics: Dict[str, Any] = {}
        self.last_schedule: Dict[str, Any] = {}

    def load_state(self) -> Dict[str, Any]:
        from .config import default_state

        state = load_state(self.state_path)
        if state is None:
            state = default_state(self.config)
        return validate_state(self.config, state)

    def save_state(self, state: Dict[str, Any]) -> None:
        validate_state(self.config, state)
        save_state(self.state_path, state)

    def apply_and_run(
        self,
        command: Dict[str, Any],
        *,
        mode: str = "incremental",
    ) -> Dict[str, Any]:
        state = self.load_state()
        next_state = apply_command(self.config, state, command)
        self.save_state(next_state)
        return self.run(next_state, mode=mode)

    def run(
        self,
        state: Dict[str, Any],
        *,
        mode: str = "incremental",
        bundle: Optional[SourceBundle] = None,
        commit_snapshot: bool = True,
        injection_id: Optional[str] = None,
        _derive_cache_plan: bool = True,
    ) -> Dict[str, Any]:
        if mode not in ("incremental", "full"):
            raise ValueError("mode must be incremental or full")
        state = copy_state(state)
        validate_state(self.config, state)
        source_bundle = bundle or read_source_bundle(self.config)
        injection_id = injection_id or source_bundle.injection_id
        loaded = load_all_sources(
            self.config,
            source_bundle,
            activation_overrides=state.get("activation_overrides", {}),
        )
        source_fp = _safe_source_fingerprint(loaded)
        base_failures = list(loaded.failures)

        active_version = str(state["exchange_version"])
        target_version = "v2" if active_version == "v1" else "v1"
        active_source = loaded.weights_v1 if active_version == "v1" else loaded.weights_v2
        active_weights, override_failures = self._apply_parameters(
            loaded,
            active_source,
            state.get("parameter_overrides", {}),
            active_version,
        )
        base_failures.extend(override_failures)
        exchange_fp = self._exchange_fingerprint(active_weights, active_version, state)

        previous_manifest = self.store.load_manifest() if mode == "incremental" else None
        previous_cache = self.store.load_cache(previous_manifest, self.config.config_hash)
        plan_key_sets: Optional[Dict[str, set[str]]] = None
        if mode == "incremental":
            plan_key_sets = {
                stage: {
                    cache_key
                    for cache_key, record in records.items()
                    if _cache_record_valid(record)
                }
                for stage, records in previous_cache.items()
            }
        elif _derive_cache_plan and not injection_id:
            plan_key_sets = self._derive_previous_plan_key_sets(state, source_bundle)
        cache_status = _cache_status_store()
        failures: list[Dict[str, Any]] = list(base_failures)

        conversion_records: Dict[Tuple[str, int], Dict[str, Any]] = {}
        conversion_failures: list[Dict[str, Any]] = []
        conversion_artifacts: Dict[Tuple[str, int], Dict[str, Any]] = {}
        for layer_name in sorted(loaded.layers):
            layer = loaded.layers[layer_name]
            for row_index, _values in layer.ordered_rows():
                key = (layer_name, row_index)
                weight = active_weights.get(key)
                if weight is None:
                    continue
                cache_key = conversion_artifact_key(
                    self.config,
                    weight,
                    target_version,
                    exchange_version=active_version,
                    fractional_bits=int(state["fractional_bits"]),
                    quant_level=int(state["quant_level"]),
                )
                cached = previous_cache["version_conversion"].get(cache_key)
                if mode == "incremental" and _cache_record_valid(cached):
                    record = cached
                    cache_status["version_conversion"].append(
                        {"layer": layer_name, "row": row_index, "status": "reused"}
                    )
                else:
                    planned_reuse = (
                        mode == "full"
                        and plan_key_sets is not None
                        and cache_key in plan_key_sets.get("version_conversion", set())
                    )
                    cache_status["version_conversion"].append(
                        {
                            "layer": layer_name,
                            "row": row_index,
                            "status": "reused" if planned_reuse else "recomputed",
                        }
                    )
                    try:
                        artifact = conversion_artifact(self.config, weight, target_version)
                        public_artifact = {
                            "direction": artifact["direction"],
                            "source_version": artifact["source_version"],
                            "target_version": artifact["target_version"],
                            "source": artifact["source"],
                            "target": artifact["target"],
                            "sidecar": artifact["sidecar"],
                            "sidecar_hash": artifact["sidecar_hash"],
                            "artifact_fingerprint": artifact["artifact_fingerprint"],
                        }
                        dependencies = {
                            "parameter_fingerprint": parameter_set_fingerprint(weight, None),
                        }
                        content_hash = stage_content_hash(
                            result=public_artifact, dependencies=dependencies
                        )
                        record = {
                            "layer": layer_name,
                            "row": row_index,
                            "stage": "version_conversion",
                            "cache_key": cache_key,
                            "content_hash": content_hash,
                            "dependencies": dependencies,
                            "artifact": public_artifact,
                        }
                    except LabError as exc:
                        exc.injection_id = injection_id
                        conversion_failures.append(exc.failure())
                        continue
                    except Exception as exc:  # pragma: no cover - defensive
                        conversion_failures.append(
                            make_failure(
                                E_INTERNAL,
                                "version_conversion",
                                f"unexpected conversion failure: {exc}",
                                layer=layer_name,
                                row=row_index,
                                injection_id=injection_id,
                            )
                        )
                        continue
                conversion_records[key] = record
                conversion_artifacts[key] = record["artifact"]
        failures.extend(conversion_failures)

        task_layers = [
            (layer_name, loaded.layers[layer_name].dim, sorted(loaded.layers[layer_name].rows))
            for layer_name in sorted(loaded.layers)
        ]
        schedule = build_schedule(
            task_layers,
            batch_element_budget=int(state["batch_element_budget"]),
            per_layer_element_quota=int(state["per_layer_element_quota"]),
        )
        self.last_schedule = schedule
        failures.extend(schedule["failures"])

        context = FixedContext(
            fractional_bits=int(state["fractional_bits"]),
            quant_level=int(state["quant_level"]),
        )
        approximation = self.config.exp_approximation_name(context.quant_level)
        polynomial_order = self.config.exp_polynomial_order(context.quant_level)
        lock = threading.Lock()
        stage_failures: Dict[Tuple[str, int], list[Dict[str, Any]]] = {}

        softmax_records: Dict[Tuple[str, int], Dict[str, Any]] = {}
        stats_records: Dict[Tuple[str, int], Dict[str, Any]] = {}
        output_records: Dict[Tuple[str, int], Dict[str, Any]] = {}

        def worker(task: Dict[str, Any]) -> Dict[str, Any]:
            layer_name = task["layer"]
            row_index = task["row"]
            layer = loaded.layers[layer_name]
            values = layer.rows[row_index]
            local_failures: list[Dict[str, Any]] = []
            statuses: list[Tuple[str, Dict[str, Any]]] = []
            activation_fp = activation_row_fingerprint(layer_name, row_index, values)
            conversion = conversion_artifacts.get((layer_name, row_index))
            sidecar_hash = conversion.get("sidecar_hash") if conversion else None

            softmax_record: Optional[Dict[str, Any]] = None
            sm_key = softmax_cache_key(
                layer=layer_name,
                row=row_index,
                activation_fingerprint=activation_fp,
                exchange_version=active_version,
                fractional_bits=context.fractional_bits,
                quant_level=context.quant_level,
                exp_approximation=approximation,
            )
            cached = previous_cache["softmax"].get(sm_key)
            if mode == "incremental" and _cache_record_valid(cached):
                softmax_record = cached
                statuses.append(("softmax", {"layer": layer_name, "row": row_index, "status": "reused"}))
            else:
                planned_reuse = (
                    mode == "full"
                    and plan_key_sets is not None
                    and sm_key in plan_key_sets.get("softmax", set())
                )
                statuses.append(
                    (
                        "softmax",
                        {
                            "layer": layer_name,
                            "row": row_index,
                            "status": "reused" if planned_reuse else "recomputed",
                        },
                    )
                )
                try:
                    result = softmax_row(
                        values,
                        context,
                        polynomial_order=polynomial_order,
                        approximation_name=approximation,
                    )
                    dependencies = {
                        "activation_row_fingerprint": activation_fp,
                        "parameter_set": None,
                    }
                    softmax_record = {
                        "layer": layer_name,
                        "row": row_index,
                        "stage": "softmax",
                        "cache_key": sm_key,
                        "content_hash": stage_content_hash(result=result, dependencies=dependencies),
                        "dependencies": dependencies,
                        "result": result,
                    }
                except LabError as exc:
                    exc.injection_id = injection_id
                    local_failures.append(exc.failure())
                except Exception as exc:
                    local_failures.append(
                        make_failure(
                            E_INTERNAL,
                            "softmax",
                            f"unexpected softmax failure: {exc}",
                            layer=layer_name,
                            row=row_index,
                            injection_id=injection_id,
                        )
                    )

            stats_record: Optional[Dict[str, Any]] = None
            st_key = layernorm_stats_cache_key(
                layer=layer_name,
                row=row_index,
                activation_fingerprint=activation_fp,
                exchange_version=active_version,
                fractional_bits=context.fractional_bits,
                quant_level=context.quant_level,
                variance_floor=self.config.variance_floor,
            )
            cached = previous_cache["layernorm_stats"].get(st_key)
            if mode == "incremental" and _cache_record_valid(cached):
                stats_record = cached
                statuses.append(("layernorm_stats", {"layer": layer_name, "row": row_index, "status": "reused"}))
            else:
                planned_reuse = (
                    mode == "full"
                    and plan_key_sets is not None
                    and st_key in plan_key_sets.get("layernorm_stats", set())
                )
                statuses.append(
                    (
                        "layernorm_stats",
                        {
                            "layer": layer_name,
                            "row": row_index,
                            "status": "reused" if planned_reuse else "recomputed",
                        },
                    )
                )
                try:
                    statistics = layernorm_statistics(
                        values,
                        context,
                        variance_floor=self.config.variance_floor,
                    )
                    dependencies = {
                        "activation_row_fingerprint": activation_fp,
                        "parameter_set": None,
                    }
                    stats_record = {
                        "layer": layer_name,
                        "row": row_index,
                        "stage": "layernorm_stats",
                        "cache_key": st_key,
                        "content_hash": stage_content_hash(result=statistics, dependencies=dependencies),
                        "dependencies": dependencies,
                        "result": statistics,
                    }
                except LabError as exc:
                    exc.injection_id = injection_id
                    local_failures.append(exc.failure())
                except Exception as exc:
                    local_failures.append(
                        make_failure(
                            E_INTERNAL,
                            "layernorm",
                            f"unexpected LayerNorm statistics failure: {exc}",
                            layer=layer_name,
                            row=row_index,
                            injection_id=injection_id,
                        )
                    )

            output_record: Optional[Dict[str, Any]] = None
            weight = active_weights.get((layer_name, row_index))
            if stats_record is not None and weight is not None:
                scheme = weight.scheme
                bias_enabled = (
                    True
                    if active_version == "v1"
                    else bool(self.config.scheme_bias_policy.get(scheme, False))
                )
                param_fp = parameter_set_fingerprint(weight, sidecar_hash)
                out_key = layernorm_output_cache_key(
                    layer=layer_name,
                    row=row_index,
                    stats_cache_key=st_key,
                    stats_content_hash=stats_record["content_hash"],
                    parameter_fingerprint=param_fp,
                    exchange_version=active_version,
                    fractional_bits=context.fractional_bits,
                    quant_level=context.quant_level,
                    scheme=scheme,
                    bias_enabled=bias_enabled,
                )
                cached = previous_cache["layernorm_output"].get(out_key)
                if mode == "incremental" and _cache_record_valid(cached):
                    output_record = cached
                    statuses.append(("layernorm_output", {"layer": layer_name, "row": row_index, "status": "reused"}))
                else:
                    planned_reuse = (
                        mode == "full"
                        and plan_key_sets is not None
                        and out_key in plan_key_sets.get("layernorm_output", set())
                    )
                    statuses.append(
                        (
                            "layernorm_output",
                            {
                                "layer": layer_name,
                                "row": row_index,
                                "status": "reused" if planned_reuse else "recomputed",
                            },
                        )
                    )
                    try:
                        applied = layernorm_apply_from_stats(
                            stats_record["result"],
                            weight,
                            context,
                            path_version=active_version,
                            bias_enabled=bias_enabled,
                        )
                        dependencies = {
                            "activation_row_fingerprint": activation_fp,
                            "stats_cache_key": st_key,
                            "stats_content_hash": stats_record["content_hash"],
                            "parameter_fingerprint": param_fp,
                            "parameter_set": weight.source_payload(),
                            "conversion_sidecar_hash": sidecar_hash,
                        }
                        output_record = {
                            "layer": layer_name,
                            "row": row_index,
                            "stage": "layernorm_output",
                            "cache_key": out_key,
                            "content_hash": stage_content_hash(result=applied, dependencies=dependencies),
                            "dependencies": dependencies,
                            "result": applied,
                        }
                    except LabError as exc:
                        exc.injection_id = injection_id
                        local_failures.append(exc.failure())
                    except Exception as exc:
                        local_failures.append(
                            make_failure(
                                E_INTERNAL,
                                "layernorm",
                                f"unexpected LayerNorm output failure: {exc}",
                                layer=layer_name,
                                row=row_index,
                                injection_id=injection_id,
                            )
                        )
            elif weight is None:
                local_failures.append(
                    make_failure(
                        E_PARAM_MISSING,
                        "layernorm",
                        "no active-version parameter row for this activation row",
                        layer=layer_name,
                        row=row_index,
                        details={"exchange_version": active_version},
                        injection_id=injection_id,
                    )
                )

            return {
                "layer": layer_name,
                "row": row_index,
                "softmax": softmax_record,
                "stats": stats_record,
                "output": output_record,
                "failures": local_failures,
                "statuses": statuses,
                "parameter_fingerprint": (
                    parameter_set_fingerprint(weight, sidecar_hash) if weight is not None else None
                ),
                "activation_fingerprint": activation_fp,
            }

        task_results, execution_metrics = execute_schedule(
            schedule["tasks"], worker, max_workers=int(state["max_workers"])
        )
        self.last_metrics = execution_metrics
        for key, value in task_results.items():
            layer_name, row_index = key
            if isinstance(value.get("_scheduler_error"), str):
                stage_failures[key] = [
                    make_failure(
                        E_INTERNAL,
                        "row_execution",
                        f"task raised outside row guards: {value['_scheduler_error']}",
                        layer=layer_name,
                        row=row_index,
                        injection_id=injection_id,
                    )
                ]
                failures.extend(stage_failures[key])
                continue
            if value.get("softmax") is not None:
                softmax_records[key] = value["softmax"]
            if value.get("stats") is not None:
                stats_records[key] = value["stats"]
            if value.get("output") is not None:
                output_records[key] = value["output"]
            stage_failures[key] = list(value.get("failures", []))
            failures.extend(stage_failures[key])
            for stage_name, status in value.get("statuses", []):
                cache_status[stage_name].append(status)

        for layer_name, layer in loaded.layers.items():
            for row_index, _ in layer.ordered_rows():
                key = (layer_name, row_index)
                if key not in schedule_task_keys(schedule) and not any(
                    pending.get("layer") == layer_name and pending.get("row") == row_index
                    for pending in schedule["pending_queue"]
                ):
                    failures.append(
                        make_failure(
                            E_PARAM_MISSING if key not in active_weights else E_INTERNAL,
                            "load_validation",
                            "activation row was not admitted to the schedule",
                            layer=layer_name,
                            row=row_index,
                            injection_id=injection_id,
                        )
                    )

        if injection_id:
            for failure in failures:
                if failure.get("injection_id") is None:
                    failure["injection_id"] = injection_id
        failures = sorted_failures(failures)
        layers_public, completed_layers = self._build_layers_public(
            loaded, active_weights, conversion_records, softmax_records, stats_records, output_records,
            schedule, failures,
        )
        cache_plan = {
            stage: {
                "reused_rows": _row_refs(
                    (item["layer"], item["row"])
                    for item in items
                    if item["status"] == "reused"
                ),
                "recomputed_rows": _row_refs(
                    (item["layer"], item["row"])
                    for item in items
                    if item["status"] == "recomputed"
                ),
            }
            for stage, items in cache_status.items()
        }
        report = {
            "schema": SCHEMA_REPORT,
            "global_version": int(state.get("revision", 0)),
            "state": state,
            "config_hash": self.config.config_hash,
            "source_fingerprint": source_fp,
            "exchange_version": active_version,
            "exchange_version_fingerprint": exchange_fp,
            "operator_contract": {
                "softmax": {
                    "stable_max_subtraction": True,
                    "accumulation_order": "column_ascending",
                    "exp_approximation": approximation,
                    "polynomial_order": polynomial_order,
                },
                "layernorm": {
                    "accumulation_order": "column_ascending",
                    "variance": "population",
                    "variance_floor": self.config.variance_floor,
                    "low_variance_rule": "replace_variance_with_configured_floor",
                    "v1_bias_source": "integer_bias_code/1000",
                    "v2_bias_policy": dict(self.config.scheme_bias_policy),
                },
            },
            "layers": layers_public,
            "completed_layers": completed_layers,
            "failures": failures,
            "cache_plan": cache_plan,
            "schedule": {
                "layer_row_order": schedule["tasks"],
                "budget_ledger": schedule["budget_ledger"],
                "pending_queue": schedule["pending_queue"],
                "initial_budget": schedule["initial_budget"],
                "consumed_budget": schedule["consumed_budget"],
                "remaining_budget": schedule["remaining_budget"],
                "per_layer_element_quota": schedule["per_layer_element_quota"],
            },
            "execution_metrics": execution_metrics,
        }

        summary_report = copy.deepcopy(report)
        summary_report.pop("execution_metrics", None)
        summary_report.pop("snapshot", None)
        stage_payloads = {
            "load_validation": {
                "schema": "opexlab.stage.load_validation.v1",
                "source_fingerprint": source_fp,
                "layers": [_layer_public(layer) for _, layer in sorted(loaded.layers.items())],
                "weights_v1": _weight_public(loaded.weights_v1),
                "weights_v2": _weight_public(loaded.weights_v2),
                "active_weights": _weight_public(active_weights),
                "failures": [item for item in failures if item["stage"] == "load_validation"],
            },
            "version_conversion": {
                "schema": "opexlab.stage.version_conversion.v1",
                "rows": [conversion_records[key] for key in sorted(conversion_records)],
                "failures": [item for item in failures if item["stage"] == "version_conversion"],
            },
            "softmax": {
                "schema": "opexlab.stage.softmax.v1",
                "rows": [softmax_records[key] for key in sorted(softmax_records)],
                "failures": [item for item in failures if item["stage"] == "softmax"],
            },
            "layernorm": {
                "schema": "opexlab.stage.layernorm.v1",
                "stats_rows": [stats_records[key] for key in sorted(stats_records)],
                "output_rows": [output_records[key] for key in sorted(output_records)],
                "failures": [item for item in failures if item["stage"] == "layernorm"],
            },
            "summary": {
                "schema": "opexlab.stage.summary.v1",
                "report": summary_report,
            },
        }
        if commit_snapshot:
            manifest = self.store.commit(
                config_hash=self.config.config_hash,
                state_hash=state_hash(state),
                state=state,
                exchange_version=active_version,
                exchange_version_fingerprint=exchange_fp,
                completed_layers=completed_layers,
                consumed_budget=int(schedule["consumed_budget"]),
                stage_payloads=stage_payloads,
            )
            report["snapshot"] = {
                "global_version": manifest.get("global_version"),
                "snapshot_sequence": manifest.get("snapshot_sequence"),
                "snapshot_id": manifest.get("snapshot_id"),
                "snapshot_path": manifest.get("snapshot_path"),
                "stage_versions": manifest.get("stage_versions", {}),
                "commit_decision": manifest.get("commit_decision"),
            }
        else:
            report["snapshot"] = None
        return report

    def _derive_previous_plan_key_sets(
        self,
        state: Dict[str, Any],
        bundle: SourceBundle,
    ) -> Optional[Dict[str, set[str]]]:
        """Derive key-stability status from the immediately previous state.

        Incremental runs obtain this information from the previous snapshot.
        A one-shot full rebuild has no output cache by definition, but it can
        still compare final keys with the preceding logical state recorded in
        the editing history.  This makes the invalidation/reuse manifest
        identical while the full rebuild still computes every product.
        """
        history = state.get("history", [])
        if not isinstance(history, list) or not history:
            return None
        try:
            from .config import default_state

            previous_state = default_state(self.config)
            for command in history[:-1]:
                previous_state = apply_command(self.config, previous_state, command)
            temp_path = Path(
                tempfile.mkdtemp(prefix="plan-prev-", dir=str(self.store.temp_dir))
            )
            try:
                previous_engine = ExecutionEngine(self.config, runtime_dir=temp_path)
                previous_report = previous_engine.run(
                    previous_state,
                    mode="full",
                    bundle=bundle,
                    commit_snapshot=False,
                    _derive_cache_plan=False,
                )
            finally:
                shutil.rmtree(temp_path, ignore_errors=True)
        except Exception:
            return None
        sets: Dict[str, set[str]] = {
            "version_conversion": set(),
            "softmax": set(),
            "layernorm_stats": set(),
            "layernorm_output": set(),
        }
        for layer in previous_report.get("layers", []):
            for row in layer.get("rows", []):
                mapping = {
                    "version_conversion": row.get("conversion"),
                    "softmax": row.get("softmax"),
                    "layernorm_stats": row.get("layernorm_stats"),
                    "layernorm_output": row.get("layernorm_output"),
                }
                for stage, record in mapping.items():
                    if isinstance(record, dict) and isinstance(record.get("cache_key"), str):
                        sets[stage].add(record["cache_key"])
        return sets

    def _build_layers_public(
        self,
        loaded: LoadedSources,
        active_weights: Dict[Tuple[str, int], WeightRow],
        conversion_records: Dict[Tuple[str, int], Dict[str, Any]],
        softmax_records: Dict[Tuple[str, int], Dict[str, Any]],
        stats_records: Dict[Tuple[str, int], Dict[str, Any]],
        output_records: Dict[Tuple[str, int], Dict[str, Any]],
        schedule: Dict[str, Any],
        failures: list[Dict[str, Any]],
    ) -> tuple[list[Dict[str, Any]], list[str]]:
        pending_keys = {(item["layer"], item["row"]) for item in schedule["pending_queue"]}
        failed_layers = {item["layer"] for item in failures}
        layers_public: list[Dict[str, Any]] = []
        completed: list[str] = []
        for layer_name in sorted(loaded.layers):
            layer = loaded.layers[layer_name]
            rows_public = []
            layer_complete = True
            for row_index, values in layer.ordered_rows():
                key = (layer_name, row_index)
                weight = active_weights.get(key)
                conversion = conversion_records.get(key)
                softmax = softmax_records.get(key)
                stats = stats_records.get(key)
                output = output_records.get(key)
                if key in pending_keys or softmax is None or output is None:
                    layer_complete = False
                rows_public.append(
                    {
                        "row": row_index,
                        "activation_fingerprint": activation_row_fingerprint(
                            layer_name, row_index, values
                        ),
                        "parameter_fingerprint": (
                            parameter_set_fingerprint(
                                weight,
                                conversion.get("sidecar_hash") if conversion else None,
                            )
                            if weight is not None
                            else None
                        ),
                        "conversion": conversion,
                        "softmax": softmax,
                        "layernorm_stats": stats,
                        "layernorm_output": output,
                    }
                )
            if layer_complete and layer_name not in failed_layers:
                completed.append(layer_name)
            layers_public.append(
                {
                    "layer": layer_name,
                    "dim": layer.dim,
                    "source_row_count": len(layer.rows),
                    "rows": rows_public,
                }
            )
        return layers_public, completed

    def _apply_parameters(
        self,
        loaded: LoadedSources,
        source: Dict[Tuple[str, int], WeightRow],
        overrides: Dict[str, Any],
        active_version: str,
    ) -> tuple[Dict[Tuple[str, int], WeightRow], list[Dict[str, Any]]]:
        result = {key: row.clone() for key, row in source.items()}
        failures: list[Dict[str, Any]] = []
        for layer_name in sorted(overrides):
            override = overrides[layer_name]
            if not isinstance(override, dict):
                failures.append(
                    make_failure(
                        E_SHAPE,
                        "load_validation",
                        "parameter override must be an object",
                        layer=layer_name,
                    )
                )
                continue
            rows = sorted(row for (name, row) in source if name == layer_name)
            if not rows and layer_name in loaded.layers:
                rows = sorted(loaded.layers[layer_name].rows)
            if not rows:
                failures.append(
                    make_failure(
                        E_SHAPE,
                        "load_validation",
                        "parameter override layer has no rows",
                        layer=layer_name,
                    )
                )
                continue
            scale = override.get("scale")
            bias = override.get("bias", 0.0)
            if isinstance(scale, list):
                if len(scale) != len(rows):
                    failures.append(
                        make_failure(
                            E_SHAPE,
                            "load_validation",
                            "parameter override scale list length mismatch",
                            layer=layer_name,
                            details={"expected": len(rows), "actual": len(scale)},
                        )
                    )
                    continue
                scales = [float(item) for item in scale]
            else:
                scales = [float(scale)] * len(rows)
            if isinstance(bias, list):
                if len(bias) != len(rows):
                    failures.append(
                        make_failure(
                            E_SHAPE,
                            "load_validation",
                            "parameter override bias list length mismatch",
                            layer=layer_name,
                            details={"expected": len(rows), "actual": len(bias)},
                        )
                    )
                    continue
                biases = [float(item) for item in bias]
            else:
                biases = [float(bias)] * len(rows)
            scheme = override.get("scheme", self.config.v1_upgrade_scheme)
            precision = override.get("precision", next(iter(self.config.quant_codebook)))
            quant_bits = self.config.quant_codebook[precision]
            for row_index, scale_value, bias_value in zip(rows, scales, biases):
                base = result.get((layer_name, row_index))
                if active_version == "v1":
                    scale_permille = round_half_away(scale_value * 1000.0)
                    bias_code = round_half_away(bias_value * 1000.0)
                    sidecar = {
                        "origin": "parameter_override",
                        "requested_scheme": scheme,
                        "requested_precision": precision,
                        "scale_permille": scale_permille,
                        "bias_code": bias_code,
                    }
                    sidecar["sidecar_hash"] = sha256_obj(sidecar)
                    row = WeightRow(
                        layer=layer_name,
                        row=row_index,
                        scale=scale_permille / 1000.0,
                        bias=bias_code / 1000.0,
                        scheme=self.config.v1_upgrade_scheme,
                        precision=precision,
                        quant_bits=quant_bits,
                        source_version="v1",
                        raw={
                            "scale_permille": scale_permille,
                            "bias_code": bias_code,
                            "quant_bits": quant_bits,
                            "override": True,
                        },
                        sidecar=sidecar,
                        source_line=base.source_line if base else 0,
                    )
                else:
                    sidecar = {
                        "origin": "parameter_override",
                        "requested_scheme": scheme,
                        "requested_precision": precision,
                    }
                    sidecar["sidecar_hash"] = sha256_obj(sidecar)
                    row = WeightRow(
                        layer=layer_name,
                        row=row_index,
                        scale=scale_value,
                        bias=bias_value,
                        scheme=scheme,
                        precision=precision,
                        quant_bits=quant_bits,
                        source_version="v2",
                        raw={"precision_tag": "logical_override"},
                        sidecar=sidecar,
                        source_line=base.source_line if base else 0,
                    )
                result[(layer_name, row_index)] = row
        return result, failures

    def _exchange_fingerprint(
        self,
        weights: Dict[Tuple[str, int], WeightRow],
        active_version: str,
        state: Dict[str, Any],
    ) -> str:
        return sha256_obj(
            {
                "exchange_version": active_version,
                "fractional_bits": state["fractional_bits"],
                "quant_level": state["quant_level"],
                "weights": _weight_public(weights),
            }
        )


def schedule_task_keys(schedule: Dict[str, Any]) -> set[Tuple[str, int]]:
    return {(item["layer"], item["row"]) for item in schedule["tasks"]}
