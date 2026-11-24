"""End-to-end acceptance checks for the local sample workspace."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .commands import apply_command, load_command_stream
from .config import LabConfig, copy_state, default_state
from .conversion import conversion_artifact
from .engine import ExecutionEngine
from .injections import run_injections
from .loaders import load_all_sources, read_source_bundle
from .models import WeightRow
from .operators import softmax_row
from .fixed import FixedContext
from .util import canonical_json, write_json


def _business_projection(report: Dict[str, Any], *, include_state: bool = True) -> Dict[str, Any]:
    value = {
        "schema": report.get("schema"),
        "global_version": report.get("global_version"),
        "config_hash": report.get("config_hash"),
        "source_fingerprint": report.get("source_fingerprint"),
        "exchange_version": report.get("exchange_version"),
        "exchange_version_fingerprint": report.get("exchange_version_fingerprint"),
        "operator_contract": report.get("operator_contract"),
        "layers": report.get("layers"),
        "completed_layers": report.get("completed_layers"),
        "failures": report.get("failures"),
        "schedule": report.get("schedule"),
        "cache_plan": report.get("cache_plan"),
    }
    if include_state:
        value["state"] = report.get("state")
    return value


def _equal(left: Any, right: Any) -> bool:
    return canonical_json(left) == canonical_json(right)


def _layer_map(report: Dict[str, Any]) -> Dict[tuple[str, int], Dict[str, Any]]:
    result = {}
    for layer in report.get("layers", []):
        for row in layer.get("rows", []):
            result[(layer["layer"], row["row"])] = row
    return result


def _rows(items: Iterable[Dict[str, Any]]) -> set[tuple[str, int]]:
    return {(item["layer"], int(item["row"])) for item in items}


def _make_runtime(config: LabConfig, name: str) -> Path:
    base = config.runtime_dir / "verification"
    base.mkdir(parents=True, exist_ok=True)
    path = base / name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _apply_commands_in_memory(config: LabConfig, commands: list[Dict[str, Any]]) -> Dict[str, Any]:
    state = default_state(config)
    for command in commands:
        state = apply_command(config, state, command)
    return state


def run_verification(config: LabConfig, *, report_path: Optional[Path] = None) -> Dict[str, Any]:
    checks: list[Dict[str, Any]] = []

    def record(name: str, passed: bool, **details: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "details": details})

    commands = load_command_stream(config.paths["commands"])
    bundle = read_source_bundle(config)
    loaded = load_all_sources(config, bundle)

    # 1. Bidirectional conversion and sidecar observability.
    conversion_errors = []
    for key in sorted(loaded.weights_v1):
        original = loaded.weights_v1[key]
        upgraded = conversion_artifact(config, original, "v2")["converted_row"]
        downgraded = conversion_artifact(config, upgraded, "v1")["converted_row"]
        if (
            downgraded.raw.get("scale_permille") != original.raw.get("scale_permille")
            or downgraded.raw.get("bias_code") != original.raw.get("bias_code")
            or downgraded.raw.get("quant_bits") != original.raw.get("quant_bits")
        ):
            conversion_errors.append({"direction": "v1-v2-v1", "layer": key[0], "row": key[1]})
    for key in sorted(loaded.weights_v2):
        original = loaded.weights_v2[key]
        downgraded_artifact = conversion_artifact(config, original, "v1")
        upgraded = conversion_artifact(config, downgraded_artifact["converted_row"], "v2")[
            "converted_row"
        ]
        if (
            upgraded.scale != original.scale
            or upgraded.bias != original.bias
            or upgraded.quant_bits != original.quant_bits
        ):
            conversion_errors.append({"direction": "v2-v1-v2", "layer": key[0], "row": key[1]})
    record(
        "v1_v2_roundtrip_restorable_fields",
        not conversion_errors,
        v1_rows=len(loaded.weights_v1),
        v2_rows=len(loaded.weights_v2),
        errors=conversion_errors[:5],
    )
    lossy = WeightRow(
        layer="layer_alpha",
        row=0,
        scale=1.234567,
        bias=-0.00789,
        scheme="layernorm_scale_only",
        precision="int16",
        quant_bits=16,
        source_version="v2",
    )
    lossy_artifact = conversion_artifact(config, lossy, "v1")
    sidecar = lossy_artifact["sidecar"]
    record(
        "lossy_conversion_sidecar",
        bool(sidecar.get("sidecar_hash"))
        and sidecar.get("dropped_scheme") == "layernorm_scale_only"
        and abs(float(sidecar.get("scale_residual", 0.0))) > 0,
        sidecar=sidecar,
    )

    # 2. Fixed-point deterministic Softmax.
    context = FixedContext(
        fractional_bits=config.fractional_bits,
        quant_level=config.quant_level,
    )
    sample_row = loaded.layers["layer_alpha"].rows[0]
    sm_a = softmax_row(
        sample_row,
        context,
        polynomial_order=config.exp_polynomial_order(context.quant_level),
        approximation_name=config.exp_approximation_name(context.quant_level),
    )
    sm_b = softmax_row(
        sample_row,
        context,
        polynomial_order=config.exp_polynomial_order(context.quant_level),
        approximation_name=config.exp_approximation_name(context.quant_level),
    )
    record(
        "softmax_deterministic_and_normalized",
        _equal(sm_a, sm_b) and sum(sm_a["output_fixed"]) == context.scale,
        output_fixed=sm_a["output_fixed"],
        denominator_fixed=sm_a["denominator_fixed"],
        exp_approximation=sm_a["metadata"]["exp_approximation"],
    )

    # 3. Base v1/v2 operator-path equivalence.
    v1_runtime = _make_runtime(config, "path-v1")
    v2_runtime = _make_runtime(config, "path-v2")
    v1_state = default_state(config)
    v1_state["exchange_version"] = "v1"
    v2_state = default_state(config)
    v2_state["exchange_version"] = "v2"
    v1_report = ExecutionEngine(config, runtime_dir=v1_runtime).run(
        v1_state, mode="full", commit_snapshot=False
    )
    v2_report = ExecutionEngine(config, runtime_dir=v2_runtime).run(
        v2_state, mode="full", commit_snapshot=False
    )
    v1_map = _layer_map(v1_report)
    v2_map = _layer_map(v2_report)
    path_mismatches = []
    for key in sorted(set(v1_map) & set(v2_map)):
        v1_row = v1_map[key]
        v2_row = v2_map[key]
        if (
            v1_row["layernorm_stats"]["result"]["stats"]
            != v2_row["layernorm_stats"]["result"]["stats"]
            or v1_row["layernorm_output"]["result"]["output_fixed"]
            != v2_row["layernorm_output"]["result"]["output_fixed"]
        ):
            path_mismatches.append({"layer": key[0], "row": key[1]})
    record(
        "v1_v2_same_logical_parameters_same_fixed_result",
        not path_mismatches and not v1_report["failures"] and not v2_report["failures"],
        rows_compared=len(set(v1_map) & set(v2_map)),
        mismatches=path_mismatches[:5],
    )

    # 4. Quantization level must change real integer products.
    q0_runtime = _make_runtime(config, "quant-0")
    q2_runtime = _make_runtime(config, "quant-2")
    q0_state = default_state(config)
    q0_state["quant_level"] = 0
    q2_state = default_state(config)
    q2_state["quant_level"] = 2
    q0_report = ExecutionEngine(config, runtime_dir=q0_runtime).run(
        q0_state, mode="full", commit_snapshot=False
    )
    q2_report = ExecutionEngine(config, runtime_dir=q2_runtime).run(
        q2_state, mode="full", commit_snapshot=False
    )
    q0_output = _layer_map(q0_report)[("layer_alpha", 0)]["softmax"]["result"]["output_fixed"]
    q2_output = _layer_map(q2_report)[("layer_alpha", 0)]["softmax"]["result"]["output_fixed"]
    record(
        "quant_level_changes_softmax_integer_vector",
        q0_output != q2_output,
        quant_0=q0_output,
        quant_2=q2_output,
    )

    # 5. Repeated full runs are byte-for-byte equal on business fields.
    repeat_runtime = _make_runtime(config, "repeat")
    repeat_engine = ExecutionEngine(config, runtime_dir=repeat_runtime)
    repeat_a = repeat_engine.run(default_state(config), mode="full", commit_snapshot=False)
    repeat_b = repeat_engine.run(default_state(config), mode="full", commit_snapshot=False)
    record(
        "repeated_run_business_fields_equal",
        _equal(_business_projection(repeat_a), _business_projection(repeat_b)),
    )

    # 6. Exact incremental row invalidation.
    inc_runtime = _make_runtime(config, "incremental-row")
    inc_engine = ExecutionEngine(config, runtime_dir=inc_runtime)
    base_state = default_state(config)
    inc_engine.save_state(base_state)
    first = inc_engine.run(base_state, mode="incremental")
    row_edit = {
        "type": "replace_activation_row",
        "layer": "layer_beta",
        "row": 1,
        "values": [1.1, -0.4, 0.2, 0.9, -1.3],
    }
    edited_state = apply_command(config, base_state, row_edit)
    inc_engine.save_state(edited_state)
    second = inc_engine.run(edited_state, mode="incremental")
    row_plan = second["cache_plan"]
    row_check = (
        _rows(row_plan["version_conversion"]["reused_rows"]) == set(loaded.weights_v1)
        and _rows(row_plan["softmax"]["recomputed_rows"]) == {("layer_beta", 1)}
        and _rows(row_plan["layernorm_stats"]["recomputed_rows"]) == {("layer_beta", 1)}
        and _rows(row_plan["layernorm_output"]["recomputed_rows"]) == {("layer_beta", 1)}
    )
    record(
        "single_activation_row_invalidation_exact",
        row_check,
        softmax_recomputed=sorted(_rows(row_plan["softmax"]["recomputed_rows"])),
        stats_recomputed=sorted(_rows(row_plan["layernorm_stats"]["recomputed_rows"])),
        output_recomputed=sorted(_rows(row_plan["layernorm_output"]["recomputed_rows"])),
    )

    # 7. Layer-parameter edit invalidates conversion and output only on that layer.
    param_runtime = _make_runtime(config, "incremental-param")
    param_engine = ExecutionEngine(config, runtime_dir=param_runtime)
    param_base = default_state(config)
    param_engine.save_state(param_base)
    param_engine.run(param_base, mode="incremental")
    param_state = apply_command(
        config,
        param_base,
        {
            "type": "replace_layer_params",
            "layer": "layer_beta",
            "scale": 1.35,
            "bias": 0.02,
            "scheme": "layernorm_affine",
            "precision": "int16",
        },
    )
    param_engine.save_state(param_state)
    param_report = param_engine.run(param_state, mode="incremental")
    param_plan = param_report["cache_plan"]
    layer_beta_rows = {("layer_beta", 0), ("layer_beta", 1), ("layer_beta", 2)}
    param_check = (
        _rows(param_plan["version_conversion"]["recomputed_rows"]) == layer_beta_rows
        and not param_plan["softmax"]["recomputed_rows"]
        and not param_plan["layernorm_stats"]["recomputed_rows"]
        and _rows(param_plan["layernorm_output"]["recomputed_rows"]) == layer_beta_rows
    )
    record(
        "single_layer_parameter_invalidation_exact",
        param_check,
        conversion_recomputed=sorted(_rows(param_plan["version_conversion"]["recomputed_rows"])),
        softmax_recomputed=sorted(_rows(param_plan["softmax"]["recomputed_rows"])),
        stats_recomputed=sorted(_rows(param_plan["layernorm_stats"]["recomputed_rows"])),
        output_recomputed=sorted(_rows(param_plan["layernorm_output"]["recomputed_rows"])),
    )

    # 8. Final incremental state versus one-shot full rebuild.
    final_runtime = _make_runtime(config, "command-sequence")
    final_engine = ExecutionEngine(config, runtime_dir=final_runtime)
    final_engine.save_state(default_state(config))
    final_engine.run(default_state(config), mode="incremental")
    for command in commands:
        final_engine.apply_and_run(command, mode="incremental")
    final_incremental = final_engine.run(final_engine.load_state(), mode="incremental")
    final_state = _apply_commands_in_memory(config, commands)
    full_runtime = _make_runtime(config, "blank-full")
    full_report = ExecutionEngine(config, runtime_dir=full_runtime).run(
        final_state, mode="full", commit_snapshot=False
    )
    record(
        "incremental_final_equals_blank_full_business_fields",
        _equal(_business_projection(final_incremental), _business_projection(full_report)),
        incremental_failures=len(final_incremental["failures"]),
        full_failures=len(full_report["failures"]),
        layers=len(final_incremental["layers"]),
    )
    record(
        "incremental_final_cache_keys_equal_blank_full",
        _equal(
            _stage_keys(final_incremental),
            _stage_keys(full_report),
        ),
        note="All stage cache keys and row products are compared.",
    )
    record(
        "incremental_final_reuse_lists_equal_blank_full",
        _equal(final_incremental["cache_plan"], full_report["cache_plan"]),
        incremental_cache_plan=final_incremental["cache_plan"],
        full_cache_plan=full_report["cache_plan"],
    )

    replay_runtime = _make_runtime(config, "blank-replay")
    replay_engine = ExecutionEngine(config, runtime_dir=replay_runtime)
    replay_engine.save_state(default_state(config))
    replay_engine.run(default_state(config), mode="incremental")
    for command in commands:
        replay_engine.apply_and_run(command, mode="incremental")
    replay_report = replay_engine.run(replay_engine.load_state(), mode="incremental")
    record(
        "blank_command_replay_cache_plan_stable",
        _equal(final_incremental["cache_plan"], replay_report["cache_plan"]),
        incremental_cache_plan=final_incremental["cache_plan"],
        replay_cache_plan=replay_report["cache_plan"],
    )

    # 9. Serial/parallel scheduling equivalence.
    serial_runtime = _make_runtime(config, "serial")
    parallel_runtime = _make_runtime(config, "parallel")
    serial_state = default_state(config)
    serial_state["max_workers"] = 1
    parallel_state = default_state(config)
    parallel_state["max_workers"] = 4
    serial_engine = ExecutionEngine(config, runtime_dir=serial_runtime)
    parallel_engine = ExecutionEngine(config, runtime_dir=parallel_runtime)
    serial_report = serial_engine.run(serial_state, mode="full", commit_snapshot=False)
    parallel_report = parallel_engine.run(parallel_state, mode="full", commit_snapshot=False)
    serial_business = _business_projection(serial_report, include_state=False)
    parallel_business = _business_projection(parallel_report, include_state=False)
    record(
        "serial_parallel_schedule_and_results_equal",
        _equal(serial_business, parallel_business)
        and parallel_engine.last_metrics["peak_concurrent_tasks"] <= 4,
        requested_max_workers=4,
        peak_concurrent_tasks=parallel_engine.last_metrics.get("peak_concurrent_tasks"),
        serial_budget=serial_report["schedule"]["budget_ledger"],
        parallel_budget=parallel_report["schedule"]["budget_ledger"],
    )

    # 10. Fault injections, each isolated and mapped to a business code.
    injection_report = run_injections(config)
    record(
        "fault_injections_hit_and_map_to_business_codes",
        injection_report["all_matched"],
        items=[
            {
                "id": item["id"],
                "expected_code": item["expected_code"],
                "matched": item["matched"],
                "hit_count": item["hit_count"],
            }
            for item in injection_report["items"]
        ],
    )

    result = {
        "schema": "opexlab.verification.v1",
        "all_passed": all(check["passed"] for check in checks),
        "checks": checks,
        "injection_report": injection_report,
        "summary": {
            "passed": sum(1 for check in checks if check["passed"]),
            "total": len(checks),
            "failed": [check["name"] for check in checks if not check["passed"]],
        },
    }
    report_path = report_path or (config.runtime_dir / "reports" / "verification.json")
    write_json(report_path, result)
    return result


def _stage_keys(report: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for layer in report.get("layers", []):
        for row in layer.get("rows", []):
            key = f"{layer['layer']}:{row['row']}"
            result[key] = {
                "activation": row.get("activation_fingerprint"),
                "parameter": row.get("parameter_fingerprint"),
                "conversion": _record_key(row.get("conversion")),
                "softmax": _record_key(row.get("softmax")),
                "stats": _record_key(row.get("layernorm_stats")),
                "output": _record_key(row.get("layernorm_output")),
            }
    return result


def _record_key(record: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if record is None:
        return None
    return {
        "cache_key": record.get("cache_key"),
        "content_hash": record.get("content_hash"),
    }
