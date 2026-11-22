from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generate_inputs import build_config, build_plan, build_rows  # noqa: E402
from groupcalib.pipeline import PipelineArgs, run_pipeline  # noqa: E402
from groupcalib.scheduler import BudgetTask, schedule_budget  # noqa: E402


CANONICAL_FILES = (
    "calibration.jsonl",
    "quantized.jsonl",
    "attribution.jsonl",
    "sensitivity.jsonl",
    "groups.jsonl",
    "threshold_exceedances.jsonl",
    "effective_table.jsonl",
    "effective_groups.json",
    "failures.jsonl",
    "reuse_recompute.jsonl",
    "canonical_reuse_plan.jsonl",
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    text = "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) for row in rows)
    path.write_text(text + "\n", encoding="utf-8")


def _empty_plan() -> dict:
    return {"schema_version": 1, "batches": []}


def _write_case(
    root: Path,
    *,
    config: dict | None = None,
    rows: list[dict] | None = None,
    plan: dict | None = None,
) -> dict[str, str]:
    inputs = root / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    config_path = inputs / "config.json"
    table_path = inputs / "features.jsonl"
    plan_path = inputs / "plan.json"
    _write_json(config_path, copy.deepcopy(config if config is not None else build_config()))
    _write_jsonl(table_path, copy.deepcopy(rows if rows is not None else build_rows()))
    _write_json(plan_path, copy.deepcopy(plan if plan is not None else build_plan()))
    return {"config": str(config_path), "table": str(table_path), "plan": str(plan_path)}


def _run(
    root: Path,
    paths: dict[str, str],
    *,
    out: str = "out",
    through_batch: str | None = None,
    full_rebuild: bool = False,
):
    return run_pipeline(
        PipelineArgs(
            config=paths["config"],
            table=paths["table"],
            plan=paths["plan"],
            out=str(root / out),
            through_batch=through_batch,
            full_rebuild=full_rebuild,
        )
    )


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _assert_canonical_equal(left: Path, right: Path, names=CANONICAL_FILES) -> None:
    for name in names:
        assert (left / name).read_bytes() == (right / name).read_bytes(), name


def test_full_pipeline_contract_and_determinism(tmp_path: Path) -> None:
    paths = _write_case(tmp_path)
    first = _run(tmp_path, paths, out="out_a")
    assert first.exit_code == 0
    assert first.report["status"] == "ok"

    manifest = _read_json(tmp_path / "out_a" / "manifest.json")
    assert manifest["global_version"] == 1
    assert manifest["config_hash"]
    assert manifest["group_partition_hash"]
    assert manifest["completed_groups"] == ["g_numeric", "g_ratio", "g_sensor"]
    assert manifest["active_bit_width"] == 8
    assert set(manifest["stage_versions"]) == {
        "quantile_calibration",
        "fixed_rescale",
        "error_attribution",
        "sensitivity",
    }
    assert manifest["consumed_budget"] > 0

    calibration = _read_jsonl(tmp_path / "out_a" / "calibration.jsonl")
    quantized = _read_jsonl(tmp_path / "out_a" / "quantized.jsonl")
    attribution = _read_jsonl(tmp_path / "out_a" / "attribution.jsonl")
    sensitivity = _read_jsonl(tmp_path / "out_a" / "sensitivity.jsonl")
    assert len(calibration) == 24
    assert len(sensitivity) == 9
    assert any(row["active_bit_width"] for row in calibration)
    assert any(row["clipped"] for row in quantized)
    assert any(row["truncation_count"] > 0 for row in sensitivity)
    assert {row["dominant_error_source"] for row in attribution if row.get("is_outlier")} >= {
        "saturation_truncation",
        "quantile_discretization",
        "scale_quantization",
        "dequant_rounding",
    }
    zero_rows = [row for row in attribution if row.get("zero_value_branch")]
    assert zero_rows
    assert all(row["relative_error"] is None for row in zero_rows)
    assert all(row["zero_relative_branch"] == "absolute_only" for row in zero_rows)

    # Different groups have independently derived fixed-point parameters.
    active = [row for row in calibration if row["active_bit_width"]]
    signatures = {
        group: {
            (row["column"], row["scale_factor"], row["zero_point"])
            for row in active
            if row["group_id"] == group
        }
        for group in ("g_numeric", "g_sensor", "g_ratio")
    }
    assert all(signatures.values())
    assert len({frozenset(value) for value in signatures.values()}) == 3

    second = _run(tmp_path, paths, out="out_b", full_rebuild=True)
    assert second.exit_code == 0
    _assert_canonical_equal(tmp_path / "out_a", tmp_path / "out_b", CANONICAL_FILES + ("manifest.json",))


def test_incremental_run_matches_full_and_reuses_unaffected_group(tmp_path: Path) -> None:
    config = build_config()
    rows = build_rows()
    plan_first = {
        "schema_version": 1,
        "batches": [
            {
                "batch_id": "b_num",
                "actions": [{"op": "update", "row_id": "r0003", "values": {"score": 7.75}}],
            }
        ],
    }
    plan_next = {
        "schema_version": 1,
        "batches": [
            {
                "batch_id": "b_ratio",
                "actions": [{"op": "update", "row_id": "r0008", "values": {"load": 61.0}}],
            }
        ],
    }
    plan_combined = {
        "schema_version": 1,
        "batches": [plan_first["batches"][0], plan_next["batches"][0]],
    }

    incremental_root = tmp_path / "incremental"
    ref_root = tmp_path / "reference"
    inc_paths = _write_case(incremental_root, config=config, rows=rows, plan=plan_first)
    first = _run(incremental_root, inc_paths, out="out")
    assert first.exit_code == 0

    next_plan_path = incremental_root / "inputs" / "plan_next.json"
    _write_json(next_plan_path, plan_next)
    next_paths = {
        "config": inc_paths["config"],
        "table": str(incremental_root / "out" / "effective_table.jsonl"),
        "plan": str(next_plan_path),
    }
    second = _run(incremental_root, next_paths, out="out")
    assert second.exit_code == 0
    reuse = _read_jsonl(incremental_root / "out" / "reuse_recompute.jsonl")
    reused_groups = {row["group_id"] for row in reuse if row["status"] == "reused"}
    recomputed_groups = {row["group_id"] for row in reuse if row["status"] == "recomputed"}
    assert {"g_numeric", "g_sensor"} <= reused_groups
    assert recomputed_groups == {"g_ratio"}

    ref_paths = _write_case(ref_root, config=config, rows=rows, plan=plan_combined)
    reference = _run(ref_root, ref_paths, out="out", full_rebuild=True)
    assert reference.exit_code == 0
    # The two runs use different plan files, so operational batch IDs in the
    # canonical reuse plan intentionally differ; all data products must match.
    _assert_canonical_equal(
        incremental_root / "out",
        ref_root / "out",
        tuple(
            name
            for name in CANONICAL_FILES
            if name not in {"reuse_recompute.jsonl", "canonical_reuse_plan.jsonl"}
        ),
    )


def test_same_plan_partial_then_full_matches_full_rebuild_including_reuse_plan(tmp_path: Path) -> None:
    paths = _write_case(tmp_path)
    reference = _run(tmp_path, paths, out="reference", full_rebuild=True)
    assert reference.exit_code == 0
    partial = _run(tmp_path, paths, out="partial", through_batch="b02_row_regroup")
    assert partial.exit_code == 0
    completed = _run(tmp_path, paths, out="partial")
    assert completed.exit_code == 0
    _assert_canonical_equal(tmp_path / "reference", tmp_path / "partial")


def test_validation_codes_and_failed_rows_are_excluded(tmp_path: Path) -> None:
    invalid_config = build_config()
    invalid_config["quantile_permille"] = {"lower": 960, "upper": 950}
    invalid_config["column_axis"] = 4
    config_paths = _write_case(tmp_path / "bad_config", config=invalid_config, plan=_empty_plan())
    result = _run(tmp_path / "bad_config", config_paths, out="out")
    assert result.exit_code == 2
    failures = _read_jsonl(tmp_path / "bad_config" / "out" / "failures.jsonl")
    assert failures
    assert all(row["code"] == "E_CONFIG_INVALID" for row in failures)
    assert not (tmp_path / "bad_config" / "out" / "calibration.jsonl").exists()

    rows = build_rows()
    duplicate = dict(rows[0])
    unknown_group = dict(rows[1])
    unknown_group["row_id"] = "rbad_group"
    unknown_group["group_id"] = "g_not_declared"
    invalid_value = dict(rows[2])
    invalid_value["row_id"] = "rbad_value"
    invalid_value["age"] = "not-a-number"
    rows.extend([duplicate, unknown_group, invalid_value])
    data_paths = _write_case(tmp_path / "bad_data", rows=rows, plan=_empty_plan())
    result = _run(tmp_path / "bad_data", data_paths, out="out")
    assert result.exit_code == 3
    failures = _read_jsonl(tmp_path / "bad_data" / "out" / "failures.jsonl")
    codes = Counter(row["code"] for row in failures)
    assert codes["E_DATA_INVALID"] >= 2
    assert codes["E_GROUP_UNKNOWN"] >= 1
    effective = _read_jsonl(tmp_path / "bad_data" / "out" / "effective_table.jsonl")
    effective_ids = {row["row_id"] for row in effective}
    assert "rbad_group" not in effective_ids
    assert "rbad_value" not in effective_ids
    assert _read_jsonl(tmp_path / "bad_data" / "out" / "calibration.jsonl")


def test_budget_scheduler_pending_and_pipeline_ledger(tmp_path: Path) -> None:
    tasks = [
        BudgetTask("g1", "a", 4, 4),
        BudgetTask("g1", "a", 6, 4),
        BudgetTask("g1", "b", 4, 4),
    ]
    result = schedule_budget(tasks, budget_per_cycle=5)
    assert result.complete
    assert result.consumed == 12
    assert len(result.cycles) == 3
    assert result.cycles[0]["event_code"] == "E_CALIB_BUDGET_PENDING"
    assert result.cycles[0]["pending"] == [
        {"task_key": "g1|a|006", "group_id": "g1", "column": "a", "bit_width": 6, "remaining": 3},
        {"task_key": "g1|b|004", "group_id": "g1", "column": "b", "bit_width": 4, "remaining": 4},
    ]
    # A head task is split rather than skipped, and the pending head runs first.
    assert result.cycles[1]["deductions"][0]["task_key"] == "g1|a|006"
    assert result.cycles[1]["deductions"][0]["consumed"] == 3

    partial = schedule_budget(tasks, budget_per_cycle=3, max_cycles=1)
    assert not partial.complete
    assert partial.pending
    assert partial.consumed == 3

    paths = _write_case(tmp_path)
    run = _run(tmp_path, paths, out="out")
    assert run.exit_code == 0
    ledger = _read_jsonl(tmp_path / "out" / "batch_ledger.jsonl")
    pending_cycles = [cycle for cycle in ledger if cycle.get("event_code") == "E_CALIB_BUDGET_PENDING"]
    assert pending_cycles
    assert all(cycle["pending"] for cycle in pending_cycles)
    first_deductions = ledger[0]["deductions"]
    task_order = [(item["group_id"], item["column"], item["bit_width"]) for item in first_deductions]
    assert task_order == sorted(task_order)
    business_events = _read_jsonl(tmp_path / "out" / "business_events.jsonl")
    assert any(event["code"] == "E_CALIB_BUDGET_PENDING" for event in business_events)


def test_snapshot_stale_is_reported_and_rebuilt(tmp_path: Path) -> None:
    paths = _write_case(tmp_path / "base")
    reference = _run(tmp_path / "base", paths, out="out")
    assert reference.exit_code == 0

    copied = tmp_path / "stale_config"
    shutil.copytree(tmp_path / "base" / "out", copied / "out")
    copied_paths = _write_case(copied)
    manifest_path = copied / "out" / "manifest.json"
    manifest = _read_json(manifest_path)
    manifest["config_hash"] = "deliberately-stale"
    _write_json(manifest_path, manifest)
    rerun = _run(copied, copied_paths, out="out")
    assert rerun.exit_code == 0
    events = _read_jsonl(copied / "out" / "business_events.jsonl")
    assert any(event["code"] == "E_SNAPSHOT_STALE" for event in events)
    _assert_canonical_equal(tmp_path / "base" / "out", copied / "out")

    stage_copy = tmp_path / "stale_stage"
    shutil.copytree(tmp_path / "base" / "out", stage_copy / "out")
    stage_paths = _write_case(stage_copy)
    stage_manifest_path = stage_copy / "out" / "manifest.json"
    stage_manifest = _read_json(stage_manifest_path)
    stage_manifest["stage_versions"]["sensitivity"] = "0" * 64
    _write_json(stage_manifest_path, stage_manifest)
    stage_rerun = _run(stage_copy, stage_paths, out="out")
    assert stage_rerun.exit_code == 0
    stage_events = _read_jsonl(stage_copy / "out" / "business_events.jsonl")
    assert any(
        event["code"] == "E_SNAPSHOT_STALE" and event.get("stage") == "sensitivity"
        for event in stage_events
    )
    _assert_canonical_equal(tmp_path / "base" / "out", stage_copy / "out")


def test_bit_width_gear_and_fractional_bits_change_fixed_point_values(tmp_path: Path) -> None:
    base_config = build_config()
    paths = _write_case(tmp_path, config=base_config, plan=_empty_plan())
    base = _run(tmp_path, paths, out="base", full_rebuild=True)
    assert base.exit_code == 0
    base_active = {
        (row["group_id"], row["row_id"], row["column"]): (
            row["scale_factor"],
            row["q_value"],
            row["dequantized"],
        )
        for row in _read_jsonl(tmp_path / "base" / "attribution.jsonl")
        if row.get("active_bit_width") and not row.get("missing")
    }

    gear_config = copy.deepcopy(base_config)
    gear_config["quantization"]["gear"] = 0
    gear_paths = _write_case(tmp_path / "gear", config=gear_config, plan=_empty_plan())
    gear = _run(tmp_path / "gear", gear_paths, out="out", full_rebuild=True)
    assert gear.exit_code == 0
    gear_active = {
        (row["group_id"], row["row_id"], row["column"]): (
            row["scale_factor"],
            row["q_value"],
            row["dequantized"],
        )
        for row in _read_jsonl(tmp_path / "gear" / "out" / "attribution.jsonl")
        if row.get("active_bit_width") and not row.get("missing")
    }
    assert gear_active.keys() == base_active.keys()
    assert any(gear_active[key] != base_active[key] for key in base_active)

    frac_config = copy.deepcopy(base_config)
    frac_config["quantization"]["fractional_bits"] = 2
    frac_paths = _write_case(tmp_path / "frac", config=frac_config, plan=_empty_plan())
    frac = _run(tmp_path / "frac", frac_paths, out="out", full_rebuild=True)
    assert frac.exit_code == 0
    frac_active = {
        (row["group_id"], row["row_id"], row["column"]): (
            row["scale_factor"],
            row["q_value"],
            row["dequantized"],
        )
        for row in _read_jsonl(tmp_path / "frac" / "out" / "attribution.jsonl")
        if row.get("active_bit_width") and not row.get("missing")
    }
    assert any(frac_active[key] != base_active[key] for key in base_active)


def test_concurrency_is_bounded_and_results_are_stable(tmp_path: Path) -> None:
    config_one = build_config()
    config_one["thread_pool_size"] = 1
    paths_one = _write_case(tmp_path / "one", config=config_one, plan=_empty_plan())
    one = _run(tmp_path / "one", paths_one, out="out", full_rebuild=True)
    assert one.exit_code == 0
    assert one.report["peak_concurrency"] == 1

    config_four = build_config()
    config_four["thread_pool_size"] = 4
    paths_four = _write_case(tmp_path / "four", config=config_four, plan=_empty_plan())
    four = _run(tmp_path / "four", paths_four, out="out", full_rebuild=True)
    assert four.exit_code == 0
    assert 1 <= four.report["peak_concurrency"] <= 4
    _assert_canonical_equal(tmp_path / "one" / "out", tmp_path / "four" / "out", CANONICAL_FILES)


def test_symmetric_scheme_derives_zero_centered_quantisation(tmp_path: Path) -> None:
    config = build_config()
    config["scheme"] = "symmetric"
    config["quantile_permille"] = {"lower": 0, "upper": 1000}
    paths = _write_case(tmp_path, config=config, plan=_empty_plan())
    result = _run(tmp_path, paths, out="out", full_rebuild=True)
    assert result.exit_code == 0
    calibration = _read_jsonl(tmp_path / "out" / "calibration.jsonl")
    assert calibration
    assert all(row["zero_point"] == 0 and not row["zero_point_clipped"] for row in calibration)
    assert all(row["qmin"] == -(1 << (row["bit_width"] - 1)) for row in calibration)
    assert all(row["qmax"] == (1 << (row["bit_width"] - 1)) - 1 for row in calibration)
    assert all(row["scale_factor"] > 0 for row in calibration)
    attribution = _read_jsonl(tmp_path / "out" / "attribution.jsonl")
    assert any(row.get("original", 0) < 0 and not row.get("missing") for row in attribution)


def test_repartition_redistributes_columns_and_sensitivity(tmp_path: Path) -> None:
    paths = _write_case(tmp_path)
    before = _run(tmp_path, paths, out="before", through_batch="b03_small_corrections")
    assert before.exit_code == 0
    after = _run(tmp_path, paths, out="after", through_batch="b04_column_repartition")
    assert after.exit_code == 0

    before_groups = _read_json(tmp_path / "before" / "effective_groups.json")
    after_groups = _read_json(tmp_path / "after" / "effective_groups.json")
    assert "score" in before_groups["g_numeric"]
    assert "score" in after_groups["g_sensor"]
    assert "latency" in before_groups["g_sensor"]
    assert "latency" in after_groups["g_ratio"]

    after_sensitivity = {
        (row["group_id"], row["bit_width"]): row
        for row in _read_jsonl(tmp_path / "after" / "sensitivity.jsonl")
    }
    for group_id, bit_width in after_sensitivity:
        assert set(after_sensitivity[(group_id, bit_width)]["columns"]) == set(after_groups[group_id])

    reuse = _read_jsonl(tmp_path / "after" / "reuse_recompute.jsonl")
    b04 = [row for row in reuse if row["edit_batch_id"] == "b04_column_repartition"]
    assert b04
    assert all(row["status"] == "recomputed" for row in b04)


def test_module_cli_success_and_failure_exit_codes(tmp_path: Path) -> None:
    paths = _write_case(tmp_path)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    command = [
        sys.executable,
        "-m",
        "groupcalib.pipeline",
        "--config",
        paths["config"],
        "--table",
        paths["table"],
        "--plan",
        paths["plan"],
        "--out",
        str(tmp_path / "cli_out"),
    ]
    completed = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / "cli_out" / "manifest.json").exists()

    failed = subprocess.run(
        command[:-1] + [str(tmp_path / "failed_out"), "--full-rebuild"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert failed.returncode == 0
    missing_table_command = command[:-1] + [str(tmp_path / "bad_out")]
    missing_table_command[missing_table_command.index("--table") + 1] = str(tmp_path / "missing.jsonl")
    broken = subprocess.run(
        missing_table_command,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert broken.returncode != 0
    assert (tmp_path / "bad_out" / "diagnostics.json").exists()
