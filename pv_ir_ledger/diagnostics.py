from __future__ import annotations

import json
import platform
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .commands import apply_command
from .config import load_config, validate_config
from .exchange import (
    downgrade_v2_to_v1,
    logical_comparison_view,
    parse_v1_text,
    parse_v2_document,
    upgrade_v1_to_v2,
    strict_internal_view,
)
from .hashing import canonical_json
from .imaging import load_and_decode_png
from .pipeline import run_pipeline
from .recognition import determine_grade
from .reporting import domain_canonical, event_order_canonical, export_judgments
from .samplegen import generate_sample_tree
from .scheduler import plan_budget, run_concurrent, stable_task_key
from .snapshot import SnapshotStore, ignore_stale_transactions
from .truth import load_truth, validate_truth


def make_check(check_id: str, title: str, ok: bool, **details: Any) -> dict[str, Any]:
    return {"id": check_id, "title": title, "ok": bool(ok), "details": details}


def first_difference(left: Any, right: Any, path: str = "$") -> dict[str, Any] | None:
    if type(left) is not type(right):
        if isinstance(left, dict) and isinstance(right, dict):
            pass
        elif isinstance(left, list) and isinstance(right, list):
            pass
        elif left == right:
            return None
        else:
            return {"path": path, "left_type": type(left).__name__, "right_type": type(right).__name__}
    if isinstance(left, dict):
        keys = sorted(set(left) | set(right))
        for key in keys:
            if key not in left or key not in right:
                return {"path": f"{path}.{key}", "reason": "missing_key"}
            diff = first_difference(left[key], right[key], f"{path}.{key}")
            if diff:
                return diff
    elif isinstance(left, list):
        if len(left) != len(right):
            return {"path": path, "reason": "length", "left": len(left), "right": len(right)}
        for index, (a, b) in enumerate(zip(left, right)):
            diff = first_difference(a, b, f"{path}[{index}]")
            if diff:
                return diff
    elif left != right:
        return {"path": path, "left": left, "right": right}
    return None


def bootstrap_samples(samples: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    config, config_issues = load_config(samples / "recognition_config.json")
    if config is None:
        raise RuntimeError(config_issues)
    truth, truth_issues = load_truth(samples / "array_truth.json", config)
    ledger, _ = run_pipeline(
        str(samples),
        config,
        truth_by_image=truth,
        truth_issues=truth_issues,
        exchange_version="v2",
    )
    export_judgments(ledger, samples)
    return config, truth, truth_issues


def verify_exchange(samples: Path) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    v2_raw = json.loads((samples / "judgments_v2.json").read_text(encoding="utf-8"), parse_float=str)
    v2, v2_issues = parse_v2_document(v2_raw)
    sidecar = json.loads((samples / "judgments_v1.sidecar.json").read_text(encoding="utf-8"))
    v1_text = (samples / "judgments_v1.txt").read_text(encoding="utf-8")
    v1, v1_issues = parse_v1_text(v1_text, sidecar)

    originals = {item["sidecar_key"]: logical_comparison_view(item) for item in v2}
    mismatches: list[str] = []
    for judgment in v1:
        restored = upgrade_v1_to_v2(judgment, sidecar)
        key = restored["sidecar_key"]
        if logical_comparison_view(restored) != originals.get(key):
            mismatches.append(key)

    strict_mismatches: list[str] = []
    for original in v2:
        downgraded, sidecar_entry = downgrade_v2_to_v1(original)
        restored = upgrade_v1_to_v2(downgraded, {"entries": [sidecar_entry]})
        if canonical_json(strict_internal_view(restored)) != canonical_json(strict_internal_view(original)):
            strict_mismatches.append(original["sidecar_key"])

    checks.append(
        make_check(
            "EXCHANGE_ROUNDTRIP",
            "v2 -> v1+sidecar -> v2 logical representation is identical",
            len(v2) == 72
            and len(v1) == 72
            and not v2_issues
            and not v1_issues
            and not mismatches
            and not strict_mismatches,
            v2_count=len(v2),
            v1_count=len(v1),
            parse_issue_count=len(v2_issues) + len(v1_issues),
            mismatches=mismatches[:10],
            strict_mismatches=strict_mismatches[:10],
            strict_mismatch_count=len(strict_mismatches),
        )
    )

    short_judgments, short_issues = parse_v1_text(
        (samples / "injections" / "INJ07_truncated_or_illegal_v1.txt").read_text(encoding="utf-8")
    )
    checks.append(
        make_check(
            "EXCHANGE_E_FIELD_TRUNCATED",
            "short and illegal v1 fields emit E_FIELD_TRUNCATED",
            len(short_issues) >= 2 and all(item["code"] == "E_FIELD_TRUNCATED" for item in short_issues),
            issue_count=len(short_issues),
            reasons=[item.get("reason") for item in short_issues],
            parsed_records=len(short_judgments),
        )
    )
    return checks


def _stage_actions(ledger: dict[str, Any], stage: str) -> list[str]:
    return [event["action"] for event in ledger["ledger_events"] if event["stage"] == stage]


def _all_action(ledger: dict[str, Any], stage: str, action: str) -> bool:
    actions = _stage_actions(ledger, stage)
    return bool(actions) and all(item == action for item in actions)


def verify_incremental(workspace: Path, samples: Path, temp_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    case_dir = temp_root / "incremental-case"
    shutil.copytree(samples, case_dir)
    config, config_issues = load_config(case_dir / "recognition_config.json")
    if config is None:
        raise RuntimeError(config_issues)
    truth, truth_issues = load_truth(case_dir / "array_truth.json", config)
    store = SnapshotStore(temp_root / "incremental-state", workspace_root=workspace)

    baseline, baseline_state = run_pipeline(
        str(case_dir), config, truth, truth_issues, previous_state=None, exchange_version="v1"
    )
    store.commit(baseline_state)
    baseline_full, _ = run_pipeline(
        str(case_dir), config, truth, truth_issues, previous_state=None, exchange_version="v1"
    )
    checks.append(
        make_check(
            "INCREMENTAL_BASELINE_DETERMINISM",
            "blank baseline domain ledger is deterministic",
            domain_canonical(baseline) == domain_canonical(baseline_full),
            component_count=len(baseline["components"]),
            failure_count=len(baseline["failures"]),
        )
    )

    command_specs = [
        ("EXCHANGE_V2", {"op": "set_exchange_version", "version": "v2"}, "exchange_only"),
        ("GRID_5X7", {"op": "set_grid", "grid_rows": 5, "grid_cols": 7}, "grid"),
        ("PERCENTILES", {"op": "set_percentiles", "percentile_low_permille": 360, "percentile_high_permille": 930}, "percentiles"),
        ("THRESHOLDS", {"op": "set_thresholds", "grade_thresholds_permille": [220, 380, 540, 700]}, "thresholds"),
        ("QUANT_Q4_L32", {"op": "set_quantization", "fixed_decimals": 4, "quant_levels": 32}, "quant"),
        ("QUANT_Q2_L16", {"op": "set_quantization", "fixed_decimals": 2, "quant_levels": 16}, "quant"),
        ("REPLACE_IMAGE_A", {"op": "replace_image", "image": "IR_ARRAY_A.png", "variant": 3}, "replace"),
        ("DELETE_IMAGE_C", {"op": "delete_image", "image": "IR_ARRAY_C.png"}, "delete"),
        ("EXCHANGE_V1", {"op": "set_exchange_version", "version": "v1"}, "exchange_back"),
    ]

    steps: list[dict[str, Any]] = []
    previous_step_ledger = baseline
    current_config = config
    version = "v1"
    all_equal = True

    for index, (label, command, expectation) in enumerate(command_specs, start=1):
        current_config, version, apply_issues, mutation = apply_command(command, current_config, version, case_dir)
        loaded_state, recovery_issues = store.load()
        incremental, next_state = run_pipeline(
            str(case_dir),
            current_config,
            truth,
            truth_issues,
            previous_state=loaded_state,
            exchange_version=version,
        )
        store.commit(next_state)
        full, _ = run_pipeline(
            str(case_dir),
            current_config,
            truth,
            truth_issues,
            previous_state=None,
            exchange_version=version,
        )
        inc_sidecar = export_judgments(incremental, temp_root / f"inc-export-{index}")["sidecar"]
        full_sidecar = export_judgments(full, temp_root / f"full-export-{index}")["sidecar"]
        domain_equal = domain_canonical(incremental) == domain_canonical(full)
        events_equal = event_order_canonical(incremental) == event_order_canonical(full)
        sidecar_equal = canonical_json(inc_sidecar) == canonical_json(full_sidecar)
        diff = first_difference(incremental, full) if not domain_equal else None
        all_equal = all_equal and domain_equal and events_equal and sidecar_equal and not apply_issues

        expectation_ok = True
        expectation_detail: dict[str, Any] = {}
        if expectation == "exchange_only":
            for stage in ("grid", "stats", "regression", "intensity", "grade", "array", "image"):
                expectation_ok &= _all_action(incremental, stage, "REUSED")
            expectation_detail["conversion_recomputed"] = _stage_actions(incremental, "conversion").count("RECOMPUTED")
            expectation_ok &= _stage_actions(incremental, "conversion").count("RECOMPUTED") == len(incremental["components"])
        elif expectation == "grid":
            expectation_detail["grid_recomputed"] = _stage_actions(incremental, "grid").count("RECOMPUTED")
            expectation_ok &= _all_action(incremental, "grid", "RECOMPUTED")
        elif expectation == "percentiles":
            expectation_ok &= _all_action(incremental, "grid", "REUSED")
            expectation_detail["grid_reused"] = len(_stage_actions(incremental, "grid"))
        elif expectation == "thresholds":
            expectation_ok &= _all_action(incremental, "stats", "REUSED")
            expectation_ok &= _all_action(incremental, "regression", "REUSED")
            expectation_ok &= _all_action(incremental, "intensity", "REUSED")
            expectation_detail["grade_recomputed"] = _stage_actions(incremental, "grade").count("RECOMPUTED")
        elif expectation == "quant":
            expectation_ok &= _all_action(incremental, "regression", "REUSED")
            current_values = {
                (c["image"], c["component_id"]): (c["quant_integer"], c["intensity_fixed"], c["intensity"]["fixed_decimals"])
                for c in incremental["components"]
            }
            old_values = {
                (c["image"], c["component_id"]): (c["quant_integer"], c["intensity_fixed"], c["intensity"]["fixed_decimals"])
                for c in previous_step_ledger["components"]
                if (c["image"], c["component_id"]) in current_values
            }
            changed = sum(1 for key, value in current_values.items() if key in old_values and (value[0], value[1]) != (old_values[key][0], old_values[key][1]))
            expectation_detail["quant_values_changed"] = changed
            expectation_detail["fixed_decimals"] = current_config["fixed_decimals"]
            expectation_ok &= changed > 0
        elif expectation == "replace":
            unchanged = [c for c in incremental["components"] if c["image"] != "IR_ARRAY_A.png"]
            expectation_ok &= bool(unchanged) and all(
                event["action"] == "REUSED"
                for event in incremental["ledger_events"]
                if event["stage"] == "regression" and event["image"] != "IR_ARRAY_A.png"
            )
            expectation_ok &= any(
                event["action"] == "RECOMPUTED"
                for event in incremental["ledger_events"]
                if event["stage"] == "regression" and event["image"] == "IR_ARRAY_A.png"
            )
            expectation_detail["unchanged_regression_reused"] = len(unchanged)
        elif expectation == "delete":
            expectation_ok &= bool(incremental["components"]) and all(
                event["action"] == "REUSED" for event in incremental["ledger_events"]
            )
            expectation_detail["remaining_components"] = len(incremental["components"])
        elif expectation == "exchange_back":
            for stage in ("grid", "stats", "regression", "intensity", "grade"):
                expectation_ok &= _all_action(incremental, stage, "REUSED")

        step_ok = domain_equal and events_equal and sidecar_equal and expectation_ok and not apply_issues
        checks.append(
            make_check(
                f"COMMAND_{index:02d}_{label}",
                f"incremental and blank full rebuild match after {label}",
                step_ok,
                domain_equal=domain_equal,
                event_order_equal=events_equal,
                sidecar_equal=sidecar_equal,
                cache_expectation_ok=expectation_ok,
                apply_issue_count=len(apply_issues),
                component_count=len(incremental["components"]),
                failure_count=len(incremental["failures"]),
                pending_count=len(incremental["pending"]),
                first_difference=diff,
                **expectation_detail,
            )
        )
        steps.append(
            {
                "label": label,
                "command": command,
                "domain_equal": domain_equal,
                "event_order_equal": events_equal,
                "sidecar_equal": sidecar_equal,
                "cache_expectation_ok": expectation_ok,
                "components": len(incremental["components"]),
                "failures": len(incremental["failures"]),
            }
        )
        previous_step_ledger = incremental

    checks.append(
        make_check(
            "INCREMENTAL_SEQUENCE_ALL_MATCH",
            "every command step matches a blank full rebuild field-for-field",
            all_equal,
            steps=steps,
        )
    )
    return checks, {"steps": steps, "final_config_hash": current_config["config_hash"]}


def verify_budget_and_snapshot(workspace: Path, samples: Path, temp_root: Path) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    case_dir = temp_root / "budget-case"
    shutil.copytree(samples, case_dir)
    config, _ = load_config(case_dir / "recognition_config.json")
    truth, _ = load_truth(case_dir / "array_truth.json", config)

    full, full_state = run_pipeline(str(case_dir), config, truth, previous_state=None, budget=10**12)
    cost_rows = sorted(
        (c["image"], c["component_id"], int(c["stats"]["pixel_count"]) * int(config["heat_bin_count"]))
        for c in full["components"]
    )
    task_dicts = [
        {"scheduler_key": stable_task_key(image, component_id, "heat_sampling:v1"), "cost": cost}
        for image, component_id, cost in cost_rows
    ]
    prefix_ok = True
    for k in (0, 1, 5, 24, len(task_dicts)):
        budget = sum(row[2] for row in cost_rows[:k])
        scheduled, pending, deductions = plan_budget(task_dicts, budget)
        prefix_ok &= [item["scheduler_key"] for item in scheduled] == [item["scheduler_key"] for item in task_dicts[:k]]
        prefix_ok &= len(pending) == len(task_dicts) - k
        prefix_ok &= [item["scheduler_key"] for item in deductions] == [item["scheduler_key"] for item in task_dicts[:k]]
    checks.append(
        make_check(
            "BUDGET_STABLE_PREFIX",
            "different exhaustion points consume the same stable sorted prefix",
            prefix_ok,
            task_count=len(task_dicts),
            first_costs=[row[2] for row in cost_rows[:5]],
        )
    )

    first_image_count = len([row for row in cost_rows if row[0] == "IR_ARRAY_A.png"])
    first_image_budget = sum(row[2] for row in cost_rows if row[0] == "IR_ARRAY_A.png")
    store = SnapshotStore(temp_root / "budget-state", workspace_root=workspace)
    partial, partial_state = run_pipeline(
        str(case_dir), config, truth, previous_state=None, budget=first_image_budget
    )
    store.commit(partial_state)
    loaded_partial, recovery_issues = store.load()
    partial_pending_ok = (
        len(partial["components"]) == first_image_count
        and len(partial["pending"]) == len(cost_rows) - first_image_count
        and any(issue["code"] == "E_SAMPLE_BUDGET_PENDING" for issue in partial["failures"])
        and loaded_partial is not None
        and loaded_partial["manifest"]["completed_images"] == ["IR_ARRAY_A.png"]
        and not recovery_issues
    )
    checks.append(
        make_check(
            "BUDGET_PENDING_SNAPSHOT",
            "budget shortfall creates ordered pending queue and committed manifest",
            partial_pending_ok,
            processed=len(partial["components"]),
            pending=len(partial["pending"]),
            manifest=loaded_partial["manifest"] if loaded_partial else None,
        )
    )

    resumed, resumed_state = run_pipeline(
        str(case_dir), config, truth, previous_state=loaded_partial, budget=10**12
    )
    store.commit(resumed_state)
    domain_equal = domain_canonical(resumed) == domain_canonical(full)
    deductions_equal = canonical_json(resumed_state["budget"]["deductions"]) == canonical_json(full_state["budget"]["deductions"])
    checks.append(
        make_check(
            "SNAPSHOT_RECOVERY_FULL_EQUAL",
            "resume from pending snapshot yields full ledger and identical deduction sequence",
            domain_equal and deductions_equal,
            domain_equal=domain_equal,
            deductions_equal=deductions_equal,
            deduction_count=len(resumed_state["budget"]["deductions"]),
            first_difference=first_difference(resumed, full) if not domain_equal else None,
        )
    )

    same_hash = loaded_partial["manifest"]["config_hash"] == config["config_hash"]
    changed_config, changed_issues = validate_config({**config, "grid_rows": 7})
    changed_detected = bool(changed_config) and loaded_partial["manifest"]["config_hash"] != changed_config["config_hash"]
    stale_dir = store.temp_root / "snapshot-v-stale-interrupted"
    stale_dir.mkdir(parents=True, exist_ok=True)
    ignored = ignore_stale_transactions(store.temp_root)
    checks.append(
        make_check(
            "SNAPSHOT_VERSION_CONFIG_DETECTION",
            "startup recognizes schema version and configuration hash changes; stale temp transactions are ignored",
            same_hash
            and changed_detected
            and loaded_partial["manifest"]["state_schema_version"] == 1
            and any("stale-interrupted" in path for path in ignored),
            state_schema_version=loaded_partial["manifest"]["state_schema_version"],
            global_version_after_resume=resumed_state["manifest"]["global_version"],
            same_config_hash=same_hash,
            changed_config_detected=changed_detected,
            ignored_temp_dirs=ignored,
            changed_config_issue_count=len(changed_issues),
        )
    )

    def dummy_worker(task: dict[str, Any]) -> str:
        import time

        time.sleep(0.004)
        return task["scheduler_key"]

    dummy_tasks = [
        {"scheduler_key": stable_task_key("IMG.png", i, "stage"), "cost": 1}
        for i in range(12)
    ]
    _, concurrency = run_concurrent(dummy_tasks, dummy_worker, int(config["max_concurrency"]))
    checks.append(
        make_check(
            "CONCURRENCY_LIMIT",
            "thread pool never exceeds configured maximum and results stay sorted",
            concurrency["max_observed_concurrency"] <= concurrency["max_configured_concurrency"] <= config["max_concurrency"],
            **concurrency,
        )
    )
    invalid_concurrency, invalid_issues = validate_config({**config, "max_concurrency": 99})
    checks.append(
        make_check(
            "CONCURRENCY_UPPER_BOUND_VALIDATION",
            "concurrency above hard upper limit is rejected",
            invalid_concurrency is None and any(i["code"] == "E_CONFIG_INVALID" for i in invalid_issues),
            issue_count=len(invalid_issues),
            hard_upper_limit=8,
        )
    )
    return checks


def verify_injections(samples: Path, temp_root: Path) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    injection_dir = samples / "injections"

    inj01 = temp_root / "inj01"
    inj01.mkdir(parents=True, exist_ok=True)
    shutil.copy(injection_dir / "INJ01_truncated.png", inj01 / "INJ01_truncated.png")
    shutil.copy(samples / "IR_ARRAY_B.png", inj01 / "IR_NEIGHBOR_GOOD.png")
    config01, _ = load_config(injection_dir / "INJ01_image_config.json")
    ledger01, _ = run_pipeline(str(inj01), config01)
    neighbor_components = [c for c in ledger01["components"] if c["image"] == "IR_NEIGHBOR_GOOD.png"]
    checks.append(
        make_check(
            "INJ01_TRUNCATED_PNG_ISOLATED",
            "truncated PNG emits E_IMAGE_FORMAT while neighbor image completes",
            any(i["code"] == "E_IMAGE_FORMAT" for i in ledger01["failures"])
            and len(neighbor_components) == 24
            and all(c["grade_code"] != 9 for c in neighbor_components),
            failure_codes=[i["code"] for i in ledger01["failures"]],
            neighbor_components=len(neighbor_components),
        )
    )

    bad_map_config, bad_map_issues = load_config(injection_dir / "INJ02_bad_colormap_config.json")
    checks.append(
        make_check(
            "INJ02A_COLORMAP_CONFIG",
            "out-of-table pseudo-color mapping config emits E_CONFIG_INVALID",
            bad_map_config is None and any(i["code"] == "E_CONFIG_INVALID" for i in bad_map_issues),
            issue_count=len(bad_map_issues),
        )
    )

    config, _ = load_config(samples / "recognition_config.json")
    image_data, color_issues = load_and_decode_png(injection_dir / "INJ02_unknown_pseudo_color.png", config)
    checks.append(
        make_check(
            "INJ02B_COLORMAP_PIXEL_CODE",
            "unknown pseudo-color pixel code emits E_IMAGE_FORMAT",
            image_data is None and any(i["code"] == "E_IMAGE_FORMAT" for i in color_issues),
            issues=color_issues,
        )
    )

    inj03 = temp_root / "inj03"
    inj03.mkdir(parents=True, exist_ok=True)
    shutil.copy(injection_dir / "INJ03_silent_cell.png", inj03 / "INJ03_silent_cell.png")
    ledger03, _ = run_pipeline(str(inj03), config)
    silent = [c for c in ledger03["components"] if c["component_id"] == 8]
    others = [c for c in ledger03["components"] if c["component_id"] != 8]
    checks.append(
        make_check(
            "INJ03_SILENT_CELL_ISOLATED",
            "silent cell is degraded and does not pollute other cells",
            len(ledger03["components"]) == 24
            and len(silent) == 1
            and silent[0]["grade_code"] == 9
            and any(i["code"] == "E_CELL_DEGRADED" for i in ledger03["failures"])
            and all(c["grade_code"] != 9 for c in others)
            and len([i for i in ledger03["failures"] if i["code"] == "E_CELL_DEGRADED"]) == 1,
            component_count=len(ledger03["components"]),
            silent_grade=silent[0]["grade_name"] if silent else None,
            other_degraded_count=len([c for c in others if c["grade_code"] == 9]),
        )
    )

    for injection_id, filename in (
        ("INJ04_BBOX_OUTSIDE_GRID", "INJ04_bbox_outside_grid_truth.json"),
        ("INJ05_BBOX_OUTSIDE_IMAGE", "INJ05_bbox_outside_image_truth.json"),
    ):
        raw = json.loads((injection_dir / filename).read_text(encoding="utf-8"))
        _, issues = validate_truth(raw, config)
        checks.append(
            make_check(
                injection_id,
                f"{filename} emits E_TRUTH_INVALID",
                any(i["code"] == "E_TRUTH_INVALID" for i in issues),
                issue_count=len(issues),
                reasons=[i.get("reason") for i in issues],
            )
        )

    invalid_config, config_issues = load_config(injection_dir / "INJ06_invalid_config.json")
    checks.append(
        make_check(
            "INJ06_INVALID_CONFIG",
            "independent invalid config reports all parameter violations",
            invalid_config is None and sum(i["code"] == "E_CONFIG_INVALID" for i in config_issues) >= 6,
            issue_count=len(config_issues),
            fields=sorted({i.get("field") for i in config_issues if i.get("field")}),
        )
    )

    _, field_issues = parse_v1_text((injection_dir / "INJ07_truncated_or_illegal_v1.txt").read_text(encoding="utf-8"))
    checks.append(
        make_check(
            "INJ07_FIELD_TRUNCATED",
            "injected v1 truncation and illegal characters emit E_FIELD_TRUNCATED",
            len(field_issues) >= 2 and all(i["code"] == "E_FIELD_TRUNCATED" for i in field_issues),
            issue_count=len(field_issues),
        )
    )

    boundary_grade = determine_grade(
        {"intensity_permille": 333},
        {"grade_thresholds_permille": [100, 333, 500, 700]},
        False,
    )
    checks.append(
        make_check(
            "BOUNDARY_EQUAL_THRESHOLD",
            "equality to a threshold selects the higher grade and marks the boundary",
            boundary_grade["threshold_boundary"] is True and boundary_grade["grade_code"] == 2,
            **boundary_grade,
        )
    )
    return checks


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# 光伏红外组件账本诊断报告",
        "",
        f"- 总体结果：{'通过' if report['ok'] else '失败'}",
        f"- Python：{report['python']}",
        f"- 平台：{report['platform']}",
        f"- 检查数：{report['summary']['check_count']}；通过：{report['summary']['passed']}；失败：{report['summary']['failed']}",
        f"- 样本组件数：{report['sample']['component_count']}；真值一致：{report['sample']['truth_ok']}/{report['sample']['truth_total']}",
        "",
        "## 检查结果",
        "",
        "| ID | 结果 | 说明 |",
        "| --- | --- | --- |",
    ]
    for check in report["checks"]:
        lines.append(f"| `{check['id']}` | {'PASS' if check['ok'] else 'FAIL'} | {check['title']} |")
    lines.extend(["", "## 增量步骤", "", "| 步骤 | 域一致 | 事件顺序一致 | 旁路一致 | 缓存断言 | 组件数 |", "| --- | --- | --- | --- | --- | --- |"])
    for step in report.get("incremental", {}).get("steps", []):
        lines.append(
            f"| {step['label']} | {step['domain_equal']} | {step['event_order_equal']} | {step['sidecar_equal']} | {step['cache_expectation_ok']} | {step['components']} |"
        )
    lines.extend(["", "## 注入隔离", ""])
    for check in report["checks"]:
        if check["id"].startswith("INJ"):
            lines.append(f"- `{check['id']}`: {'通过' if check['ok'] else '失败'} — {check['title']}")
    lines.append("")
    return "\n".join(lines)


def run_verification(workspace: str | Path, refresh_samples: bool = True) -> dict[str, Any]:
    workspace = Path(workspace)
    samples = workspace / "samples"
    if refresh_samples or not (samples / "judgments_v1.txt").exists():
        generate_sample_tree(samples)
    config, truth, _ = bootstrap_samples(samples)

    with tempfile.TemporaryDirectory(dir=workspace, prefix=".verify-tmp-") as temporary:
        temp_root = Path(temporary)
        checks: list[dict[str, Any]] = []
        checks.extend(verify_exchange(samples))
        incremental_checks, incremental_summary = verify_incremental(workspace, samples, temp_root)
        checks.extend(incremental_checks)
        checks.extend(verify_budget_and_snapshot(workspace, samples, temp_root))
        checks.extend(verify_injections(samples, temp_root))

    sample_ledger, _ = run_pipeline(str(samples), config, truth, previous_state=None, exchange_version="v2")
    truth_total = len(sample_ledger["components"])
    truth_ok = sum(c["truth_attribution"].get("status") == "OK" for c in sample_ledger["components"])
    report: dict[str, Any] = {
        "schema": "pv-ir-diagnostic-report-v1",
        "ok": all(check["ok"] for check in checks),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "summary": {
            "check_count": len(checks),
            "passed": sum(check["ok"] for check in checks),
            "failed": sum(not check["ok"] for check in checks),
        },
        "sample": {
            "image_count": len(sample_ledger["image_fingerprints"]),
            "component_count": truth_total,
            "truth_ok": truth_ok,
            "truth_total": truth_total,
            "boundary_item_count": len(sample_ledger["boundary_grade_items"]),
            "failure_count": len(sample_ledger["failures"]),
        },
        "incremental": incremental_summary,
        "checks": checks,
    }
    report["markdown"] = render_markdown(report)
    return report
