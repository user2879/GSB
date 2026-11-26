from __future__ import annotations

import platform
import shutil
import sys
import tempfile
import json
from pathlib import Path
from typing import Any

from .commands import apply_commands, initialize_workspace
from .config import load_config
from .diagnostics import run_diagnostics
from .ledger import ContentLedger, SnapshotManager
from .pipeline import Pipeline, canonical_fingerprint, canonical_pipeline_state
from .samplegen import generate_sample_set
from .utils import canonical_dumps, fingerprint, read_json, write_json


COMMAND_SEQUENCE: list[dict[str, Any]] = [
    {"op": "replace_view", "view_id": "v1", "variant": 7},
    {"op": "set_extrinsic", "view_id": "v2", "center": [14, 0, 0]},
    {"op": "set_param", "key": "min_parallax", "value": 6},
    {"op": "set_param", "key": "voxel_size", "value": 12},
    {"op": "set_param", "key": "residual_threshold", "value": 1200},
    {"op": "set_param", "key": "quant_level", "value": "Q2"},
    {"op": "add_view", "view_id": "v4", "sequence": 4, "pair_with": ["v3"]},
    {"op": "replace_view", "view_id": "v2", "variant": 3},
    {"op": "remove_view", "view_id": "v3"},
    {"op": "replace_view", "view_id": "missing-view", "variant": 1},
    {"op": "set_param", "key": "min_parallax", "value": 0},
    {"op": "frobnicate", "view_id": "v0"},
]


def setup_workspace(root: Path, config_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    initialize_workspace(root)
    config = load_config(root / "configs" / "reconstruction.json")
    if config_overrides:
        _deep_update(config, config_overrides)
        write_json(root / "configs" / "reconstruction.json", config)
    generate_sample_set(root / "samples", config)
    return config


def _deep_update(target: dict[str, Any], updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value


def _first_diff(left: Any, right: Any, path: str = "$") -> str | None:
    if type(left) is not type(right):
        if isinstance(left, dict) and isinstance(right, dict):
            pass
        else:
            return f"{path}: type {type(left).__name__} != {type(right).__name__}"
    if isinstance(left, dict):
        keys = sorted(set(left) | set(right), key=repr)
        for key in keys:
            if key not in left:
                return f"{path}.{key}: missing in incremental"
            if key not in right:
                return f"{path}.{key}: missing in full"
            diff = _first_diff(left[key], right[key], f"{path}.{key}")
            if diff:
                return diff
    elif isinstance(left, list):
        if len(left) != len(right):
            return f"{path}: list length {len(left)} != {len(right)}"
        for index, (a, b) in enumerate(zip(left, right)):
            diff = _first_diff(a, b, f"{path}[{index}]")
            if diff:
                return diff
    elif left != right:
        return f"{path}: {left!r} != {right!r}"
    return None


def _temp_root(prefix: str) -> Path:
    return Path(tempfile.mkdtemp(prefix=prefix))


def verify_edit_parity() -> dict[str, Any]:
    incremental_root = _temp_root("mdr-incremental-")
    full_root = _temp_root("mdr-full-")
    try:
        setup_workspace(incremental_root)
        setup_workspace(full_root)

        Pipeline(incremental_root, parallel=True).run()
        cumulative_errors: list[dict[str, str]] = []
        step_summaries: list[dict[str, Any]] = []
        for index, command in enumerate(COMMAND_SEQUENCE):
            result = apply_commands(incremental_root, [command])
            cumulative_errors.extend(result["errors"])
            state = Pipeline(
                incremental_root,
                parallel=(index % 2 == 0),
                command_errors=cumulative_errors,
            ).run()
            step_summaries.append(
                {
                    "index": index,
                    "op": command.get("op"),
                    "fingerprint": canonical_fingerprint(state)[:16],
                    "pending": len(state["pending"]),
                    "input_errors": state["business_code_counts"]["E_INPUT_INVALID"],
                }
            )

        full_errors: list[dict[str, str]] = []
        for command in COMMAND_SEQUENCE:
            result = apply_commands(full_root, [command])
            full_errors.extend(result["errors"])
        full_state = Pipeline(full_root, parallel=False, command_errors=full_errors).run()
        incremental_state = Pipeline(incremental_root, parallel=False, command_errors=cumulative_errors).run()

        canonical_a = json.loads(canonical_dumps(canonical_pipeline_state(incremental_state)))
        canonical_b = json.loads(canonical_dumps(canonical_pipeline_state(full_state)))
        diff = _first_diff(canonical_a, canonical_b)
        return {
            "passed": diff is None,
            "first_diff": diff,
            "incremental_fingerprint": canonical_fingerprint(incremental_state),
            "full_fingerprint": canonical_fingerprint(full_state),
            "command_count": len(COMMAND_SEQUENCE),
            "command_error_count": len(cumulative_errors),
            "step_summaries": step_summaries,
            "canonical_ledger_entries": len(incremental_state["canonical_ledger"]),
            "business_code_counts": incremental_state["business_code_counts"],
        }
    finally:
        shutil.rmtree(incremental_root, ignore_errors=True)
        shutil.rmtree(full_root, ignore_errors=True)


def verify_precise_invalidation() -> dict[str, Any]:
    root = _temp_root("mdr-invalidation-")
    try:
        setup_workspace(root)
        Pipeline(root).run()
        apply_commands(root, [{"op": "replace_view", "view_id": "v1", "variant": 11}])
        state = Pipeline(root, parallel=False).run()
        decisions: dict[tuple[str, str], str] = {}
        for event in state["cache_events"]:
            key = (event["stage"], event["unit_id"])
            if event["decision"] in {"hit", "rebuilt"}:
                decisions[key] = event["decision"]
        checks = {
            "unrelated_feature_v0_hit": decisions.get(("feature", "v0")) == "hit",
            "replaced_feature_v1_rebuilt": decisions.get(("feature", "v1")) == "rebuilt",
            "affected_pair_v0_v1_not_hit": decisions.get(("match", "v0|v1")) == "rebuilt",
            "unrelated_pair_v2_v3_hit": decisions.get(("match", "v2|v3")) == "hit",
        }
        return {"passed": all(checks.values()), "checks": checks}
    finally:
        shutil.rmtree(root, ignore_errors=True)


def verify_concurrency_and_budgets() -> dict[str, Any]:
    worker_fingerprints: dict[str, str] = {}
    for workers in (1, 2, 4):
        root = _temp_root(f"mdr-workers-{workers}-")
        try:
            setup_workspace(root, {"budgets": {"max_workers": workers}})
            state = Pipeline(root, parallel=workers > 1).run()
            worker_fingerprints[str(workers)] = canonical_fingerprint(state)
        finally:
            shutil.rmtree(root, ignore_errors=True)
    worker_ok = len(set(worker_fingerprints.values())) == 1

    budget_cases: dict[str, Any] = {}
    for case_name, overrides in (
        ("point_budget", {"budgets": {"global_point_budget": 5, "view_point_quota": 10_000}}),
        ("voxel_subdivision_budget", {"budgets": {"max_subdivisions_per_batch": 1}}),
    ):
        serial_root = _temp_root(f"mdr-budget-serial-{case_name}-")
        parallel_root = _temp_root(f"mdr-budget-parallel-{case_name}-")
        try:
            setup_workspace(serial_root, overrides)
            setup_workspace(parallel_root, {**overrides, "budgets": {**overrides["budgets"], "max_workers": 4}})
            serial_state = Pipeline(serial_root, parallel=False).run()
            parallel_state = Pipeline(parallel_root, parallel=True).run()
            budget_cases[case_name] = {
                "passed": canonical_fingerprint(serial_state) == canonical_fingerprint(parallel_state),
                "fingerprint": canonical_fingerprint(serial_state),
                "pending": len(serial_state["pending"]),
                "business_codes": serial_state["business_code_counts"],
            }
        finally:
            shutil.rmtree(serial_root, ignore_errors=True)
            shutil.rmtree(parallel_root, ignore_errors=True)

    resume_root = _temp_root("mdr-resume-")
    full_root = _temp_root("mdr-resume-full-")
    try:
        setup_workspace(resume_root, {"budgets": {"global_point_budget": 3, "max_subdivisions_per_batch": 1}})
        first = Pipeline(resume_root, parallel=True).run()
        first_pending = len(first["pending"])
        apply_commands(
            resume_root,
            [
                {"op": "set_param", "key": "global_point_budget", "value": 10_000},
                {"op": "set_param", "key": "view_point_quota", "value": 10_000},
                {"op": "set_param", "key": "max_subdivisions_per_batch", "value": 10_000},
            ],
        )
        resumed = Pipeline(resume_root, parallel=True).run()
        setup_workspace(
            full_root,
            {
                "budgets": {
                    "global_point_budget": 10_000,
                    "view_point_quota": 10_000,
                    "max_subdivisions_per_batch": 10_000,
                }
            },
        )
        full_state = Pipeline(full_root, parallel=False).run()
        resumed_products = canonical_pipeline_state(resumed)
        full_products = canonical_pipeline_state(full_state)
        resumed_products["budget"] = {}
        full_products["budget"] = {}
        product_match = fingerprint(resumed_products) == fingerprint(full_products)
        resume_ok = not resumed["pending"] and product_match
        budget_cases["resume_priority"] = {
            "passed": resume_ok,
            "first_pending": first_pending,
            "resumed_pending": len(resumed["pending"]),
            "fingerprint_match_full": product_match,
        }
    finally:
        shutil.rmtree(resume_root, ignore_errors=True)
        shutil.rmtree(full_root, ignore_errors=True)

    return {
        "passed": worker_ok and all(case["passed"] for case in budget_cases.values()),
        "worker_fingerprints": worker_fingerprints,
        "worker_determinism": worker_ok,
        "budget_cases": budget_cases,
    }


def verify_recovery() -> dict[str, Any]:
    root = _temp_root("mdr-recovery-")
    try:
        setup_workspace(root)
        Pipeline(root).run()
        manager = SnapshotManager(ContentLedger(root))
        healthy = manager.recover()
        staging = root / "ledger" / "snapshots" / ".staging-killed-during-commit"
        staging.mkdir(parents=True, exist_ok=True)
        (staging / "partial.json").write_text("{partial", encoding="utf-8")
        with_stale = manager.recover()
        manager.current_path.write_text("{broken pointer", encoding="utf-8")
        broken = manager.recover()
        result = {
            "passed": (
                healthy["decision"] == "resume_with_snapshot"
                and with_stale["decision"] == "resume_with_snapshot"
                and len(with_stale["stale_staging"]) >= 1
                and broken["decision"] == "rebuild"
            ),
            "healthy_decision": healthy["decision"],
            "stale_staging_ignored": len(with_stale["stale_staging"]),
            "broken_decision": broken["decision"],
            "broken_code": broken["code"],
        }
        return result
    finally:
        shutil.rmtree(root, ignore_errors=True)


def run_acceptance(root: str | Path | None = None, run_diagnostics_report: bool = True) -> dict[str, Any]:
    workspace_root = Path(root or Path.cwd()).resolve()
    setup_workspace(workspace_root)
    main_state = Pipeline(workspace_root, parallel=True).run()

    parity = verify_edit_parity()
    invalidation = verify_precise_invalidation()
    concurrency = verify_concurrency_and_budgets()
    recovery = verify_recovery()

    diagnostics_summary: dict[str, Any] | None = None
    if run_diagnostics_report:
        diagnostics_report = run_diagnostics(workspace_root)
        diagnostics_summary = {
            "default_counts": diagnostics_report["default_comparison"]["counts"],
            "boundary_triggered": diagnostics_report["boundary_probe"].get("triggered", False),
            "boundary": diagnostics_report["boundary_probe"],
        }

    environment = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "executable": sys.executable,
    }
    try:
        import numpy
        import PIL

        environment["numpy"] = numpy.__version__
        environment["pillow"] = PIL.__version__
    except Exception as exc:
        environment["dependency_error"] = repr(exc)
    try:
        import tkinter  # noqa: F401

        environment["tkinter"] = "available"
    except Exception as exc:
        environment["tkinter"] = f"unavailable: {type(exc).__name__}: {exc}"

    report = {
        "passed": all(
            [
                parity["passed"],
                invalidation["passed"],
                concurrency["passed"],
                recovery["passed"],
            ]
        ),
        "environment": environment,
        "main_run": {
            "fingerprint": canonical_fingerprint(main_state),
            "business_code_counts": main_state["business_code_counts"],
            "feature_counts": {k: v["feature_count"] for k, v in main_state["features"].items()},
            "match_counts": {k: v["match_count"] for k, v in main_state["matches"].items()},
            "triangulation": {
                "accepted": main_state["triangulation"]["point_count"],
                "rejected": main_state["triangulation"]["rejected_count"],
            },
            "depth_conflicts": {k: v["conflict_count"] for k, v in main_state["depth_maps"].items()},
            "voxel_counts": main_state["voxels"]["counts"],
        },
        "edit_parity": parity,
        "precise_invalidation": invalidation,
        "concurrency_and_budgets": concurrency,
        "recovery": recovery,
        "diagnostics": diagnostics_summary,
    }
    write_json(workspace_root / "reports" / "acceptance-report.json", report)
    write_markdown_report(workspace_root / "reports" / "acceptance-report.md", report)
    return report


def write_markdown_report(path: Path, report: dict[str, Any]) -> None:
    parity = report["edit_parity"]
    inv = report["precise_invalidation"]
    cb = report["concurrency_and_budgets"]
    rec = report["recovery"]
    diag = report.get("diagnostics") or {}
    lines = [
        "# 多视图重建原型验收报告",
        "",
        f"- 总结果：{'通过' if report['passed'] else '未通过'}",
        f"- Python：{report['environment']['python']} ({report['environment']['platform']})",
        f"- NumPy/Pillow：{report['environment'].get('numpy')} / {report['environment'].get('pillow')}",
        f"- Tkinter：{report['environment'].get('tkinter')}",
        f"- 主运行规范指纹：`{report['main_run']['fingerprint']}`",
        "",
        "## 增量与全量一致性",
        "",
        f"- 结果：{'通过' if parity['passed'] else '未通过'}",
        f"- 增量指纹：`{parity['incremental_fingerprint']}`",
        f"- 全量指纹：`{parity['full_fingerprint']}`",
        f"- 命令数：{parity['command_count']}；输入错误码：{parity['command_error_count']}",
        f"- 规范化账本条目：{parity['canonical_ledger_entries']}",
    ]
    if parity.get("first_diff"):
        lines.append(f"- 首个差异：`{parity['first_diff']}`")
    lines.extend([
        "",
        "## 精确失效传播",
        "",
        f"- 结果：{'通过' if inv['passed'] else '未通过'}",
    ])
    for key, value in inv["checks"].items():
        lines.append(f"- `{key}`：{value}")
    lines.extend([
        "",
        "## 并发与预算",
        "",
        f"- 不同线程数指纹一致：{cb['worker_determinism']}",
    ])
    for name, case in cb["budget_cases"].items():
        lines.append(f"- `{name}`：passed={case['passed']}，pending={case.get('pending', case.get('first_pending'))}")
    lines.extend([
        "",
        "## 恢复判定",
        "",
        f"- 结果：{'通过' if rec['passed'] else '未通过'}",
        f"- 健康快照：`{rec['healthy_decision']}`；残留临时目录数：{rec['stale_staging_ignored']}",
        f"- 损坏当前指针：`{rec['broken_decision']}` / `{rec['broken_code']}`",
        "",
        "## 误差归因",
        "",
        f"- 默认档位统计：`{diag.get('default_counts', {})}`",
        f"- 边界探针触发：{diag.get('boundary_triggered')}",
    ])
    boundary = diag.get("boundary") or {}
    if boundary.get("triggered"):
        lines.append(f"- 探针体素：`{boundary['target_voxel_id']}`；阈值：{boundary['probe_threshold_q']}")
        lines.append(f"- 翻转归因统计：`{boundary.get('attribution_summary', {})}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
