from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from .clock import ManualClock, MonotonicClock
from .config import WorkbenchConfig
from .environment import check_environment
from .hashing import canonical_json
from .injection import run_injection_suite
from .leases import LeaseExecutor
from .studio import Studio


def _same(a: Any, b: Any) -> bool:
    return canonical_json(a) == canonical_json(b)


def _core_signature(result: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "samples",
        "windows",
        "components",
        "keypoints",
        "reading",
        "conflicts",
        "failures",
        "pending_queue",
        "business_code_counts",
        "reuse",
        "recompute",
        "cache_stats",
    ]
    signature = {key: result.get(key) for key in keys}
    budget = dict(result.get("budget", {}))
    for transient in ("events", "comparison_events"):
        budget.pop(transient, None)
    signature["budget"] = budget
    return signature


def verify_command_replay() -> dict[str, Any]:
    steps: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="vw-replay-") as td:
        root_a = Path(td) / "a"
        root_b = Path(td) / "b"
        config = WorkbenchConfig(sample_count=3, global_point_budget=180, max_active_leases=24)
        studio_a = Studio.create(root_a, config, ManualClock(1_700_000_000_000))
        studio_b = Studio.create(root_b, config, ManualClock(1_700_000_000_000))
        baseline_a = studio_a.engine.run(1)
        baseline_b = studio_b.engine.run(1)
        steps.append({"step": "baseline", "same": _same(baseline_a, baseline_b)})

        commands = [
            {"op": "update_config", "patch": {"stride": 32}},
            {"op": "update_config", "patch": {"quant_gear": "fine", "fixed_frac_bits": 16}},
            {"op": "replace_annotation_version", "version_id": "kv-relabel"},
            {
                "op": "replace_annotation_version",
                "version_id": "kv-content",
                "jitter_by_image": {"sample_001": 1},
            },
            {
                "op": "update_config",
                "patch": {"window_size": 32, "stride": 16, "quant_gear": "normal", "fixed_frac_bits": 8},
            },
            {
                "op": "replace_image",
                "image_id": "sample_002",
                "columns": 4,
                "counts": [7, 6, 8, 5],
                "seed": 4242,
            },
            {"op": "add_image", "image_id": "sample_004", "columns": 3, "counts": [6, 7, 8], "seed": 777},
            {"op": "delete_image", "image_id": "sample_003"},
            {"op": "set_budget", "patch": {"global_point_budget": 70, "column_point_quota": 12}},
            {"op": "set_lease_limit", "max_active_leases": 24},
            {"op": "set_budget", "patch": {"global_point_budget": 260, "column_point_quota": 12}},
            {"op": "update_config", "patch": {"window_size": 64, "stride": 64}},
        ]
        for index, command in enumerate(commands, start=2):
            result_a = studio_a.apply(command)
            result_b = studio_b.apply(command)
            same = _same(result_a, result_b)
            steps.append({"step": index, "op": command["op"], "same": same})
            if not same:
                for key in result_a:
                    if not _same(result_a.get(key), result_b.get(key)):
                        steps[-1]["first_different_key"] = key
                        break
                break
    return {"ok": all(step["same"] for step in steps), "steps": steps}


def verify_concurrency() -> dict[str, Any]:
    outcomes = []
    with tempfile.TemporaryDirectory(prefix="vw-concurrency-") as td:
        root = Path(td) / "ws"
        config = WorkbenchConfig(sample_count=3, stride=64, max_active_leases=24)
        Studio.create(root, config)
        signatures = {}
        for workers in (1, 2, 4):
            raw = json.loads((root / "configs" / "workbench.json").read_text(encoding="utf-8"))
            raw["max_workers"] = workers
            (root / "configs" / "workbench.json").write_text(
                json.dumps(raw, sort_keys=True), encoding="utf-8"
            )
            engine = Studio(root, ManualClock(1_700_000_100_000)).engine
            engine.write_snapshots = False
            result = engine.run(1)
            signatures[workers] = _core_signature(result)
            outcomes.append({"workers": workers, "keypoints": len(result["keypoints"])})
        ok = _same(signatures[1], signatures[2]) and _same(signatures[1], signatures[4])

    from .scheduler import run_ordered

    jobs = list(range(20))
    ordered = run_ordered(jobs, lambda x: x, 4)
    return {"ok": ok and ordered == jobs, "runs": outcomes, "ordered_helper": ordered == jobs}


def verify_lease_reclamation() -> dict[str, Any]:
    executor = LeaseExecutor(MonotonicClock(), max_active_leases=4)
    granted = executor.acquire("sample-x:c0", 100, 160, reserved_points=7, ttl_ms=40)
    lease = granted["lease"]
    before = executor.active_lease_for_target("sample-x:c0")
    time.sleep(0.12)
    expired = executor.reap_expired()
    after = executor.active_lease_for_target("sample-x:c0")

    generation_executor = LeaseExecutor(ManualClock(5_000), max_active_leases=4)
    generation_lease = generation_executor.acquire("sample-y:c0", 10, 80, 2, 10_000)["lease"]
    stale = generation_executor.release(generation_lease["lease_id"], generation_lease["generation"] + 999)
    return {
        "ok": granted["ok"]
        and before is not None
        and len(expired) == 1
        and after is None
        and executor.reserved_points == 0
        and stale["code"] == "E_LEASE_GENERATION",
        "expired_events": expired,
        "stale_generation_code": stale["code"],
    }


def verify_budget() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="vw-budget-") as td:
        root = Path(td) / "budget"
        config = WorkbenchConfig(sample_count=2, global_point_budget=18, column_point_quota=12)
        studio = Studio.create(root, config, ManualClock(1_700_000_200_000))
        first = studio.engine.run(1)
        first_pending = first["pending_queue"][0]["unit_id"] if first["pending_queue"] else None
        budget_pending_ok = bool(first_pending) and first["business_code_counts"]["E_POINT_BUDGET_PENDING"] > 0

        raw = studio.read_config_raw()
        raw["global_point_budget"] = 500
        studio.write_config_raw(raw)
        second = studio.engine.run(2)
        priority_ok = bool(
            first_pending
            and second["budget"]["candidate_ids"]
            and second["budget"]["candidate_ids"][0] == first_pending
            and second["budget"]["approved_ids"]
            and second["budget"]["approved_ids"][0] == first_pending
        )

        raw = studio.read_config_raw()
        raw["column_point_quota"] = 3
        studio.write_config_raw(raw)
        third = studio.engine.run(3)
        quota_ok = third["business_code_counts"]["E_QUOTA_EXCEEDED"] > 0 and bool(
            third["budget"]["quota_blocked_ids"]
        )
        return {
            "ok": budget_pending_ok and priority_ok and quota_ok,
            "first_pending": first_pending,
            "first_pending_count": len(first["pending_queue"]),
            "priority_ok": priority_ok,
            "quota_ok": quota_ok,
            "quota_events": third["business_code_counts"]["E_QUOTA_EXCEEDED"],
        }


def verify_gui_smoke(root: str | Path) -> dict[str, Any]:
    try:
        from .gui import WorkbenchGUI, _bootstrap_tkinter

        tk, _ttk = _bootstrap_tkinter()
        gui_root = tk.Tk()
        WorkbenchGUI(gui_root, Path(root))
        gui_root.update_idletasks()
        gui_root.update()
        gui_root.destroy()
        return {"ok": True}
    except Exception as exc:
        return {"ok": False, "detail": f"{type(exc).__name__}: {exc}"}


def run_pytest(root: Path) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q"],
            cwd=root,
            text=True,
            capture_output=True,
            timeout=180,
        )
        log = (proc.stdout + "\n" + proc.stderr)[-12000:]
        (root / "reports" / "pytest_output.txt").write_text(log, encoding="utf-8")
        return {"ok": proc.returncode == 0, "returncode": proc.returncode, "tail": log[-2000:]}
    except Exception as exc:
        return {"ok": False, "detail": repr(exc)}


def run_acceptance(root: str | Path = ".", run_tests: bool = True) -> dict[str, Any]:
    root = Path(root).resolve()
    (root / "reports").mkdir(parents=True, exist_ok=True)
    environment = check_environment()

    if not (root / "configs" / "workbench.json").exists():
        Studio.create(root)
    baseline_studio = Studio(root, ManualClock(1_700_000_300_000))
    baseline = baseline_studio.engine.run(8000)

    injection = run_injection_suite(root)
    replay = verify_command_replay()
    concurrency = verify_concurrency()
    lease = verify_lease_reclamation()
    budget = verify_budget()
    gui = verify_gui_smoke(root)
    tests = run_pytest(root) if run_tests else {"ok": True, "skipped": True}

    report = {
        "ok": all(
            [
                environment["python"]["ok"],
                environment["numpy"]["ok"],
                environment["loopback_tcp"]["ok"],
                environment["multiprocessing"]["ok"],
                environment["tkinter"]["ok"],
                not baseline["failures"],
                replay["ok"],
                concurrency["ok"],
                lease["ok"],
                budget["ok"],
                injection["all_injections_hit"],
                injection["good_samples_isolated"],
                gui["ok"],
                tests["ok"],
            ]
        ),
        "environment": environment,
        "baseline": {
            "samples": len(baseline["samples"]),
            "keypoints": len(baseline["keypoints"]),
            "conflicts": len(baseline["conflicts"]),
            "failures": len(baseline["failures"]),
            "windows": len(baseline["windows"]),
        },
        "command_replay": replay,
        "concurrency": concurrency,
        "lease_reclamation": lease,
        "budget": budget,
        "injection": {
            "all_injections_hit": injection["all_injections_hit"],
            "good_samples_isolated": injection["good_samples_isolated"],
            "results": injection["injection_results"],
        },
        "gui_smoke": gui,
        "pytest": tests,
    }
    report_path = root / "reports" / "acceptance_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, default=str), encoding="utf-8")
    markdown = _render_markdown(report)
    (root / "reports" / "verification.md").write_text(markdown, encoding="utf-8")
    return report


def _render_markdown(report: dict[str, Any]) -> str:
    env = report["environment"]
    lines = [
        "# 实际验证结果",
        "",
        f"总体验收：{'通过' if report['ok'] else '未通过'}",
        "",
        "## 环境",
        f"- Python {env['python']['version']} ({env['python']['executable']})",
        f"- NumPy {env['numpy'].get('version')}",
        f"- Tkinter: {env['tkinter']}",
        f"- 回环 TCP: {env['loopback_tcp']}",
        f"- 多进程: {env['multiprocessing']}",
        "",
        "## 主样本基线",
        f"- {report['baseline']}",
        "",
        "## 一致性",
        f"- 命令重放逐字段一致：{report['command_replay']['ok']}",
        f"- 并发度 1/2/4 核心字段一致：{report['concurrency']['ok']}",
        f"- 租约真实回收/代次拒绝：{report['lease_reclamation']['ok']}",
        f"- 预算/配额/待办优先：{report['budget']['ok']}",
        "",
        "## 注入",
    ]
    for item in report["injection"]["results"]:
        lines.append(f"- {item['injection_id']} {item['kind']}: hit={item['hit']}")
    lines += [
        f"- 正常样本隔离：{report['injection']['good_samples_isolated']}",
        "",
        "## GUI",
        f"- {report['gui_smoke']}",
        "",
        "## pytest",
        f"- returncode={report['pytest'].get('returncode', 'skipped')} ok={report['pytest']['ok']}",
        "",
    ]
    return "\n".join(lines)
