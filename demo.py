"""Repeatable end-to-end demonstration using only relative workspace paths."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from walcrash.engine import ReplayEngine
from walcrash.snapshots import create_snapshot, restore_snapshot
from walcrash.storage import Storage
from walcrash.validation import load_sample
from walcrash.verification import verify_replay


PROJECT_ROOT = Path(__file__).resolve().parent
MAIN_WORKSPACE = "demo_workspace"
ALT_WORKSPACE = "demo_alt_workspace"
SAMPLE = "samples/sample.json"
ALT_SAMPLE = "samples/alt_pages.json"
SNAPSHOT = "snapshots/demo_session.json"


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def short(payload: dict[str, Any], keys: tuple[str, ...]) -> str:
    return json.dumps({key: payload.get(key) for key in keys}, ensure_ascii=False, sort_keys=True)


def max_concurrency(tasks: list[dict[str, Any]], field: str, global_limit: int) -> int:
    events: list[tuple[int, int]] = []
    for task in tasks:
        start = task.get("started_ms")
        finish = task.get("finished_ms")
        if start is None:
            continue
        if finish is None:
            finish = start
        events.append((start, 1))
        events.append((finish, -1))
    # End events sort before start events at the same millisecond.
    events.sort(key=lambda item: (item[0], item[1]))
    active = 0
    maximum = 0
    for _, delta in events:
        active += delta
        maximum = max(maximum, active)
    if maximum > global_limit:
        raise AssertionError(f"observed concurrency {maximum} exceeds global limit {global_limit}")
    return maximum


def cli(*args: str, expect: int = 0) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-m", "walcrash", *args],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=90,
    )
    if result.returncode != expect:
        raise AssertionError(
            f"command returned {result.returncode}, expected {expect}: {args}\n"
            f"stdout={result.stdout}\nstderr={result.stderr}"
        )
    return result


def main() -> int:
    os.chdir(PROJECT_ROOT)
    for name in (MAIN_WORKSPACE, ALT_WORKSPACE):
        path = Path(name)
        if path.exists():
            shutil.rmtree(path)

    section("1. 加载强校验样本")
    sample = load_sample(SAMPLE)
    storage = Storage(MAIN_WORKSPACE)
    storage.load_fresh(sample)
    print(
        json.dumps(
            {
                "pages": [page.page_id for page in sample.pages],
                "wal_entries": len(sample.entries),
                "crash_point_lsn": sample.crash_point_lsn,
                "committed_txids": list(sample.committed_txids),
                "wal_digest": sample.wal_digest,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    section("2. 预算内并行重放（日志语义由单线程 LSN 定序提交）")
    replay = ReplayEngine(storage).replay(delay_ms=18)
    tasks = storage.tasks(replay["run_id"])
    observed = max_concurrency(tasks, "started_ms", storage.budget().global_slots)
    per_tx: dict[str, int] = defaultdict(int)
    for task in tasks:
        per_tx[task["txid"]] += 1
    print(short(replay, ("run_id", "task_count", "batch_count", "cursor_start", "cursor_end")))
    print(
        json.dumps(
            {
                "observed_global_concurrency": observed,
                "global_slot_limit": storage.budget().global_slots,
                "task_count_by_transaction": dict(sorted(per_tx.items())),
                "per_transaction_quota": storage.budget().per_tx_quota,
                "waiting_tasks": sum(1 for task in tasks if task.get("wait_started_ms") is not None),
                "page_files": len(storage.page_versions()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if observed > storage.budget().per_tx_quota * 3:
        raise AssertionError("unexpected scheduler concurrency")

    section("3. 检查点保存与正常重放校验")
    checkpoint = storage.save_checkpoint("demo-manual")
    verification = verify_replay(storage)
    print(json.dumps({"checkpoint": checkpoint, "verification": verification}, ensure_ascii=False, indent=2))
    snapshot = create_snapshot(storage, SNAPSHOT)
    print(json.dumps({"snapshot": snapshot}, ensure_ascii=False, indent=2))
    assert verification["crash_point_lsn"] == 10
    assert verification["committed_txids"] == ["tx-001", "tx-002", "tx-003"]

    section("4. 强制终止模拟：只保留完整提交批次，不生成退出报告")
    cli("--workspace", MAIN_WORKSPACE, "--action", "load", "--input", SAMPLE)
    crash = cli(
        "--workspace",
        MAIN_WORKSPACE,
        "--action",
        "run",
        "--delay-ms",
        "90",
        "--crash-after-batches",
        "1",
        expect=91,
    )
    crashed = Storage(MAIN_WORKSPACE)
    crashed_status = crashed.status()
    active_states = {
        state: sum(1 for task in crashed.tasks() if task["state"] == state)
        for state in sorted({task["state"] for task in crashed.tasks()})
    }
    print(
        json.dumps(
            {
                "child_exit_code": crash.returncode,
                "ledger_cursor_after_crash": crashed.ledger_cursor(),
                "committed_batches": [
                    {"batch": row["batch_id"], "range": [row["start_lsn"], row["end_lsn"]]}
                    for row in crashed.batches()
                ],
                "task_states": active_states,
                "last_run": crashed_status["last_run"],
                "exit_report_claimed": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    assert crashed.ledger_cursor() == 4
    assert len(crashed.batches()) == 1
    assert active_states.get("running", 0) + active_states.get("queued", 0) > 0
    assert "second run report" not in json.dumps(crashed_status)

    section("5. 断点恢复与剩余日志重放校验")
    resumed = ReplayEngine(crashed).replay(delay_ms=8)
    print(short(resumed, ("run_id", "cursor_start", "cursor_end", "recovered_orphan_tasks", "batch_count")))
    assert resumed["recovered_orphan_tasks"] > 0
    resumed_verification = verify_replay(crashed)
    print(json.dumps(resumed_verification, ensure_ascii=False, indent=2))
    ledger_check = crashed.verify_ledger()
    print(json.dumps({"ledger_verification": ledger_check}, ensure_ascii=False, indent=2))
    assert ledger_check["ok"]
    ledger_export = cli(
        "--workspace",
        MAIN_WORKSPACE,
        "--action",
        "export-ledger",
        "--output",
        "ledger.json",
    )
    print(ledger_export.stdout.strip())

    section("6. 快照版本识别与整体替换差异")
    alt = Storage(ALT_WORKSPACE)
    alt.load_fresh(load_sample(ALT_SAMPLE))
    ReplayEngine(alt).replay(delay_ms=5)
    alt_snapshot_dir = alt.snapshots_dir
    alt_snapshot_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(storage.workspace / SNAPSHOT, alt_snapshot_dir / "demo_session.json")
    restored = restore_snapshot(alt, "snapshots/demo_session.json")
    print(
        json.dumps(
            {
                "restore_mode": restored["restore_mode"],
                "log_changed": restored["log_changed"],
                "page_diff": restored["page_diff"],
                "restored": restored["restored"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    assert restored["page_diff"]["added"] == ["p3"]
    assert restored["page_diff"]["deleted"] == ["p4"]
    assert [item["page_id"] for item in restored["page_diff"]["changed"]] == ["p1", "p2"]
    alt_verify = verify_replay(alt)
    assert alt_verify["ok"]
    print(json.dumps({"restored_workspace_verification": short(alt_verify, ("ok", "cursor", "crash_point_lsn"))}, ensure_ascii=False))

    section("7. 三类显式异常注入")
    injections = []
    for mode, expected_code, expected_exit in (
        ("page_size", "E_PAGE", 10),
        ("order", "E_ORDER", 11),
        ("snapshot_digest", "E_SNAPSHOT", 13),
    ):
        result = cli("--workspace", MAIN_WORKSPACE, "--inject", mode, expect=expected_exit)
        document = json.loads(result.stdout)
        if document.get("code") != expected_code:
            raise AssertionError(f"injection {mode} returned {document}")
        injections.append(
            {
                "mode": mode,
                "code": document["code"],
                "field": document.get("context", {}).get("field"),
                "exit_code": result.returncode,
            }
        )
    print(json.dumps(injections, ensure_ascii=False, indent=2))
    injection_log = storage.logs_dir / "injections.ndjson"
    lines = [line for line in injection_log.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) >= 3
    print(f"structured_injection_log={injection_log} lines={len(lines)}")

    report = {
        "ok": True,
        "sample": SAMPLE,
        "main_workspace": MAIN_WORKSPACE,
        "alt_workspace": ALT_WORKSPACE,
        "normal_replay": {
            "task_count": replay["task_count"],
            "batch_count": replay["batch_count"],
            "observed_global_concurrency": observed,
            "crash_point_lsn": verification["crash_point_lsn"],
        },
        "crash_drill": {
            "exit_code": crash.returncode,
            "ledger_cursor": 4,
            "recovered_orphan_tasks": resumed["recovered_orphan_tasks"],
            "exit_report_claimed": False,
        },
        "resume_verification": resumed_verification,
        "snapshot_restore": {
            "restore_mode": restored["restore_mode"],
            "page_diff": restored["page_diff"],
            "ledger": restored["ledger"],
        },
        "injections": injections,
        "ledger_verification": ledger_check,
        "artifacts": {
            "snapshot": f"{MAIN_WORKSPACE}/{SNAPSHOT}",
            "ledger_export": f"{MAIN_WORKSPACE}/ledger.json",
            "injection_log": f"{MAIN_WORKSPACE}/logs/injections.ndjson",
            "database": f"{MAIN_WORKSPACE}/state.sqlite3",
            "pages": f"{MAIN_WORKSPACE}/pages/",
        },
    }
    report_path = storage.workspace / "demo_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    section("演示完成")
    print(f"report={report_path.relative_to(PROJECT_ROOT)}")
    print(f"snapshot={Path(MAIN_WORKSPACE) / SNAPSHOT}")
    print(f"ledger_export={Path(MAIN_WORKSPACE) / 'ledger.json'}")
    print("all_assertions=passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

