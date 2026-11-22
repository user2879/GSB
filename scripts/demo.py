"""一键验收演示：覆盖需求 1-8 的端到端场景并把结果写入 artifacts/demo/。"""
from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts" / "demo"


def run_cli(*args: str) -> tuple[int, str, str]:
    proc = subprocess.run(
        [sys.executable, "-m", "configreg.cli", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


def save(name: str, payload: str) -> None:
    ART.mkdir(parents=True, exist_ok=True)
    (ART / name).write_text(payload, encoding="utf-8")


def main() -> int:
    ART.mkdir(parents=True, exist_ok=True)
    ledger = ART / "ledger.sqlite3"
    if ledger.exists():
        ledger.unlink()
    summary: dict[str, object] = {"steps": []}

    def step(title: str, code: int, stdout: str, stderr: str, filename: str) -> dict:
        save(filename, stdout or stderr)
        record = {"step": title, "exit_code": code, "ok": code == 0}
        try:
            record["payload"] = json.loads(stdout) if stdout.strip() else None
        except json.JSONDecodeError:
            record["payload"] = None
        summary["steps"].append(record)
        print(f"[{'OK' if code == 0 else 'FAIL'}] {title} (exit={code})")
        if stderr.strip():
            print("  stderr:", stderr.strip().splitlines()[-1][:160])
        return record

    code, out, err = run_cli("validate")
    step("1. 清单与注册表强校验", code, out, err, "01_validate.json")

    code, out, err = run_cli("diff", "--precision", "4")
    step("2. 差异度量/影响链/定点误差界", code, out, err, "02_diff.json")

    common_run = [
        "run",
        "--quiet",
        "--ledger",
        str(ledger),
        "--log-dir",
        str(ART / "logs"),
        "--precision",
        "4",
        "--batch-size",
        "2",
        "--socket-timeout-ms",
        "800",
        "--max-attempts",
        "5",
        "--max-elapsed-delta-ms",
        "50",
        "--max-mismatch-nodes",
        "0",
    ]
    code, out, err = run_cli(*common_run)
    step("3-6. 子进程采集/预算判定/并行重试/账本事务", code, out, err, "03_run.json")

    conn = sqlite3.connect(ledger)
    retry_rows = [
        json.loads(row[1])
        for row in conn.execute("SELECT task_key, payload FROM scheduler_records")
        if len(json.loads(row[1])["attempts"]) > 1
    ]
    summary["retry_evidence"] = [
        {
            "task_key": row["task_key"],
            "attempts": [(a["attempt_no"], a["outcome"]) for a in row["attempts"]],
            "global_wait_ms": row["global_wait_ms"],
            "node_wait_ms": row["node_wait_ms"],
        }
        for row in retry_rows
    ]
    conn.close()

    code, out, err = run_cli(
        "replay",
        "--quiet",
        "--ledger",
        str(ledger),
        "--precision",
        "4",
        "--socket-timeout-ms",
        "800",
        "--max-attempts",
        "5",
    )
    step("8. 重放校验（应一致）", code, out, err, "04_replay.json")

    replace_run = list(common_run)
    replace_run[replace_run.index("--precision") + 1] = "6"
    code, out, err = run_cli(*replace_run)
    step("7. 精度口径变化触发快照整体替换", code, out, err, "05_replace.json")

    if ledger.exists():
        ledger.unlink()
    code, out, err = run_cli(*common_run, "--crash-after-batches", "1")
    crash_record = step("8a. 强制终止演练（无退出报告）", code, out, err, "06_crash.json")
    summary["crash_exit_code"] = code

    code, out, err = run_cli(*common_run)
    step("8b. 重启恢复：未知项重跑并补齐批次", code, out, err, "07_resume.json")

    code, out, err = run_cli(
        "replay",
        "--quiet",
        "--ledger",
        str(ledger),
        "--precision",
        "4",
        "--socket-timeout-ms",
        "800",
        "--max-attempts",
        "5",
    )
    step("8c. 恢复后重放校验", code, out, err, "08_replay_after_resume.json")

    dispersion_ledger = ART / "dispersion.sqlite3"
    if dispersion_ledger.exists():
        dispersion_ledger.unlink()
    code, out, err = run_cli(
        "run",
        "--quiet",
        "--baseline",
        "data/dispersion/baseline.json",
        "--candidate",
        "data/dispersion/candidate.json",
        "--registry",
        "data/dispersion/registry.json",
        "--ledger",
        str(dispersion_ledger),
        "--log-dir",
        str(ART / "dispersion_logs"),
        "--batch-size",
        "2",
        "--max-elapsed-delta-ms",
        "20",
    )
    step("5. 节点执行离散归因（仅耗时超预算）", code, out, err, "09_dispersion.json")

    summary["overall_ok"] = all(
        record["ok"]
        for record in summary["steps"]
        if record["step"] != "8a. 强制终止演练（无退出报告）"
    ) and crash_record["exit_code"] == 3 and bool(retry_rows)
    save("00_summary.json", json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print("重试证据任务数:", len(retry_rows))
    print("总体结论:", "PASS" if summary["overall_ok"] else "CHECK")
    return 0 if summary["overall_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
