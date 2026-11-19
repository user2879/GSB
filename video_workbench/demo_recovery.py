"""强制终止与恢复判定的端到端验证（真实子进程 + 操作系统信号）。

场景
----
1. hard_kill：子进程在 denoise_003 产物落盘后立即 os._exit(137)，
   模拟无法处理的 SIGKILL/断电；重启仅依据已完整替换的产物恢复。
2. SIGTERM：向运行中的子进程发送可处理退出信号，进程在阶段边界
   完成当前产物落盘并以退出码 0 退出，不生成额外退出报告。
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKDIR = ROOT / "runs" / "recovery"
CONFIG = ROOT / "config" / "default_config.json"


def run_cli(*extra: str, wait: bool = True) -> subprocess.Popen | None:
    cmd = [
        sys.executable,
        "-m",
        "vbench.cli",
        "run",
        "--config",
        str(CONFIG),
        "--workdir",
        str(WORKDIR),
        "--report",
        str(ROOT / "runs" / "recovery_report.json"),
        *extra,
    ]
    proc = subprocess.Popen(
        cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if wait:
        stdout, stderr = proc.communicate()
        return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)
    return proc


def snapshot_manifest() -> dict:
    manifest = WORKDIR / "snapshots" / "current" / "manifest.json"
    return json.loads(manifest.read_text())


def scenario_hard_kill() -> dict:
    shutil.rmtree(WORKDIR, ignore_errors=True)
    proc = run_cli("--injections", str(ROOT / "config" / "injections_hard_kill.json"))
    assert proc.returncode == 137, f"硬终止退出码应为 137，实际 {proc.returncode}"
    orphans = [
        name
        for name in os.listdir(WORKDIR / "snapshots")
        if name.startswith("tmp-")
    ]
    manifest = snapshot_manifest()
    killed_at = "denoise_003"
    assert killed_at in manifest["completed_stages"]
    assert manifest["step_cursor"] == 3
    # 重启恢复：未完成阶段续算，已完成阶段全部复用。
    recover = run_cli()
    report = json.loads(recover.stdout)
    assert report["status"] == "ok"
    assert "denoise_003" in report["reused"]
    assert report["stats"]["recomputed_count"] == report["stats"]["stage_total"] - len(
        manifest["completed_stages"]
    )
    return {
        "scenario": "hard_kill",
        "killed_exit_code": 137,
        "orphan_staging_left": sorted(orphans),
        "recovery_point": report["recovery_point"],
        "resume_reused_count": report["stats"]["reused_count"],
        "resume_recomputed_count": report["stats"]["recomputed_count"],
    }


def scenario_sigterm() -> dict:
    shutil.rmtree(WORKDIR, ignore_errors=True)
    proc = run_cli(wait=False)
    # 等待信号处理器安装且首个阶段快照提交，再发可处理退出信号，
    # 进程会在阶段边界完成当前产物落盘并以退出码 0 退出。
    deadline = time.time() + 10
    snapshot_dir = WORKDIR / "snapshots"
    while time.time() < deadline:
        if (snapshot_dir / "current" / "manifest.json").exists():
            break
        if proc.poll() is not None:
            raise RuntimeError(f"进程提前退出: {proc.returncode}")
        time.sleep(0.01)
    proc.send_signal(signal.SIGTERM)
    proc.wait(timeout=10)
    manifest = snapshot_manifest()
    return {
        "scenario": "sigterm_graceful",
        "exit_code": proc.returncode,
        "committed": manifest["committed"],
        "completed_stages": manifest["completed_stages"],
        "step_cursor": manifest["step_cursor"],
        "frame_cursor": manifest["frame_cursor"],
        "note": "阶段边界完成当前产物落盘后退出，未承诺额外退出报告",
    }


def main() -> None:
    results = [scenario_hard_kill(), scenario_sigterm()]
    out = ROOT / "runs" / "recovery_demo.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True))
    print(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
