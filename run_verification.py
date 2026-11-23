"""端到端验证脚本：一次跑完全部验收项并打印结论。"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.abspath(__file__))


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)


def main() -> int:
    print("== 1. 生成固定种子样本 ==")
    r = run([sys.executable, "-m", "reward_workbench", "init", "--root", ROOT])
    print(r.stdout.strip())

    print("\n== 2. 一次性全量运行（含预算/待办） ==")
    r = run([sys.executable, "-m", "reward_workbench", "run"])
    print(r.stdout.strip())
    assert r.returncode == 0

    print("\n== 3. 命令流回放：增量 vs 全量重建逐字段一致 ==")
    r = run([sys.executable, "-m", "reward_workbench", "replay"])
    print(r.stdout.strip().splitlines()[0])
    assert r.returncode == 0, "增量/全量不一致"

    print("\n== 4. 并发 vs 串行逐字段一致 ==")
    r = run([sys.executable, "-m", "reward_workbench", "verify"])
    print(r.stdout.strip())
    assert r.returncode == 0

    print("\n== 5. 异常注入隔离与业务码 ==")
    r = run([sys.executable, "-m", "reward_workbench", "inject"])
    print(r.stdout.strip())
    assert r.returncode == 0

    print("\n== 6. unittest 全量测试 ==")
    r = run([sys.executable, "-m", "unittest", "discover", "-s", "tests"])
    summary = (r.stderr or r.stdout).strip().splitlines()
    print(summary[-1] if summary else "no output")
    assert r.returncode == 0

    diag = json.load(open(os.path.join(ROOT, "reports", "diagnostics.json"),
                          encoding="utf-8"))
    print("\n== 验收产物 ==")
    print("  reports/diagnostics.json 字段:", sorted(diag.keys()))
    print("  等价结论:", diag["equivalence"])
    print("\n全部验收项通过 ✔")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
