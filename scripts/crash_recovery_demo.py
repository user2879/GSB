"""启动短生命周期子进程模拟强制终止，随后仅依据已提交清单恢复并核对。"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tactical_pathbench.encoding import write_text_file
from tactical_pathbench.persistence import PersistenceStore
from tactical_pathbench.workbench import PathfindingWorkbench


def python_exe() -> str:
    venv_win = ROOT / ".venv" / "Scripts" / "python.exe"
    venv_posix = ROOT / ".venv" / "bin" / "python"
    if venv_posix.exists():
        return str(venv_posix)
    if venv_win.exists():
        return str(venv_win)
    return sys.executable


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default="artifacts/persist-crash")
    parser.add_argument("--reports", default="artifacts/reports")
    parser.add_argument("--seed", type=int, default=20261119)
    args = parser.parse_args()

    workspace = ROOT / args.workspace
    reports = ROOT / args.reports
    worker = ROOT / "scripts" / "crash_worker.py"
    completed = subprocess.run(
        [python_exe(), str(worker), "--workspace", args.workspace, "--seed", str(args.seed)],
        capture_output=True,
        text=True,
    )
    forced = completed.returncode == 137

    store = PersistenceStore(workspace)
    recovery = store.recover()
    restored = PathfindingWorkbench.restore(store)
    routes = restored.solve_batch()

    # 与中断前已提交版本等价的独立工作台对照。
    from tactical_pathbench.workbench import build_workbench

    reference = build_workbench(seed=args.seed)
    reference.modify_terrain((0, 5), 7)
    reference_routes = reference.solve_batch()
    hash_equal = all(
        routes[key].path_hash() == reference_routes[key].path_hash() for key in routes
    )

    result = {
        "forced_termination_simulated": forced,
        "worker_returncode": completed.returncode,
        "recovery": recovery.to_payload(),
        "restored_map_version": restored.map_version,
        "path_hash_equal_after_recovery": hash_equal,
        "path_hashes": {key: route.path_hash() for key, route in sorted(routes.items())},
    }
    write_text_file(
        reports / "crash-recovery-report.json",
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if forced and recovery.recovered and hash_equal else 1


if __name__ == "__main__":
    raise SystemExit(main())
