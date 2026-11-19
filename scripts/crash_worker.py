"""子进程工作者：提交若干版本后强制终止（os._exit），不写退出报告。"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tactical_pathbench.budget import Budget
from tactical_pathbench.persistence import PersistenceStore
from tactical_pathbench.workbench import build_workbench


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--seed", type=int, default=20261119)
    args = parser.parse_args()

    store = PersistenceStore(ROOT / args.workspace)
    wb = build_workbench(seed=args.seed, budget=Budget(thread_budget=2, memory_budget=8))
    wb.solve_batch()
    wb.persist(store)

    wb.modify_terrain((0, 5), 7)
    wb.solve_batch()
    wb.persist(store)

    # 留下未提交缓存与暂态文件，模拟强制终止。
    store.write_loose_cache("uncommitted-orphan.json", {"map_version": wb.map_version + 1})
    store.tmp_path.write_text("partial-commit", encoding="utf-8")
    sys.stdout.flush()
    os._exit(137)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
