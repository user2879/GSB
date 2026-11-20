"""一键端到端演示：两轮路由（含规则演进）、重放、增量比较、异常注入与恢复。

运行：``python -m shardbench.demo``（在仓库根目录执行，数据写入 shardbench/data）。
"""

from __future__ import annotations

import json
import os
import shutil

from .cli import _default_paths
from .engine import Engine
from .injection import (
    trigger_config_injection,
    trigger_ring_injection,
    trigger_version_injection,
)
from .ledger import Ledger
from .logging_utils import StructuredLogger
from .replay import replay_round


def main() -> int:
    workspace = os.getcwd()
    paths = _default_paths(workspace)
    shutil.rmtree(os.path.dirname(paths["ledger"]), ignore_errors=True)

    with StructuredLogger(paths["log"]) as logger:
        engine = Engine(
            paths["config"],
            paths["units"],
            paths["ledger"],
            paths["snapshot"],
            logger=logger,
        )
        first = engine.run_round()
        print(f"[1] 首轮提交 round_id={first.round_id}，单元数={len(first.plans)}")
        print(
            "    调度核验 global_ok="
            f"{first.scheduler_verification['global_ok']} "
            f"per_db_ok={first.scheduler_verification['per_db_ok']}"
        )

        replay = replay_round(paths["ledger"], first.round_id)
        print(f"[2] 重放校验 status={replay['status']} 比对单元={replay['units_checked']}")

        # 模拟一次强制终止：begin 后不提交。
        orphan = engine.ledger.begin_round()
        ledger_check = Ledger(paths["ledger"])
        unknown = ledger_check.unknown_rounds()
        ledger_check.mark_unknown_for_rerun(unknown)
        ledger_check.close()
        print(f"[3] 模拟强制终止，未提交轮次 {orphan} 标记 requires_rerun={unknown}")

        engine.close()

        config_payload = json.load(open(paths["config"], encoding="utf-8"))
        units_payload = json.load(open(paths["units"], encoding="utf-8"))
        print(
            "[4] 注入 E_CONFIG ->",
            trigger_config_injection(config_payload, units_payload).code,
        )
        print(
            "[5] 注入 E_RING   ->",
            trigger_ring_injection(config_payload, units_payload).code,
        )
        print(
            "[6] 注入 E_VERSION->",
            trigger_version_injection(paths["snapshot"]).code,
        )
    print("[7] 演示完成，账本/快照/日志位于", os.path.dirname(paths["ledger"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
