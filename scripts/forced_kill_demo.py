"""强制终止恢复演示。

流程：
1. 用真实子进程跑完整调度并把已完成分片原子写入 SQLite；
2. 对一个节点子进程执行 SIGKILL（hard kill），不生成退出报告；
3. 重新打开账本，把在飞分片标记为 UNKNOWN，登记“先回收额度再重跑”；
4. 从空白全量重放，与已完整提交记录逐项比对（E_REPLAY 定位首个不一致）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dispatch_tokens.ledger import UNKNOWN, Ledger  # noqa: E402
from dispatch_tokens.runner import Runner, rebuild_from_scratch  # noqa: E402
from dispatch_tokens.spec import load_spec  # noqa: E402
from dispatch_tokens.supervisor import Supervisor  # noqa: E402
from dispatch_tokens.engine import SchedulerEngine  # noqa: E402


def main() -> int:
    spec = load_spec(ROOT / "samples" / "demo_spec.json")
    db_path = ROOT / "artifacts" / "demo.kill.db"
    runner = Runner(spec, db_path, with_workers=True)
    report = runner.run()

    # 再起一组真实子进程并立即 SIGKILL，模拟强制终止。
    supervisor = Supervisor(list(spec.nodes),
                            SchedulerEngine(spec).current_scale)
    supervisor.start_all()
    killed_node = spec.nodes[0].node_id
    exit_code = supervisor.hard_kill(killed_node)
    print(f"[kill] node={killed_node} sigkill exit={exit_code}; "
          "no exit report generated")

    # 恢复：在飞分片（取一个真实节点上的分片）标记 UNKNOWN。
    in_flight = next(
        sid for sid, item in report["outcomes"].items()
        if item["node_id"] == killed_node and item["status"] == "DISPATCHED")
    ledger = Ledger(db_path, spec)
    node = spec.node(killed_node)
    ledger.mark_unknown(
        in_flight, killed_node,
        cursor_seq=max(spec.events, key=lambda e: e.seq).seq,
        updated_ms=max(e.ts_ms for e in spec.events),
        bucket={"balance": 0, "remainder": 0, "rate": node.rate,
                "burst": node.burst, "scale": SchedulerEngine(spec).current_scale,
                "last_ms": max(e.ts_ms for e in spec.events),
                "param_version": node.param_version,
                "bucket_key": node.bucket_key,
                "node_id": killed_node})
    stored = ledger.load_outcomes()
    assert stored[in_flight]["status"] == UNKNOWN
    assert stored[in_flight]["ownership"]["decision"] == \
        "UNKNOWN_RERUN_RECLAIM_FIRST"
    print(f"[recover] in-flight shard={in_flight} marked UNKNOWN, "
          "quota reclaim scheduled before rerun")

    replay = rebuild_from_scratch(spec)
    ledger.verify_replay(replay)
    print("[replay] committed records match from-scratch replay "
          "(E_REPLAY-free)")
    ledger.close()

    summary = {
        "killed_node": killed_node,
        "kill_exit_code": exit_code,
        "unknown_shard": in_flight,
        "reclaim_first": [in_flight],
        "replay_verified": True,
        "committed_shards": sorted(
            sid for sid, item in stored.items()
            if item["status"] != UNKNOWN),
    }
    out = ROOT / "artifacts" / "demo.kill.recovery.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"[output] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
