"""演示脚本：冷启动填充 -> 版本演进派生失效 -> 快照保存与恢复重放。

用法：
    python run_demo.py                    # 顺序执行三个场景
    python run_demo.py --inject cycle     # 注入派生成环   -> E_GRAPH
    python run_demo.py --inject truncate  # 注入快照截断   -> E_SNAPSHOT
    python run_demo.py --inject digest    # 注入摘要篡改   -> E_VERSION
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from . import inject  # noqa: E402
from .errors import DiagError  # noqa: E402
from .logx import JsonlLogger  # noqa: E402
from .orchestrator import Orchestrator  # noqa: E402
from .report import build_report  # noqa: E402
from .spec import load_policy, load_requests  # noqa: E402

EXIT_CODES = {
    "E_SPEC": 10, "E_PORT": 11, "E_HANDSHAKE": 12, "E_PROTO": 13,
    "E_LEASE": 14, "E_GRAPH": 15, "E_SNAPSHOT": 16, "E_VERSION": 17,
    "E_REPLAY": 18,
}

EVOLVE_TICK_MS = 100000


def workspace_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def clean_workspace(ws):
    """演示前清理上一次生成的账本与快照，冷启动从空状态开始。"""
    for sub in ("ledger", "logs"):
        d = os.path.join(ws, sub)
        os.makedirs(d, exist_ok=True)
        for name in os.listdir(d):
            path = os.path.join(d, name)
            try:
                os.chmod(path, 0o666)  # 旧快照为只读，先恢复可写
            except OSError:
                pass
            os.remove(path)


def scenario_cold_start(orch, batches):
    print("===== 场景一：冷启动填充 =====")
    outcomes = orch.run_batches(batches)
    for o in outcomes:
        print(
            f"[batch {o['batch']}] {o['node']} {o['path']} -> {o['result']} "
            f"source={o['source']} etag={o['etag']} ttl_left={o['ttl_remaining_ms']}ms "
            f"evictions={len(o['evictions'])}"
        )
    return outcomes


def scenario_evolution(orch):
    print("===== 场景二：内容版本演进触发派生失效 =====")
    trace = orch.evolve_content(
        "/style.css", "v2-style", b"STYLE-V2", tick_ms=EVOLVE_TICK_MS
    )
    for t in trace:
        print(
            f"trigger={t['trigger_path']} affected={t['affected_path']} "
            f"node={t['node']} present={t['present']} "
            f"entry_version={t['old_entry_version']}->{t['new_entry_version']}"
        )
    return trace


def scenario_snapshot_replay(orch):
    print("===== 场景三：快照保存与恢复重放 =====")
    orch.save_snapshot()
    print("快照已保存:", orch.snapshot_path)
    orch.kill_nodes()
    print("全部节点已强制终止（模拟崩溃）")
    diff, replaced = orch.restore()
    print(f"恢复完成 full_replace={replaced} diff={diff}")
    orch.restart_nodes(restore_entries=True)
    print("节点已依据账本持久化条目重启恢复")
    # 重放从冷态重放完整账本历史
    orch.kill_nodes()
    orch.restart_nodes(restore_entries=False)
    verdict = orch.replay()
    print(f"重放校验: verdict={verdict['verdict']} compared_fields={verdict['compared_fields']}")
    return verdict


def run_injection(orch, kind, logger):
    if kind == "cycle":
        orch.evolve_content(
            "/index.html",
            "v2-index",
            b"INDEX-V2",
            tick_ms=EVOLVE_TICK_MS,
            derived_from=inject.cycle_derivation_override(),
        )
    elif kind == "truncate":
        orch.save_snapshot()
        info = inject.truncate_snapshot(orch.snapshot_path)
        logger.log("inject_truncate", **info)
        orch.restore()
    elif kind == "digest":
        orch.save_snapshot()
        info = inject.tamper_snapshot_digest(orch.snapshot_path)
        logger.log("inject_digest", **info)
        orch.restore()
    raise AssertionError(f"注入 {kind} 未触发预期错误")


def main(argv=None):
    parser = argparse.ArgumentParser(description="CDN 边缘缓存一致性诊断台演示")
    parser.add_argument("--inject", choices=["cycle", "truncate", "digest"])
    args = parser.parse_args(argv)

    ws = workspace_root()
    clean_workspace(ws)
    logger = JsonlLogger(os.path.join(ws, "logs", "demo.jsonl"))
    policy = load_policy(os.path.join(ws, "samples", "policy.json"))
    batches = load_requests(os.path.join(ws, "samples", "requests.json"))
    orch = Orchestrator(ws, policy, logger)
    try:
        orch.start()
        print(f"内容源与 {len(orch.nodes)} 个边缘节点已启动并完成握手")
        if args.inject:
            run_injection(orch, args.inject, logger)
        else:
            scenario_cold_start(orch, batches)
            scenario_evolution(orch)
            scenario_snapshot_replay(orch)
            print()
            print(build_report(orch.ledger))
        logger.log("demo_finished", result="ok")
        return 0
    except DiagError as exc:
        logger.log("demo_error", **exc.to_dict())
        print(json.dumps(exc.to_dict(), ensure_ascii=False, sort_keys=True))
        return EXIT_CODES.get(exc.code, 1)
    finally:
        orch.stop()


if __name__ == "__main__":
    raise SystemExit(main())
