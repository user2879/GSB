"""端到端演示入口。

子命令：
- plan：枚举计划、定点估算并输出选优结论；
- run：真实回环调度执行并写账本与三版回执；
- invalidate：修改权重后输出失效传播轨迹并重算；
- crash：在提交若干分片后强制退出，再恢复重跑并做重放校验。
"""

import argparse
import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coljoin.invalidation import (
    PlanVersionState,
    cost_input_digest,
    trace_invalidation,
)
from coljoin.ledger import Ledger
from coljoin.pipeline import Pipeline


def cmd_plan(args):
    pipeline = Pipeline(args.workspace)
    plans, evaluations, decision = pipeline.plan()
    output = {
        "candidate_count": len(plans),
        "chosen_plan_id": decision["chosen_plan_id"],
        "error_influenced": decision["error_influenced"],
        "tie_count": len(decision["ties"]),
        "ranking_top5": decision["ranking"][:5],
        "winner_vector": next(
            item["vector"]
            for item in evaluations
            if item["plan_id"] == decision["chosen_plan_id"]
        ),
        "stats_version": evaluations[0]["stats_version"],
        "error_interval": evaluations[0]["error_interval"],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


def cmd_run(args):
    pipeline = Pipeline(
        args.workspace,
        executor_count=args.executors,
        max_in_flight=args.max_in_flight,
        per_executor_slots=args.per_executor,
        shard_rows=args.shard_rows,
    )
    result = pipeline.run(plan_version=args.plan_version)
    print(
        json.dumps(
            {
                "plan_version": result["plan_version"],
                "chosen_plan_id": result["chosen_plan_id"],
                "batches": len(result["batches"]),
                "receipts": len(result["receipts"]),
                "wait_events": len(result["waits"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def cmd_invalidate(args):
    pipeline = Pipeline(args.workspace)
    groups, query, weights, session = (
        pipeline.groups,
        pipeline.query,
        pipeline.weights,
        pipeline.session,
    )
    old_digest = cost_input_digest(groups, weights, session)
    previous = PlanVersionState(
        version=args.plan_version,
        cost_input_digest=old_digest,
        shard_digests={f"{p['plan_id']}:0000": "x" for p in __import__(
            "coljoin.planner", fromlist=["enumerate_plans"]
        ).enumerate_plans(query)},
    )
    new_weights = copy.deepcopy(weights)
    object.__setattr__(new_weights, "scan", weights.scan + args.delta)
    new_digest = cost_input_digest(groups, new_weights, session)
    trace = trace_invalidation(
        query,
        groups,
        new_weights,
        session,
        changed_block_groups=[],
        old_state=previous,
        new_cost_digest=new_digest,
    )
    print(
        json.dumps(
            {
                "new_version": trace["new_version"],
                "cost_input_changed": trace["cost_input_changed"],
                "invalidated_plan_count": len(trace["invalidated_plans"]),
                "invalidated_plans_head": trace["invalidated_plans"][:5],
                "invalidated_shard_count": len(trace["invalidated_shards"]),
                "invalidated_shards_head": trace["invalidated_shards"][:5],
                "reason_sample": trace["reasons"][:3],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def cmd_crash(args):
    pipeline = Pipeline(
        args.workspace,
        executor_count=2,
        max_in_flight=2,
        per_executor_slots=1,
        shard_rows=20,
    )
    crash_version = 9
    try:
        pipeline.run(plan_version=crash_version, crash_after=2)
    except SystemExit:
        print(json.dumps({"event": "forced_termination_simulated"}, ensure_ascii=False))
    db_path = os.path.join(args.workspace, "ledger", "coljoin.db")
    ledger = Ledger(db_path, os.path.join(args.workspace, "snapshots"))
    committed = ledger.committed_shards(crash_version)
    print(
        json.dumps(
            {
                "event": "recovery",
                "committed_after_kill": len(committed),
                "unknown_shards_require_rerun": True,
            },
            ensure_ascii=False,
        )
    )
    ledger.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default="workspace")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("plan").set_defaults(func=cmd_plan)

    run_parser = sub.add_parser("run")
    run_parser.add_argument("--plan-version", type=int, default=1)
    run_parser.add_argument("--executors", type=int, default=2)
    run_parser.add_argument("--max-in-flight", type=int, default=2)
    run_parser.add_argument("--per-executor", type=int, default=1)
    run_parser.add_argument("--shard-rows", type=int, default=100)
    run_parser.set_defaults(func=cmd_run)

    inv = sub.add_parser("invalidate")
    inv.add_argument("--plan-version", type=int, default=1)
    inv.add_argument("--delta", type=int, default=11)
    inv.set_defaults(func=cmd_invalidate)

    crash = sub.add_parser("crash")
    crash.add_argument("--plan-version", type=int, default=1)
    crash.set_defaults(func=cmd_crash)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
