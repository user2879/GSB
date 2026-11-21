"""强制终止验证用子进程：逐阶段提交账本，运行中阶段故意留下 running 迁移。"""

import argparse
import json
import sys

from .loader import load_bundle
from .repository import Ledger


def main(argv=None):
    parser = argparse.ArgumentParser(description="崩溃模拟子进程")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--crash-after-stage", required=True,
                        help="该阶段迁入 running 后立即以指定退出码退出，模拟强制终止")
    parser.add_argument("--exit-code", type=int, default=137)
    args = parser.parse_args(argv)

    plan, events = load_bundle(args.plan, args.events)
    ledger = Ledger(args.ledger)
    ledger.save_plan(plan, events)

    from .engine import LogicalClock, _emit
    from .graph import topological_order

    clock = LogicalClock()
    transitions = []
    cursor = None
    topo = topological_order(plan.stages)
    mapping = plan.stage_map
    for stage_id in topo:
        stage = mapping[stage_id]
        start_clock = clock.now_ms
        with ledger.conn:
            _emit(transitions, clock, stage_id, "pending", "running", "stage_start",
                  plan.version, start_clock)
            transition = transitions[-1]
            entry_digest = __import__("emergency_drill.models", fromlist=["digest_of"]).digest_of(
                {
                    "plan_version": transition["plan_version"],
                    "stage_id": transition["stage_id"],
                    "from_state": transition["from_state"],
                    "to_state": transition["to_state"],
                    "reason": transition["reason"],
                    "clock_ms": transition["clock_ms"],
                    "source_event_id": transition["source_event_id"],
                }
            )
            cur = ledger.conn.execute(
                "INSERT INTO transitions(plan_version, stage_id, from_state, to_state, reason, "
                "clock_ms, source_event_id, payload_digest) VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    transition["plan_version"], transition["stage_id"], transition["from_state"],
                    transition["to_state"], transition["reason"], transition["clock_ms"],
                    transition["source_event_id"], entry_digest,
                ),
            )
            ledger._set_head_digest(ledger._next_digest(
                {
                    "seq": cur.lastrowid,
                    "plan_version": transition["plan_version"],
                    "stage_id": transition["stage_id"],
                    "from_state": transition["from_state"],
                    "to_state": transition["to_state"],
                    "reason": transition["reason"],
                    "clock_ms": transition["clock_ms"],
                    "source_event_id": transition["source_event_id"],
                    "payload_digest": entry_digest,
                }
            ))
            ledger.conn.execute(
                "INSERT INTO progress(plan_version, cursor_stage, state, updated_seq) "
                "VALUES(?, ?, ?, ?) ON CONFLICT(plan_version) DO UPDATE SET "
                "cursor_stage=excluded.cursor_stage, state=excluded.state, updated_seq=excluded.updated_seq",
                (plan.version, stage_id, "running", cur.lastrowid),
            )
        print(json.dumps({"event": "committed_running", "stage_id": stage_id}), flush=True)
        if stage_id == args.crash_after_stage:
            print(json.dumps({"event": "simulated_sigkill", "stage_id": stage_id}), flush=True)
            sys.exit(args.exit_code)
        end_clock = start_clock + stage.duration_ms
        with ledger.conn:
            _emit(transitions, clock, stage_id, "running", "succeeded", "criterion_passed",
                  plan.version, end_clock)
            transition = transitions[-1]
            from .models import digest_of

            entry_digest = digest_of(
                {
                    "plan_version": transition["plan_version"],
                    "stage_id": transition["stage_id"],
                    "from_state": transition["from_state"],
                    "to_state": transition["to_state"],
                    "reason": transition["reason"],
                    "clock_ms": transition["clock_ms"],
                    "source_event_id": transition["source_event_id"],
                }
            )
            cur = ledger.conn.execute(
                "INSERT INTO transitions(plan_version, stage_id, from_state, to_state, reason, "
                "clock_ms, source_event_id, payload_digest) VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    transition["plan_version"], transition["stage_id"], transition["from_state"],
                    transition["to_state"], transition["reason"], transition["clock_ms"],
                    transition["source_event_id"], entry_digest,
                ),
            )
            ledger._set_head_digest(ledger._next_digest(
                {
                    "seq": cur.lastrowid,
                    "plan_version": transition["plan_version"],
                    "stage_id": transition["stage_id"],
                    "from_state": transition["from_state"],
                    "to_state": transition["to_state"],
                    "reason": transition["reason"],
                    "clock_ms": transition["clock_ms"],
                    "source_event_id": transition["source_event_id"],
                    "payload_digest": entry_digest,
                }
            ))
            ledger.conn.execute(
                "INSERT INTO progress(plan_version, cursor_stage, state, updated_seq) "
                "VALUES(?, ?, ?, ?) ON CONFLICT(plan_version) DO UPDATE SET "
                "cursor_stage=excluded.cursor_stage, state=excluded.state, updated_seq=excluded.updated_seq",
                (plan.version, stage_id, "succeeded", cur.lastrowid),
            )
        cursor = stage_id
    ledger.close()


if __name__ == "__main__":
    main()
