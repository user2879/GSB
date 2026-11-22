"""命令行入口：诊断运行、快照恢复与三类异常注入。

注入仅通过修改输入文件/参数实现，不改源码：
  --inject cycle     : 复制拓扑并追加一条成环边      -> E_CYCLE
  --inject capacity  : 复制拓扑并放大容量上限         -> E_CAPACITY
  --inject snapshot  : 打开账本后改写内容摘要行       -> E_SNAPSHOT
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

from .errors import DegPropError
from .logging_setup import configure_logging


def _inject_cycle(src: Path, dst: Path) -> Path:
    doc = json.loads(src.read_text(encoding="utf-8"))
    first = doc["services"][0]["service_id"]
    target = doc["services"][1]["service_id"]
    doc["edges"].append({
        "upstream": target,
        "downstream": first,
        "propagation_mode": "forced",
    })
    dst.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    return dst


def _inject_capacity(src: Path, dst: Path) -> Path:
    doc = json.loads(src.read_text(encoding="utf-8"))
    doc["services"][0]["capacity_limit"] = 2**62
    dst.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    return dst


def _inject_snapshot(ledger_path: Path) -> None:
    conn = sqlite3.connect(ledger_path)
    try:
        conn.execute(
            "UPDATE meta SET value=? WHERE key='topology_digest'",
            ("tampered-digest-0000",))
        conn.commit()
    finally:
        conn.close()


def _inject_replay(ledger_path: Path) -> None:
    """篡改一条已持久化结局，使恢复重放在首个不一致处以 E_REPLAY 报明。"""
    conn = sqlite3.connect(ledger_path)
    try:
        row = conn.execute(
            "SELECT batch_id, seq FROM call_outcomes ORDER BY seq LIMIT 1").fetchone()
        conn.execute(
            "UPDATE call_outcomes SET outcome=? WHERE batch_id=? AND seq=?",
            ("tampered_outcome", row[0], row[1]))
        conn.commit()
    finally:
        conn.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="degprop",
        description="服务降级传播与故障域收敛诊断组件")
    parser.add_argument("--topology", default="samples/topology.json")
    parser.add_argument("--events", default="samples/events.ndjson")
    parser.add_argument("--ledger", default="artifacts/session.db")
    parser.add_argument("--workdir", default="artifacts")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--inject", choices=["cycle", "capacity", "snapshot", "replay"],
                        default=None)
    parser.add_argument("--mode",
                        choices=["run", "resume", "validate-only"],
                        default="run")
    parser.add_argument("--gui", action="store_true",
                        help="可选：弹出 Tkinter 汇总窗口（无显示环境自动跳过）")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logger = configure_logging(args.log_level, args.log_file)
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    topology_path = Path(args.topology)
    try:
        if args.inject == "cycle":
            topology_path = _inject_cycle(
                topology_path, workdir / "injected-cycle.json")
            from .loader import load_topology

            load_topology(topology_path)  # 预期 E_CYCLE
        elif args.inject == "capacity":
            topology_path = _inject_capacity(
                topology_path, workdir / "injected-capacity.json")
            from .loader import load_topology

            load_topology(topology_path)  # 预期 E_CAPACITY

        # 延迟导入，避免无 Tk 的 CI 在导入期失败。
        from .engine import DiagnosticsEngine

        if args.mode == "validate-only":
            from .loader import load_events, load_topology

            topo = load_topology(topology_path)
            events = load_events(args.events, topo)
            print(json.dumps({"code": "E_OK", "services": len(topo.services),
                              "events": len(events),
                              "topological_order": topo.order_index},
                             ensure_ascii=False, indent=2))
            return 0

        if args.inject == "snapshot":
            # 先冷启动写一版账本，再篡改摘要，随后重新打开触发整体替换。
            with DiagnosticsEngine(topology_path, args.events,
                                   args.ledger) as engine:
                engine.run_batch(engine.events[:args.batch_size], 1)
            _inject_snapshot(Path(args.ledger))
        if args.inject == "replay":
            # 先冷启动写账本，再篡改一条结局；resume 重放须报 E_REPLAY。
            with DiagnosticsEngine(topology_path, args.events,
                                   args.ledger) as engine:
                engine.run_batch(engine.events[:args.batch_size], 1)
            _inject_replay(Path(args.ledger))
            args.mode = "resume"

        with DiagnosticsEngine(topology_path, args.events,
                               args.ledger) as engine:
            if args.mode == "resume":
                result = engine.resume_and_replay()
                print(json.dumps({"code": "E_OK", "phase": "resume",
                                  **result}, ensure_ascii=False, indent=2))
            else:
                reports = []
                events = engine.events
                for batch_id, start in enumerate(
                        range(0, len(events), args.batch_size), start=1):
                    batch = events[start:start + args.batch_size]
                    report = engine.run_batch(batch, batch_id)
                    reports.append(_summarize(report))
                payload = {"code": "E_OK", "phase": "run",
                           "replacement": engine.replacement,
                           "batches": reports}
                print(json.dumps(payload, ensure_ascii=False, indent=2))
                if args.gui:
                    _maybe_gui(payload)
        return 0
    except DegPropError as exc:
        safe = {f"biz_{k}": v for k, v in exc.as_dict().items()}
        logger.error("business_error", extra=safe)
        print(json.dumps(exc.as_dict(), ensure_ascii=False, indent=2))
        return 2


def _summarize(report) -> dict:
    outcome_counts: dict[str, int] = {}
    for record in report.records:
        outcome_counts[record.outcome] = outcome_counts.get(record.outcome, 0) + 1
    return {
        "batch_id": report.batch_id,
        "records": report.record_count,
        "outcomes": outcome_counts,
        "degraded_services": report.propagation.degraded_services,
        "domain_plans": {
            did: {
                "degraded_services": plan.degraded_services,
                "total_capacity": plan.total_capacity,
                "source_plan": plan.source_plan,
                "indistinguishable": plan.indistinguishable,
                "availability_rate_fixed": plan.availability_rate_fixed,
            }
            for did, plan in report.propagation.domain_plans.items()
        },
        "digest": report.digest,
    }


def _maybe_gui(payload: dict) -> None:
    try:
        from .viewer import show_summary
    except Exception as exc:  # 无 Tk 环境（如无显示的 Linux）自动降级
        print(f"[gui] 当前环境不可用，已跳过窗口: {exc}")
        return
    show_summary(payload)


if __name__ == "__main__":
    sys.exit(main())
