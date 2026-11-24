"""统一命令行入口（无网络、无账号、无 GPU）。

子命令：
  validate   强校验声明与事件流
  gen-samples 生成确定性小/中两档样本
  plan       输出路径选择理由、规模参数与工作量估算
  replay     逻辑时钟离线重放并落账本
  verify     独立校验三路径逐字节一致 + 账本重放比对 + 槽位唯一
  live       启动真实回环子进程服务，线程池实时仲裁并落账本
  diagnose   整数定点配额预算诊断与误差界
  recover    读取账本恢复活动租约/槽位，标记未提交游标
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .errors import FIError
from .fixedpoint import diagnose, global_budget, rank_ratios
from .ledger import Ledger
from .planner import choose_path, estimate_work, scale_parameters
from .replay import replay_path, verify_ledger, verify_paths, verify_slot_uniqueness
from .samples import generate_all
from .spec import load_bundle


def _print(payload: dict | list) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))


def _bundle(args: argparse.Namespace):
    if not args.spec:
        raise SystemExit("[E_SPEC] 该子命令需要 --spec 声明文件")
    return load_bundle(args.spec, args.session)


def cmd_validate(args: argparse.Namespace) -> int:
    bundle = _bundle(args)
    _print({
        "ok": True,
        "domains": len(bundle.domains),
        "instances": len(bundle.instances),
        "events": len(bundle.events),
        "first_event_seq": bundle.events[0].seq,
        "last_event_seq": bundle.events[-1].seq,
        "monotonic_arrival": all(
            bundle.events[i].arrival_ms <= bundle.events[i + 1].arrival_ms
            for i in range(len(bundle.events) - 1)
        ),
    })
    return 0


def cmd_gen_samples(args: argparse.Namespace) -> int:
    paths = generate_all(args.out)
    _print({name: {k: str(v) for k, v in files.items()} for name, files in paths.items()})
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    bundle = _bundle(args)
    _print(choose_path(bundle))
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    bundle = _bundle(args)
    outcomes = replay_path(bundle, args.path)
    ledger = Ledger(args.ledger, replace=args.replace)
    from .clock import LogicalClock
    from .engine import Engine
    engine = Engine(bundle, LogicalClock(0))
    engine.run_event_path()
    batch_size = bundle.session.batch_size
    for batch_id, start in enumerate(range(0, len(engine.outcomes), batch_size), start=1):
        chunk = engine.outcomes[start:start + batch_size]
        tr_start = sum(1 for _ in engine.transitions[:0])
        ledger.append_batch(
            batch_id=batch_id, outcomes=chunk,
            transitions=[t for t in engine.transitions
                         if any(o.lease_id == t.lease_id for o in chunk)],
            waits=[w for w in engine.waits if any(o.seq == w.seq for o in chunk)],
            slot_snapshot={d: dict(a.slot_owner) for d, a in engine.arbiters.items() if a.slot_owner},
            clock_ms=max(o.clock_ms for o in chunk),
        )
    digest = ledger.current_digest()
    ledger.close()
    _print({"path": args.path, "events": len(outcomes), "ledger": str(args.ledger),
            "digest": digest})
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    bundle = _bundle(args)
    paths = verify_paths(bundle)
    slots = verify_slot_uniqueness(bundle)
    result = {"path_byte_consistency": paths, "slot_uniqueness": slots}
    if args.ledger and Path(args.ledger).exists():
        result["ledger"] = verify_ledger(bundle, args.ledger)
    _print(result)
    return 0


def cmd_live(args: argparse.Namespace) -> int:
    from .process import ArbiterProcessGroup, LiveScheduler
    bundle = _bundle(args)
    ledger = Ledger(args.ledger, replace=args.replace)
    group = ArbiterProcessGroup(bundle)
    group.start()
    try:
        scheduler = LiveScheduler(bundle, ledger, path=args.path)
        result = scheduler.run()
    finally:
        group.stop()
        ledger.close()
    _print(result)
    return 0


def cmd_diagnose(args: argparse.Namespace) -> int:
    bundle = _bundle(args)
    power = bundle.session.scale_power
    from .clock import LogicalClock
    from .engine import Engine
    from .budget import compute_usage
    engine = Engine(bundle, LogicalClock(0))
    usage, global_usage = compute_usage(engine)  # 内部逐事件推进并采样峰值
    items, peak_items, per_domain = [], [], []
    for snap in usage:
        items.append((snap.domain_id, snap.current_used, snap.slots))
        peak_items.append((snap.domain_id, snap.peak_used, snap.slots))
        per_domain.append({
            "domain_id": snap.domain_id,
            "slots": snap.slots,
            "current": diagnose(snap.domain_id, snap.current_used, snap.slots, power).as_dict(),
            "peak": diagnose(snap.domain_id, snap.peak_used, snap.slots, power).as_dict(),
        })
    ranked = rank_ratios(items, power)
    peak_ranked = rank_ratios(peak_items, power)
    _print({
        "scale_power": power,
        "scale_factor": 10 ** power,
        "truncation": "toward_zero (integer floor)",
        "domains": per_domain,
        "global_current": global_budget(global_usage.current_used, global_usage.total_slots, power).as_dict(),
        "global_peak": global_budget(global_usage.peak_used, global_usage.total_slots, power).as_dict(),
        "ranking_current": [r.__dict__ for r in ranked],
        "ranking_peak": [r.__dict__ for r in peak_ranked],
        "error_bounds": {
            "single_quantization": f"1/{10 ** power}",
            "comparison": f"2/{10 ** power}",
        },
    })
    return 0


def cmd_recover(args: argparse.Namespace) -> int:
    bundle = _bundle(args)
    ledger = Ledger(args.ledger)
    bad = ledger.replace_if_digest_bad()
    state = ledger.recover()
    ledger.close()
    _print({"replaced": state.replaced or bad, **state.__dict__})
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="faultisolation", description="跨故障域租约仲裁与配额预算诊断工作台")
    parser.add_argument("--spec", required=False, default=None)
    parser.add_argument("--session", required=False, default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("validate").set_defaults(func=cmd_validate)
    gen = sub.add_parser("gen-samples")
    gen.add_argument("--out", default="samples")
    gen.set_defaults(func=cmd_gen_samples)
    sub.add_parser("plan").set_defaults(func=cmd_plan)

    replay = sub.add_parser("replay")
    replay.add_argument("--path", choices=["event", "batch", "merge"], default="event")
    replay.add_argument("--ledger", required=True)
    replay.add_argument("--replace", action="store_true")
    replay.set_defaults(func=cmd_replay)

    verify = sub.add_parser("verify")
    verify.add_argument("--ledger", default=None)
    verify.set_defaults(func=cmd_verify)

    live = sub.add_parser("live")
    live.add_argument("--path", choices=["event", "batch", "merge"], default=None)
    live.add_argument("--ledger", required=True)
    live.add_argument("--replace", action="store_true")
    live.set_defaults(func=cmd_live)

    sub.add_parser("diagnose").set_defaults(func=cmd_diagnose)
    recover = sub.add_parser("recover")
    recover.add_argument("--ledger", required=True)
    recover.set_defaults(func=cmd_recover)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except FIError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"[E_RUNTIME] {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
