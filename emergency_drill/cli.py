"""命令行入口：数据处理管线、验证工具与异常注入链路。"""

import argparse
import json
import os
import subprocess
import sys

from .errors import DrillError
from .loader import load_bundle, load_events, load_plan
from .logging_setup import configure_logging, log_business_error
from .repository import Ledger
from .sample_data import generate_samples
from .session import DrillSession, tamper_snapshot_digest
from .viewmodel import render_text


def _write_json(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)


def _read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def cmd_generate(args):
    names = generate_samples(args.sample_dir)
    print(json.dumps({"generated": names, "dir": os.path.abspath(args.sample_dir)},
                     ensure_ascii=False, indent=2))


def cmd_validate(args):
    if args.inject == "corrupt-plan":
        plan = load_plan(args.plan)
    elif args.inject == "cycle":
        plan = load_plan(args.plan)
    else:
        plan = load_plan(args.plan)
    events = load_events(args.events, plan)
    print(json.dumps({
        "accepted": True,
        "plan_id": plan.plan_id,
        "stage_count": len(plan.stages),
        "event_count": len(events.events),
        "plan_digest": plan.digest(),
        "event_digest": events.digest(),
    }, ensure_ascii=False, indent=2))


def cmd_run(args):
    plan, events = load_bundle(args.plan, args.events)
    ledger = Ledger(args.ledger)
    session = DrillSession(ledger)
    outcome = session.start_round(plan, events)
    result = outcome["simulation"]
    payload = {
        "version": outcome["version"],
        "stats": result.stats,
        "outcomes": [result.outcomes[sid] for sid in plan.stage_ids_topo()],
        "event_conclusions": result.event_conclusions,
        "ledger_digest": ledger.ledger_digest(),
    }
    _write_json(args.output, payload)
    print(json.dumps({"ok": True, "version": outcome["version"], "output": args.output,
                      "stats": result.stats}, ensure_ascii=False))
    ledger.close()


def cmd_edit(args):
    ledger = Ledger(args.ledger)
    version = args.version or ledger.latest_version()
    plan, events = ledger.load_plan_events(version)
    changeset = _read_json(args.changeset)
    session = DrillSession(ledger)
    outcome = session.edit_and_recompute(plan, events, changeset)
    topo = outcome["plan"].stage_ids_topo()
    payload = {
        "new_version": outcome["version"],
        "invalid_stage_ids": outcome["invalid_stage_ids"],
        "frozen_stage_ids": outcome["frozen_stage_ids"],
        "propagation": outcome["propagation"],
        "deleted_stage_ids": outcome["deleted_stage_ids"],
        "verification": outcome["verification"],
        "outcomes": [outcome["simulation"].outcomes[sid] for sid in topo],
    }
    _write_json(args.output, payload)
    print(json.dumps({"ok": True, "new_version": outcome["version"],
                      "invalid": outcome["invalid_stage_ids"],
                      "frozen": outcome["frozen_stage_ids"],
                      "verification": outcome["verification"]}, ensure_ascii=False))
    ledger.close()


def cmd_verify_incr(args):
    ledger = Ledger(args.ledger)
    version = args.version or ledger.latest_version()
    plan, events = ledger.load_plan_events(version)
    from .engine import simulate, verify_recompute

    stored = {payload["stage_id"]: payload for payload in ledger.active_outcomes(version)}
    result = simulate(plan, events)
    frozen_ids = [sid for sid in plan.stage_ids_topo() if sid in stored
                  and stored[sid]["plan_version"] < version]
    report = verify_recompute(plan, events, result, frozen_ids)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    ledger.close()


def cmd_new_round(args):
    plan, events = load_bundle(args.plan, args.events)
    ledger = Ledger(args.ledger)
    session = DrillSession(ledger)
    outcome = session.new_round(plan, events)
    print(json.dumps({"ok": True, "version": outcome["version"],
                      "replaced": outcome["replaced"],
                      "stats": outcome["simulation"].stats}, ensure_ascii=False, indent=2))
    ledger.close()


def cmd_snapshot(args):
    ledger = Ledger(args.ledger)
    version = args.version or ledger.latest_version()
    session = DrillSession(ledger)
    snapshot = session.create_snapshot(version, args.output)
    print(json.dumps({"ok": True, "snapshot": args.output, "version": version,
                      "content_digest": snapshot["content_digest"]}, ensure_ascii=False))
    ledger.close()


def cmd_open_snapshot(args):
    ledger = Ledger(args.ledger)
    version = args.version or ledger.latest_version()
    session = DrillSession(ledger)
    snapshot, diff = session.open_snapshot(version, args.snapshot)
    print(json.dumps({"ok": not diff["needs_full_replace"], "diff": diff},
                     ensure_ascii=False, indent=2))
    ledger.close()


def cmd_replay(args):
    ledger = Ledger(args.ledger)
    version = args.version or ledger.latest_version()
    session = DrillSession(ledger)
    report = session.replay_snapshot(version, args.snapshot)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    ledger.close()


def cmd_tamper_snapshot(args):
    output = args.output or args.snapshot + ".tampered.json"
    tamper_snapshot_digest(args.snapshot, output)
    print(json.dumps({"ok": True, "output": output}, ensure_ascii=False))


def cmd_recover(args):
    ledger = Ledger(args.ledger)
    version = args.version or ledger.latest_version()
    report = ledger.recover(version)
    unknown = report["unknown_stages"]
    if not unknown:
        unknown = [str(row["stage_id"]) for row in ledger.conn.execute(
            "SELECT stage_id FROM outcomes WHERE plan_version=? AND unknown=1", (version,)
        ).fetchall()]
    outcome_rows = {payload["stage_id"]: payload for payload in ledger.active_outcomes(version)}
    payload = {
        "version": version,
        "recovered_transition_count": report["recovered_transition_count"],
        "unknown_stages": unknown,
        "can_build_exit_report": report["can_build_exit_report"],
        "stats": ledger.stats(version),
        "outcomes": [outcome_rows[sid] for sid in report["plan"].stage_ids_topo()
                     if sid in outcome_rows],
    }
    if unknown and not args.allow_rerun:
        payload["exit_report"] = "REFUSED: 存在 running 未落盘的未知阶段，必须先重跑"
    _write_json(args.output, payload)
    print(json.dumps({"ok": not unknown, "unknown_stages": unknown,
                      "can_build_exit_report": report["can_build_exit_report"]},
                     ensure_ascii=False, indent=2))
    ledger.close()
    return 0 if not unknown else 2


def cmd_rerun_after_crash(args):
    ledger = Ledger(args.ledger)
    version = args.version or ledger.latest_version()
    report = ledger.recover(version)
    if not report["unknown_stages"]:
        print(json.dumps({"ok": True, "message": "无未知阶段，无需重跑"}, ensure_ascii=False))
        ledger.close()
        return
    plan, events = ledger.load_plan_events(version)
    from .engine import simulate

    result = simulate(plan, events)
    ledger.append_simulation(plan, result)
    _write_json(args.output, {
        "version": version,
        "stats": result.stats,
        "outcomes": [result.outcomes[sid] for sid in plan.stage_ids_topo()],
    })
    print(json.dumps({"ok": True, "rerun": True, "stats": result.stats}, ensure_ascii=False))
    ledger.close()


def cmd_crash_demo(args):
    ledger = Ledger(args.ledger)
    ledger.close()
    worker_cmd = [
        sys.executable, "-m", "emergency_drill.crash_worker",
        "--plan", args.plan, "--events", args.events,
        "--ledger", args.ledger, "--crash-after-stage", args.crash_stage,
    ]
    proc = subprocess.run(worker_cmd, capture_output=True, text=True, timeout=30)
    print(json.dumps({
        "worker_returncode": proc.returncode,
        "worker_stdout": proc.stdout.strip().splitlines(),
        "terminated_externally": True,
    }, ensure_ascii=False, indent=2))


def cmd_view(args):
    ledger = Ledger(args.ledger)
    version = args.version or ledger.latest_version()
    invalid = _read_json(args.invalid_file)["invalid_stage_ids"] if args.invalid_file else None
    snapshots = (
        sorted(os.listdir(args.snapshot_dir))
        if args.snapshot_dir and os.path.isdir(args.snapshot_dir)
        else []
    )
    if args.text:
        print(render_for_terminal_safe(ledger, version, invalid, snapshots))
    else:
        from .ui import launch_or_text

        view = launch_or_text(ledger, version, invalid, snapshots)
        if "tk_unavailable_reason" in view:
            from .viewmodel import render_text

            print(render_text(view))
            print(f"[info] Tk 窗口不可用，已使用文本视图: {view['tk_unavailable_reason']}")
    ledger.close()


def render_for_terminal_safe(ledger, version, invalid, snapshots):
    from .ui import render_for_terminal

    return render_for_terminal(ledger, version, invalid, snapshots)


def build_parser():
    parser = argparse.ArgumentParser(prog="drill", description="应急预案演练诊断工作台")
    parser.add_argument("--log", default="work/drill.log.jsonl")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("generate-samples")
    p.add_argument("--sample-dir", default="samples")
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("validate")
    p.add_argument("--plan", required=True)
    p.add_argument("--events", required=True)
    p.add_argument("--inject", choices=["corrupt-plan", "cycle"], default=None)
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("run")
    p.add_argument("--plan", required=True)
    p.add_argument("--events", required=True)
    p.add_argument("--ledger", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("edit")
    p.add_argument("--ledger", required=True)
    p.add_argument("--changeset", required=True)
    p.add_argument("--version", type=int)
    p.add_argument("--output", required=True)
    p.set_defaults(func=cmd_edit)

    p = sub.add_parser("verify-incr")
    p.add_argument("--ledger", required=True)
    p.add_argument("--version", type=int)
    p.set_defaults(func=cmd_verify_incr)

    p = sub.add_parser("new-round")
    p.add_argument("--plan", required=True)
    p.add_argument("--events", required=True)
    p.add_argument("--ledger", required=True)
    p.set_defaults(func=cmd_new_round)

    p = sub.add_parser("snapshot")
    p.add_argument("--ledger", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--version", type=int)
    p.set_defaults(func=cmd_snapshot)

    p = sub.add_parser("open-snapshot")
    p.add_argument("--ledger", required=True)
    p.add_argument("--snapshot", required=True)
    p.add_argument("--version", type=int)
    p.set_defaults(func=cmd_open_snapshot)

    p = sub.add_parser("replay")
    p.add_argument("--ledger", required=True)
    p.add_argument("--snapshot", required=True)
    p.add_argument("--version", type=int)
    p.set_defaults(func=cmd_replay)

    p = sub.add_parser("tamper-snapshot")
    p.add_argument("--snapshot", required=True)
    p.add_argument("--output")
    p.set_defaults(func=cmd_tamper_snapshot)

    p = sub.add_parser("recover")
    p.add_argument("--ledger", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--version", type=int)
    p.add_argument("--allow-rerun", action="store_true")
    p.set_defaults(func=cmd_recover)

    p = sub.add_parser("rerun-after-crash")
    p.add_argument("--ledger", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--version", type=int)
    p.set_defaults(func=cmd_rerun_after_crash)

    p = sub.add_parser("crash-demo")
    p.add_argument("--plan", required=True)
    p.add_argument("--events", required=True)
    p.add_argument("--ledger", required=True)
    p.add_argument("--crash-stage", required=True)
    p.set_defaults(func=cmd_crash_demo)

    p = sub.add_parser("view")
    p.add_argument("--ledger", required=True)
    p.add_argument("--version", type=int)
    p.add_argument("--invalid-file")
    p.add_argument("--snapshot-dir")
    p.add_argument("--text", action="store_true")
    p.set_defaults(func=cmd_view)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    logger = configure_logging(args.log)
    try:
        return args.func(args) or 0
    except DrillError as exc:
        log_business_error(logger, exc, args.command)
        print(json.dumps(exc.as_dict(), ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        logger.error("file missing", extra={"context": {"path": str(exc)}})
        print(json.dumps({"code": "E_PLAN", "message": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
