from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

from .generator import gen_samples
from .model import Config, canonical_bytes, content_hash
from .recompute import verify_workspace
from .scheduler import MatchTask, schedule_matches
from .storage import atomic_write


def _cmd_run(args) -> int:
    workspace = Path(args.workspace)
    cfg = Config()
    metas = gen_samples(workspace / "samples")
    tasks = [
        MatchTask(
            match_id=meta["match_id"],
            sched_priority=meta["sched_priority"],
            est_mem_mb=meta["est_mem_mb"],
            sample_path=path,
        )
        for meta, path in metas
    ]
    out = schedule_matches(
        tasks, cfg, workspace,
        max_threads=args.threads,
        mem_budget_mb=args.mem_budget_mb,
        crash_match=args.crash_match,
        crash_after=args.crash_after,
    )
    if out["crashed"]:
        print("SIMULATED CRASH: forced termination before run completion;")
        print("no report or close marker written. Re-run `run` to recover.")
        return 3

    atomic_write(workspace / "schedule.json", canonical_bytes(out["schedule"]))
    recompute = verify_workspace(workspace, cfg)
    denied_records = [
        {"match_id": d["match_id"], "status": d["code"], "reason": d["reason"]}
        for d in out["denied"]
    ]
    report = {
        "tool": "orderflow",
        "version": 1,
        "config": asdict(cfg),
        "matches": sorted(out["results"] + denied_records, key=lambda r: r["match_id"]),
        "schedule": out["schedule"],
        "recompute": recompute,
    }
    atomic_write(workspace / "report.json", canonical_bytes(report))
    atomic_write(
        workspace / "CLOSED",
        ("CLOSED " + content_hash(report) + "\n").encode("utf-8"),
    )

    print(f"workers={out['schedule']['workers']} admitted={out['schedule']['admitted']}")
    for record in report["matches"]:
        line = f"{record['match_id']}: {record['status']}"
        if record.get("final_hash"):
            line += f" steps={record['steps']} final={record['final_hash'][:16]}"
        print(line)
    for match_id, check in recompute.items():
        print(f"recompute {match_id}: {check['status']} ({check['checked_steps']} steps)")
    print(f"report={workspace / 'report.json'}")
    return 0


def _cmd_recompute(args) -> int:
    report = verify_workspace(Path(args.workspace), Config())
    ok = True
    for match_id, check in report.items():
        print(f"{match_id}: {check['status']} checked={check['checked_steps']} ledger_ok={check['ledger_ok']}")
        if check["status"] != "OK":
            ok = False
            print(f"  first_mismatch={check['first_mismatch']}")
    return 0 if ok else 1


def _cmd_gen(args) -> int:
    metas = gen_samples(Path(args.workspace) / "samples")
    for meta, path in metas:
        print(f"{meta['match_id']}: {path}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="orderflow",
        description="Deterministic delayed-order adjudication and budget scheduling",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="generate samples, adjudicate, schedule, recover, recompute")
    run_p.add_argument("--workspace", required=True)
    run_p.add_argument("--threads", type=int, default=8)
    run_p.add_argument("--mem-budget-mb", type=int, default=40)
    run_p.add_argument("--crash-match", default=None)
    run_p.add_argument("--crash-after", type=int, default=None)
    run_p.set_defaults(func=_cmd_run)

    rec_p = sub.add_parser("recompute", help="verify on-disk snapshots by independent recompute")
    rec_p.add_argument("--workspace", required=True)
    rec_p.set_defaults(func=_cmd_recompute)

    gen_p = sub.add_parser("gen", help="generate the deterministic sample set only")
    gen_p.add_argument("--workspace", required=True)
    gen_p.set_defaults(func=_cmd_gen)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
