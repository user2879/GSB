"""Command line interface for the local rate-limit / breaker chain."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .codes import RlcbError
from .engine import Engine
from .hashing import canonical_json
from .invalidate import apply_changes
from .ledger import LEDGER_FORMAT_VERSION, Ledger
from .linejson import parse_with_lines
from .planner import choose_plan
from .samples import write_samples
from .spec import load_bundle


def _print(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))


def _load_edits(path: str | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    data = parse_with_lines(Path(path).read_text("utf-8"))
    if isinstance(data, dict) and "edits" in data:
        data = data["edits"]
    return {k: _strip_meta(v) for k, v in dict(data).items()
            if not k.startswith("__")}


def _strip_meta(value):
    if isinstance(value, dict):
        return {k: _strip_meta(v) for k, v in value.items()
                if not k.startswith("__")}
    if isinstance(value, list):
        return [_strip_meta(v) for v in value]
    return value


def cmd_validate(args: argparse.Namespace) -> int:
    bundle = load_bundle(args.services, args.requests)
    _print({
        "ok": True,
        "services": [s.service_id for s in bundle.services],
        "request_count": len(bundle.requests),
        "global_limits": bundle.global_limits.__dict__,
    })
    return 0


def _all_batch_ids(bundle) -> list[str]:
    return sorted({f"{r.service_id}:{r.batch_id}" for r in bundle.requests})


def _window_count(bundle) -> int:
    total = 0
    for spec in bundle.services:
        reqs = [r for r in bundle.requests if r.service_id == spec.service_id]
        if not reqs:
            continue
        max_seq = max(r.arrival_seq for r in reqs)
        total += max_seq // spec.window_length + 1
    return total


def cmd_run(args: argparse.Namespace) -> int:
    bundle = load_bundle(args.services, args.requests)
    edits = _load_edits(args.edits)
    change_sets = []
    if edits:
        new_services, change_sets = apply_changes(bundle.services, edits)
        bundle.services = new_services
        bundle.service_by_id = {s.service_id: s for s in new_services}

    changed_services = [c.service_id for c in change_sets]
    changed_batches = sorted({
        f"{r.service_id}:{r.batch_id}"
        for r in bundle.requests if r.service_id in set(changed_services)
    })
    plan = choose_plan(
        service_count=len(bundle.services),
        batch_count=len(_all_batch_ids(bundle)),
        window_count=_window_count(bundle),
        changed_services=changed_services,
        changed_batches=changed_batches,
        all_batches=_all_batch_ids(bundle),
        force_mode=args.mode,
    )

    prior_decisions: dict[str, dict[str, Any]] = {}
    prior_ownership: dict[str, dict[str, Any]] = {}
    ledger = Ledger(args.ledger)
    if plan.mode == "incremental":
        prior_decisions = ledger.load_decisions()
        prior_ownership = ledger.load_lease_ownership()

    engine = Engine(bundle, use_workers=not args.in_process)
    result = engine.run(
        plan, change_sets=change_sets,
        prior_decisions=prior_decisions,
        prior_ownership=prior_ownership,
    )

    # Commit each batch atomically.
    by_batch: dict[str, list[dict[str, Any]]] = {}
    for decision in result.decisions:
        by_batch.setdefault(
            f"{decision['service_id']}:{decision['batch_id']}", []
        ).append(decision)
    for batch_key in sorted(by_batch):
        service_id, batch_id = batch_key.split(":", 1)
        spec = bundle.service_by_id[service_id]
        ledger.commit_batch(
            batch={"batch_id": batch_id, "service_id": service_id,
                   "config_version": spec.config_version},
            decisions=by_batch[batch_key],
            lease_ownership=result.lease_ownership,
            lease_events=result.lease_events,
            invalidation=result.invalidation if change_sets else None,
            cursor={"last_batch": batch_key,
                    "format_version": LEDGER_FORMAT_VERSION},
        )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = result.to_dict()
    report["ledger"] = {
        "path": str(Path(args.ledger).resolve()),
        "format_version": LEDGER_FORMAT_VERSION,
        "fingerprint": ledger.snapshot_fingerprint(),
    }
    report_path = out_dir / "run_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2)
        + "\n", "utf-8")
    ledger.close()
    _print({"ok": True, "mode": plan.mode, "reason": plan.reason,
            "report": str(report_path),
            "decisions": len(result.decisions),
            "content_digest": result.content_digest})
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Recompute from scratch and compare every request with the ledger."""
    bundle = load_bundle(args.services, args.requests)
    plan = choose_plan(
        service_count=len(bundle.services),
        batch_count=len(_all_batch_ids(bundle)),
        window_count=_window_count(bundle),
        changed_services=[], changed_batches=[],
        all_batches=_all_batch_ids(bundle), force_mode="full")
    engine = Engine(bundle, use_workers=not args.in_process)
    result = engine.run(plan)
    ledger = Ledger(args.ledger)
    outcome = ledger.verify_replay(result.decisions)
    ledger.close()
    _print({"ok": True, "verify": outcome,
            "fresh_digest": result.content_digest})
    return 0


def cmd_recover(args: argparse.Namespace) -> int:
    bundle = load_bundle(args.services, args.requests)
    ledger = Ledger(args.ledger)
    expected = _all_batch_ids(bundle)
    state = ledger.recover_state(expected)
    plan = choose_plan(
        service_count=len(bundle.services),
        batch_count=len(expected),
        window_count=_window_count(bundle),
        changed_services=[], changed_batches=[],
        all_batches=expected, force_mode="full")
    engine = Engine(bundle, use_workers=not args.in_process)
    result = engine.run(plan)

    stored = ledger.load_decisions()
    replayed = {d["request_id"]: d for d in result.decisions}
    # Only fully committed requests are trusted and compared item by item.
    committed_ids = sorted(stored)
    first_difference = None
    committed_match = 0
    for request_id in committed_ids:
        if canonical_json(stored[request_id]) != canonical_json(
                replayed[request_id]):
            first_difference = request_id
            break
        committed_match += 1

    # Uncommitted batches had their quota reclaimed first; recompute them
    # from the clean baseline and re-commit atomically.
    unknown = set(state["unknown_batches"])
    re_committed: list[str] = []
    by_batch: dict[str, list] = {}
    for decision in result.decisions:
        key = f"{decision['service_id']}:{decision['batch_id']}"
        if key in unknown:
            by_batch.setdefault(key, []).append(decision)
    for key in sorted(by_batch):
        service_id, batch_id = key.split(":", 1)
        spec = bundle.service_by_id[service_id]
        ledger.commit_batch(
            batch={"batch_id": batch_id, "service_id": service_id,
                   "config_version": spec.config_version},
            decisions=by_batch[key],
            lease_ownership=result.lease_ownership,
            lease_events=result.lease_events,
            invalidation=None,
            cursor={"last_batch": key,
                    "format_version": LEDGER_FORMAT_VERSION})
        re_committed.append(key)

    final_verify = ledger.verify_replay(result.decisions)
    fingerprint = ledger.snapshot_fingerprint()
    ledger.close()
    _print({
        "ok": first_difference is None,
        "recovery": state,
        "committed_requests_verified": committed_match,
        "first_difference": first_difference,
        "recommitted_unknown_batches": re_committed,
        "final_verify": final_verify,
        "fingerprint": fingerprint,
    })
    return 0 if first_difference is None else 1


def cmd_snapshot(args: argparse.Namespace) -> int:
    ledger = Ledger(args.ledger)
    incoming = json.loads(Path(args.fingerprint).read_text("utf-8"))
    action = ledger.replace_with(incoming)
    ledger.close()
    _print({"ok": True, "snapshot_action": action})
    return 0


def cmd_sample(args: argparse.Namespace) -> int:
    paths = write_samples(args.out)
    _print({"ok": True, "requests": str(paths["requests"])})
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    from .ui import launch_ui
    return launch_ui(args.report)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rlcb")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("sample")
    p.add_argument("--out", default="samples/acceptance")
    p.set_defaults(func=cmd_sample)

    p = sub.add_parser("validate")
    p.add_argument("--services", required=True)
    p.add_argument("--requests", required=True)
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("run")
    p.add_argument("--services", required=True)
    p.add_argument("--requests", required=True)
    p.add_argument("--edits")
    p.add_argument("--ledger", required=True)
    p.add_argument("--out", default="artifacts/run")
    p.add_argument("--mode", choices=["full", "incremental"])
    p.add_argument("--in-process", action="store_true",
                   help="skip real subprocesses (policy-only runs)")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("verify")
    p.add_argument("--services", required=True)
    p.add_argument("--requests", required=True)
    p.add_argument("--ledger", required=True)
    p.add_argument("--in-process", action="store_true")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("recover")
    p.add_argument("--services", required=True)
    p.add_argument("--requests", required=True)
    p.add_argument("--ledger", required=True)
    p.add_argument("--in-process", action="store_true")
    p.set_defaults(func=cmd_recover)

    p = sub.add_parser("snapshot")
    p.add_argument("--ledger", required=True)
    p.add_argument("--fingerprint", required=True)
    p.set_defaults(func=cmd_snapshot)

    p = sub.add_parser("ui")
    p.add_argument("--report", required=True)
    p.set_defaults(func=cmd_ui)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except RlcbError as exc:
        print(json.dumps(exc.to_dict(), ensure_ascii=False, sort_keys=True,
                         indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
