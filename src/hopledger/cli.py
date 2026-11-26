from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .canonical import canonical_text
from .ledger import Ledger
from .proc import ProcessManager
from .runner import Runner
from .sample import make_sample_configs
from .snapshot import open_or_replace_snapshot
from .specs import load_bundle


def _write_report(path: str | Path | None, report: dict[str, Any]) -> None:
    text = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    else:
        print(text)


def _request_summaries(ledger: Ledger) -> list[dict[str, Any]]:
    summaries = []
    for request_id in ledger.committed_request_ids():
        conclusion = ledger.get_conclusion(request_id) or {}
        attribution = ledger.get_attribution(request_id) or {}
        summaries.append({
            "request_id": request_id,
            "e2e_total_units": conclusion.get("e2e_total_units"),
            "dominant_hop": attribution.get("dominant_hop"),
            "primary_cause": attribution.get("primary_cause"),
        })
    return summaries


def cmd_init_sample(args: argparse.Namespace) -> int:
    bundle = make_sample_configs(
        args.dir / "config",
        request_count=args.requests,
        batch_size=args.batch_size,
    )
    print(canonical_text({
        "ok": True,
        "config_dir": str(bundle.config_dir),
        "ports": bundle.ports,
        "request_ids": bundle.request_ids,
    }))
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    bundle = load_bundle(args.config_dir)
    print(canonical_text({
        "ok": True,
        "hop_set_version": bundle.chain.hop_set_version,
        "scale_caliber_version": bundle.diagnostic.scale_caliber_version,
        "chain": bundle.chain.node_ids,
        "ports": bundle.ports,
        "requests": bundle.request_ids,
    }))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    bundle = load_bundle(args.config_dir)
    ledger_path = Path(args.ledger)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(ledger_path)
    manager = ProcessManager(bundle, ledger_path.parent / "logs")
    runner = Runner(bundle, ledger, manager=manager)
    try:
        manager.start_all()
        run_report = runner.run_all()
        snapshot_report = None
        if args.snapshot:
            _snapshot, replacement = open_or_replace_snapshot(
                args.snapshot, ledger, bundle.chain, bundle.diagnostic
            )
            snapshot_report = replacement
        report = {
            "ok": True,
            "request_ids": bundle.request_ids,
            "run": run_report,
            "requests": _request_summaries(ledger),
            "ledger_digest_valid": ledger.verify_content_digest(),
            "content_digest": ledger.get_meta("content_digest"),
            "snapshot": snapshot_report,
            "max_observed_global": runner.max_observed_global,
            "max_observed_hop": runner.max_observed_hop,
        }
        _write_report(args.report, report)
        return 0
    finally:
        manager.stop_all()
        ledger.close()


def cmd_recover(args: argparse.Namespace) -> int:
    bundle = load_bundle(args.config_dir)
    ledger = Ledger(args.ledger)
    manager = ProcessManager(bundle, Path(args.ledger).parent / "logs")
    runner = Runner(bundle, ledger, manager=manager)
    try:
        unknown_before = ledger.unknown_batches()
        marked = ledger.mark_unknown_batches()
        manager.start_all()
        replay_reports = runner.rerun_unknown_batches()
        snapshot_report = None
        if args.snapshot:
            _snapshot, replacement = open_or_replace_snapshot(
                args.snapshot, ledger, bundle.chain, bundle.diagnostic
            )
            snapshot_report = replacement
        report = {
            "ok": True,
            "unknown_before": unknown_before,
            "marked_unknown_batches": marked,
            "replayed_batches": replay_reports,
            "requests": _request_summaries(ledger),
            "ledger_digest_valid": ledger.verify_content_digest(),
            "snapshot": snapshot_report,
        }
        _write_report(args.report, report)
        return 0
    finally:
        manager.stop_all()
        ledger.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hopledger")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init-sample")
    init_parser.add_argument("--dir", type=Path, required=True)
    init_parser.add_argument("--requests", type=int, default=4)
    init_parser.add_argument("--batch-size", type=int, default=2)
    init_parser.set_defaults(func=cmd_init_sample)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--config-dir", type=Path, required=True)
    validate_parser.set_defaults(func=cmd_validate)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--config-dir", type=Path, required=True)
    run_parser.add_argument("--ledger", type=Path, required=True)
    run_parser.add_argument("--report", type=Path)
    run_parser.add_argument("--snapshot", type=Path)
    run_parser.set_defaults(func=cmd_run)

    recover_parser = subparsers.add_parser("recover")
    recover_parser.add_argument("--config-dir", type=Path, required=True)
    recover_parser.add_argument("--ledger", type=Path, required=True)
    recover_parser.add_argument("--report", type=Path)
    recover_parser.add_argument("--snapshot", type=Path)
    recover_parser.set_defaults(func=cmd_recover)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
