"""Command-line interface for validation, runs, verification and recovery."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
from pathlib import Path
from typing import Any, Sequence

from . import __version__
from .config import load_base_config, validate_commands
from .diagnostics import (
    run_fault_injections,
    run_forced_kill_recovery,
    run_lease_checks,
    run_reuse_check,
)
from .engine import LedgerEngine, compare_ledgers
from .errors import QLabError
from .ledger import LedgerStore, recovery_report, snapshot_digest
from .receipts import (
    arbitrate_receipts,
    downgrade_q3_to_q1,
    load_receipts,
    load_sidecars,
    parse_receipt,
    receipt_roundtrip,
    upgrade_to_q3,
)
from .util import atomic_write_json, read_json


def _json_print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))


def _summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "ledger_path": result["ledger_path"],
        "snapshot_digest": result["snapshot_digest"],
        "command_count": result["command_count"],
        "issue_count": result["issue_count"],
        "recovery_report": result["recovery_report"],
    }


def command_validate(args: argparse.Namespace) -> int:
    loaded = load_base_config(args.config)
    issues = [item.as_dict() for item in loaded.issues]
    accepted_commands = loaded.commands
    if args.commands:
        commands, command_issues = validate_commands(read_json(args.commands), loaded.state)
        accepted_commands = commands
        issues.extend(item.as_dict() for item in command_issues)
    report = {
        "ok": not issues,
        "worker_count": len(loaded.state.workers),
        "layer_count": len(loaded.state.layers),
        "batch_count": len(loaded.state.batches),
        "accepted_command_count": len(accepted_commands),
        "receipt_paths": loaded.receipt_paths,
        "issues": issues,
    }
    _json_print(report)
    return 0 if not issues else 3


def _run_engine(args: argparse.Namespace, *, rebuild: bool) -> int:
    engine = LedgerEngine(
        args.config,
        commands_path=args.commands,
        runtime_dir=args.runtime,
        ledger_path=args.ledger,
        lease_ttl_ms=args.lease_ttl_ms,
        request_timeout_s=args.timeout,
    )
    result = engine.run(rebuild=rebuild)
    _json_print(_summary(result))
    return 0


def command_run(args: argparse.Namespace) -> int:
    return _run_engine(args, rebuild=False)


def command_rebuild(args: argparse.Namespace) -> int:
    return _run_engine(args, rebuild=True)


def _write_prefix_command_file(source: str | Path, destination: Path, split: int) -> int:
    raw = read_json(source)
    commands = raw.get("commands", raw) if isinstance(raw, dict) else raw
    if not isinstance(commands, list):
        raise QLabError("E_CONFIG", "commands must be a list")
    count = max(1, min(split, len(commands) - 1)) if len(commands) > 1 else len(commands)
    atomic_write_json(destination, {"commands": commands[:count]})
    return count


def _receipt_consistency(runtime_dir: Path, ledger: dict[str, Any]) -> dict[str, Any]:
    receipts_dir = runtime_dir / "receipts"
    q_paths = [receipts_dir / "q1.json", receipts_dir / "q2.json", receipts_dir / "q3.json"]
    counts: dict[str, int] = {}
    issues: list[dict[str, Any]] = []
    normalized = []
    for path in q_paths:
        if not path.exists():
            issues.append({"code": "E_BAD_RECEIPT", "path": str(path), "message": "missing receipt file"})
            counts[path.stem] = 0
            continue
        parsed, parsed_issues = load_receipts(
            [path], sidecar_path=receipts_dir / "q3_to_q1_sidecar.json"
        )
        counts[path.stem] = len(parsed)
        normalized.extend(parsed)
        issues.extend(item.as_dict() for item in parsed_issues)
    arbitration = arbitrate_receipts(normalized)
    authority_match = all(
        batch_id in arbitration["authority"]
        and arbitration["authority"][batch_id]["decision_digest"]
        == ledger["authority"].get(batch_id, {}).get("decision_digest")
        for batch_id in sorted(arbitration["authority"])
    )
    roundtrip_ok = all(
        item.get("conclusion_unchanged") for item in ledger.get("roundtrip_checks", [])
    )
    q1_completeness = sorted(
        {
            item["completeness"]["label"]
            for item in (receipt.to_internal_dict() for receipt in normalized if receipt.format == 1)
        }
    )
    q2_completeness = sorted(
        {
            item["completeness"]["label"]
            for item in (receipt.to_internal_dict() for receipt in normalized if receipt.format == 2)
        }
    )
    q3_completeness = sorted(
        {
            item["completeness"]["label"]
            for item in (receipt.to_internal_dict() for receipt in normalized if receipt.format == 3)
        }
    )
    return {
        "ok": len(set(counts.values())) == 1 and not issues and authority_match and roundtrip_ok,
        "counts": counts,
        "issues": issues,
        "authority_decisions": len(arbitration["decisions"]),
        "authority_matches_ledger": authority_match,
        "roundtrip_conclusions_unchanged": roundtrip_ok,
        "completeness_labels": {"q1": q1_completeness, "q2": q2_completeness, "q3": q3_completeness},
    }


def command_verify(args: argparse.Namespace) -> int:
    root = Path(args.runtime).resolve()
    incremental_dir = root / "incremental"
    rebuild_dir = root / "rebuild"
    for target in (incremental_dir, rebuild_dir):
        if target.exists():
            shutil.rmtree(target)
    root.mkdir(parents=True, exist_ok=True)
    prefix_path = root / "commands_prefix.json"
    split = _write_prefix_command_file(args.commands, prefix_path, args.split)

    first_result = LedgerEngine(
        args.config,
        commands_path=prefix_path,
        runtime_dir=incremental_dir,
        lease_ttl_ms=args.lease_ttl_ms,
        request_timeout_s=args.timeout,
    ).run()
    second = LedgerEngine(
        args.config,
        commands_path=args.commands,
        runtime_dir=incremental_dir,
        lease_ttl_ms=args.lease_ttl_ms,
        request_timeout_s=args.timeout,
    ).run()
    rebuilt = LedgerEngine(
        args.config,
        commands_path=args.commands,
        runtime_dir=rebuild_dir,
        lease_ttl_ms=args.lease_ttl_ms,
        request_timeout_s=args.timeout,
    ).run(rebuild=True)
    incremental_ledger = LedgerStore(incremental_dir / "ledger.json").load()
    rebuilt_ledger = LedgerStore(rebuild_dir / "ledger.json").load()
    comparison = compare_ledgers(incremental_ledger, rebuilt_ledger)
    receipt_consistency = _receipt_consistency(incremental_dir, incremental_ledger)

    lease = run_lease_checks(args.config, runtime_dir=root / "lease")
    reuse = run_reuse_check(args.config, runtime_dir=root / "reuse")
    injections = run_fault_injections(args.config, args.injections, runtime_dir=root / "faults")
    recovery = run_forced_kill_recovery(
        args.config,
        seed_ledger=reuse["ledger_path"],
        runtime_dir=root / "kill-recovery",
    )
    report = {
        "schema": "quantledger-verification-report-v1",
        "ok": bool(
            comparison["equal"]
            and receipt_consistency["ok"]
            and lease["ok"]
            and reuse["ok"]
            and injections["all_hit"]
            and recovery["worker_force_killed"]
            and recovery["pending_queue"]
        ),
        "tool": {"name": "quantledger", "version": __version__},
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": _numpy_version(),
        },
        "fixture": {"config": str(Path(args.config).resolve()), "commands": str(Path(args.commands).resolve())},
        "incremental_staging": {
            "prefix_commands": split,
            "prefix_snapshot_digest": first_result["snapshot_digest"],
            "total_commands": second["command_count"],
        },
        "incremental": {
            "snapshot_digest": snapshot_digest(incremental_ledger),
            "command_count": second["command_count"],
            "diagnostics": incremental_ledger.get("diagnostics", {}),
        },
        "rebuilt": {
            "snapshot_digest": snapshot_digest(rebuilt_ledger),
            "command_count": rebuilt["command_count"],
            "diagnostics": rebuilt_ledger.get("diagnostics", {}),
        },
        "comparison": comparison,
        "receipts": receipt_consistency,
        "leases": lease,
        "cache_reuse": reuse,
        "fault_injections": injections,
        "forced_kill_recovery": recovery,
        "actual_validation": {
            "incremental_equals_rebuild": comparison["equal"],
            "three_receipt_formats_consistent": receipt_consistency["ok"],
            "lease_reclamation_real": lease["ok"],
            "layer_scoped_reuse_verified": reuse["ok"],
            "fault_ids_all_hit": injections["all_hit"],
            "recovery_from_atomic_ledger_only": recovery["atomic_ledger_only"],
        },
    }
    ledger_output = Path(args.ledger_output)
    receipts_output = Path(args.receipts_output)
    ledger_output.parent.mkdir(parents=True, exist_ok=True)
    receipts_output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(ledger_output, incremental_ledger)
    for name in ("q1.json", "q2.json", "q3.json", "q3_to_q1_sidecar.json"):
        source = incremental_dir / "receipts" / name
        if source.exists():
            shutil.copyfile(source, receipts_output / name)
    report["artifacts"] = {
        "ledger": str(ledger_output.resolve()),
        "receipts_dir": str(receipts_output.resolve()),
    }
    output = Path(args.output) if args.output else Path("artifacts/verification-report.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, report)
    _json_print(
        {
            "ok": report["ok"],
            "report_path": str(output.resolve()),
            "ledger_path": str(ledger_output.resolve()),
            "receipts_dir": str(receipts_output.resolve()),
            "comparison_equal": comparison["equal"],
            "comparison_digest": comparison["incremental_digest"],
            "receipts_ok": receipt_consistency["ok"],
            "lease_ok": lease["ok"],
            "reuse_ok": reuse["ok"],
            "reuse_entries": reuse["reuse_entries"],
            "faults_all_hit": injections["all_hit"],
            "recovery_pending": recovery["pending_queue"],
            "recovery_E_RECONCILE_PENDING": recovery["E_RECONCILE_PENDING"],
        }
    )
    return 0 if report["ok"] else 4


def _numpy_version() -> str:
    import numpy as np

    return np.__version__


def command_recover(args: argparse.Namespace) -> int:
    ledger = LedgerStore(args.ledger).load()
    if ledger is None:
        _json_print({"ok": False, "code": "E_LEDGER_CORRUPT", "message": "ledger does not exist"})
        return 2
    report = recovery_report(ledger, ledger_path=args.ledger)
    _json_print({"ok": True, **report})
    return 0


def command_receipts(args: argparse.Namespace) -> int:
    receipts, issues = load_receipts(args.inputs, sidecar_path=args.sidecar)
    arbitration = arbitrate_receipts(receipts)
    report = {
        "ok": not issues,
        "receipt_count": len(receipts),
        "formats": {version: sum(1 for receipt in receipts if receipt.format == version) for version in (1, 2, 3)},
        "issues": [item.as_dict() for item in issues],
        "decisions": arbitration["decisions"],
        "roundtrips": [receipt_roundtrip(receipt)["conclusion_unchanged"] for receipt in receipts],
    }
    if args.output:
        atomic_write_json(args.output, report)
    _json_print(report)
    return 0 if not issues else 3


def command_convert(args: argparse.Namespace) -> int:
    raw = read_json(args.input)
    records = raw if isinstance(raw, list) else raw.get("receipts", [raw])
    input_sidecars = load_sidecars(args.sidecar)
    output_records: list[dict[str, Any]] = []
    sidecars: list[dict[str, Any]] = []
    for item in records:
        receipt = parse_receipt(item, sidecars=input_sidecars)
        q3 = upgrade_to_q3(receipt)
        if args.to == 3:
            output_records.append(q3.to_q3())
        elif args.to == 2:
            output_records.append(q3.to_q2())
            if receipt.format == 3:
                _, sidecar = downgrade_q3_to_q1(q3)
                sidecars.append(sidecar)
        elif args.to == 1:
            q1, sidecar = downgrade_q3_to_q1(q3)
            output_records.append(q1)
            sidecars.append(sidecar)
        else:
            raise QLabError("E_BAD_RECEIPT", f"unsupported target format {args.to}")
    atomic_write_json(args.output, output_records)
    if args.sidecar and sidecars and args.to < 3:
        atomic_write_json(args.sidecar, {"schema": f"q3-to-q{args.to}-sidecar-v1", "sidecars": sidecars})
    _json_print(
        {
            "ok": True,
            "input_count": len(records),
            "output_count": len(output_records),
            "target_format": args.to,
            "output": str(Path(args.output).resolve()),
            "sidecar": str(Path(args.sidecar).resolve()) if args.sidecar and sidecars else None,
        }
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quantledger")
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate configuration and command stream")
    validate.add_argument("--config", required=True)
    validate.add_argument("--commands")
    validate.set_defaults(func=command_validate)

    for name, function, help_text in (
        ("run", command_run, "run/extend the atomic ledger"),
        ("rebuild", command_rebuild, "rebuild from the base configuration and full command stream"),
    ):
        sub = subparsers.add_parser(name, help=help_text)
        sub.add_argument("--config", required=True)
        sub.add_argument("--commands", required=True)
        sub.add_argument("--runtime", default="runtime")
        sub.add_argument("--ledger")
        sub.add_argument("--lease-ttl-ms", type=int, default=5000)
        sub.add_argument("--timeout", type=float, default=30.0)
        sub.set_defaults(func=function)

    verify = subparsers.add_parser("verify", help="run staged incremental vs blank rebuild acceptance")
    verify.add_argument("--config", required=True)
    verify.add_argument("--commands", required=True)
    verify.add_argument("--injections", required=True)
    verify.add_argument("--runtime", default="runtime/verify")
    verify.add_argument("--output", default="artifacts/verification-report.json")
    verify.add_argument("--ledger-output", default="artifacts/ledger.json")
    verify.add_argument("--receipts-output", default="artifacts/receipts")
    verify.add_argument("--split", type=int, default=6)
    verify.add_argument("--lease-ttl-ms", type=int, default=5000)
    verify.add_argument("--timeout", type=float, default=30.0)
    verify.set_defaults(func=command_verify)

    recover = subparsers.add_parser("recover", help="report recovery state using the ledger only")
    recover.add_argument("--ledger", required=True)
    recover.set_defaults(func=command_recover)

    receipts = subparsers.add_parser("receipts", help="parse and arbitrate q1/q2/q3 receipts")
    receipts.add_argument("inputs", nargs="+")
    receipts.add_argument("--sidecar")
    receipts.add_argument("--output")
    receipts.set_defaults(func=command_receipts)

    convert = subparsers.add_parser("convert", help="convert a receipt file to another historical format")
    convert.add_argument("--input", required=True)
    convert.add_argument("--to", type=int, choices=(1, 2, 3), required=True)
    convert.add_argument("--output", required=True)
    convert.add_argument("--sidecar")
    convert.set_defaults(func=command_convert)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except QLabError as exc:
        _json_print(exc.as_dict())
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
