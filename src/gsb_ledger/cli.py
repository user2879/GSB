"""Command-line entry point for validation, ledger and diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .config import ConfigValidation, config_hash, validate_project
from .conversions import Sidecar, convert_receipt, receipt_json_payload
from .engine import LedgerEngine
from .generator import generate_file
from .receipts import detect_version, parse_receipt_text
from .scenario import run_scenario
from .util import atomic_write_json, atomic_write_text, canonical_json


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def _write_or_print_json(value: Any, output: str | None) -> None:
    if output:
        atomic_write_json(Path(output), value)
        _print({"written": str(Path(output)), "summary": value.get("summary", {})})
    else:
        _print(value)


def _engine(args: argparse.Namespace) -> LedgerEngine:
    return LedgerEngine(
        args.project,
        config_path=args.config,
        requests_path=args.requests,
        state_dir=args.state_dir,
        fault_config=getattr(args, "fault_config", None),
        command_cursor=getattr(args, "cursor", 0),
    )


def _cmd_validate(args: argparse.Namespace) -> int:
    config, requests = validate_project(
        Path(args.project) / args.config,
        Path(args.project) / args.requests,
    )
    _print(
        {
            "ok": True,
            "config_hash": config_hash(config),
            "workers": len(config.workers),
            "requests": len(requests),
            "batch_size": config.batch_size,
            "receipt_sources": len(config.receipt_sources),
            "merge_entries": len(config.merge_entries),
        }
    )
    return 0


def _cmd_ledger(args: argparse.Namespace) -> int:
    engine = _engine(args)
    result = engine.build(
        resume=not args.no_resume,
        save_snapshot=True,
        render_version=args.render_version,
    )
    if args.build_info:
        _write_or_print_json({"ledger": result.ledger, "build_info": result.build_info}, args.output)
    else:
        _write_or_print_json(result.ledger, args.output)
    return 0 if not result.ledger["failures"] else 1


def _cmd_query(args: argparse.Namespace) -> int:
    engine = _engine(args)
    budget = args.budget_ms
    if args.verify or args.render_version is not None:
        result = engine.build(
            resume=not args.no_resume,
            save_snapshot=True,
            render_version=args.render_version,
        )
        view = engine.query(mode=args.mode, budget_ms=budget, result=result)
    else:
        view = engine.query(
            mode=args.mode,
            budget_ms=budget,
            resume=not args.no_resume,
        )
    if args.verify:
        authority = engine.query(
            mode="authoritative", budget_ms=None, result=result
        )
        from .views import compare_views

        issues = compare_views(view, authority)
        view["verification"] = {
            "consistent": not issues,
            "issues": [issue.to_dict() for issue in issues],
        }
    _write_or_print_json(view, args.output)
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    engine = _engine(args)
    report = engine.verify_views(budget_ms=args.budget_ms)
    _write_or_print_json(report, args.output)
    return 0 if report["consistent"] else 1


def _cmd_diagnose(args: argparse.Namespace) -> int:
    engine = _engine(args)
    if not args.fault_config:
        config = engine.load()[0]
        if not config.fault_config:
            raise ValueError("no fault config supplied or configured")
        engine.fault_config_override = config.fault_config
    result = engine.build(resume=False, save_snapshot=True)
    diagnostics = result.ledger["diagnostics"]
    missed = [
        item["injection_id"]
        for item in diagnostics
        if item.get("enabled") and not item.get("hit")
    ]
    code_map = {
        item["injection_id"]: {
            "kind": item["kind"],
            "expected_code": item["expected_code"],
            "observed_codes": item["observed_codes"],
            "hit": item["hit"],
        }
        for item in diagnostics
    }
    report = {
        "ok": not missed and bool(diagnostics),
        "diagnostics": diagnostics,
        "missed": missed,
        "code_map": code_map,
        "failure_codes": sorted({item["code"] for item in result.ledger["failures"]}),
        "summary": result.ledger["summary"],
        "ledger_version": result.ledger_version,
    }
    _write_or_print_json(report, args.output)
    return 0 if report["ok"] else 1


def _cmd_snapshot_status(args: argparse.Namespace) -> int:
    engine = _engine(args)
    _print(engine.status(render_version=args.render_version))
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    project = Path(args.project)
    rows = generate_file(
        config_path=project / args.config,
        requests_path=project / args.requests,
        output_path=project / args.output,
        worker_id=args.worker_id,
        version=args.version,
        param_version=args.param_version,
        batch_size=args.batch_size,
        request_ids=args.request_id,
        batch_prefix=args.batch_prefix,
        duration_ms=args.duration_ms,
    )
    _print({"ok": True, "rows": len(rows), "output": args.output})
    return 0


def _cmd_convert(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    source_version = detect_version(input_path, args.source_version)
    if source_version is None:
        raise ValueError("cannot determine source version; pass --source-version")
    sidecar = Sidecar.load(args.sidecar) if args.sidecar else Sidecar()
    results: list[dict[str, Any]] = []
    output_lines: list[str] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            text = raw_line.strip()
            if not text:
                continue
            outcome = parse_receipt_text(
                text,
                version=source_version,
                source_path=str(input_path),
                line=line_number,
                default_worker_id=args.worker_id,
                default_param_version=args.default_param_version,
                allow_explicit_missing=True,
            )
            if outcome.receipt is None:
                raise ValueError(
                    f"cannot parse row {line_number}: "
                    + "; ".join(issue.message for issue in outcome.issues)
                )
            converted = convert_receipt(
                outcome.receipt,
                args.target_version,
                sidecar=sidecar,
            )
            output_lines.append(canonical_json(receipt_json_payload(converted.receipt)))
            results.append(
                {
                    "line": line_number,
                    "receipt_key": outcome.receipt.key,
                    "direction": converted.direction,
                    "lost_fields": list(converted.lost_fields),
                    "preserved_fields": sorted(converted.preserved_fields),
                    "integrity_label": converted.integrity_label,
                }
            )
    if args.output:
        atomic_write_text(
            Path(args.output),
            "\n".join(output_lines) + ("\n" if output_lines else ""),
        )
    if args.sidecar:
        sidecar.save(args.sidecar)
    report = {
        "ok": True,
        "source_version": source_version,
        "target_version": args.target_version,
        "converted": results,
        "sidecar": args.sidecar,
        "output": args.output,
    }
    _print(report)
    return 0


def _cmd_scenario(args: argparse.Namespace) -> int:
    report = run_scenario(
        source_project=args.source_project,
        work_dir=args.work_dir,
        commands_path=args.commands,
        out_dir=args.out_dir,
    )
    _print(report)
    return 0 if report["identical"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gsb-ledger",
        description="Normalize r1/r2/r3 inference receipts into a deterministic ledger.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common(subparser: argparse.ArgumentParser, *, faults: bool = True) -> None:
        subparser.add_argument("--project", default="sample")
        subparser.add_argument("--config", default="config.json")
        subparser.add_argument("--requests", default="requests.ndjson")
        subparser.add_argument("--state-dir", default=".state")
        if faults:
            subparser.add_argument("--fault-config")

    validate_parser = subparsers.add_parser("validate")
    common(validate_parser, faults=False)
    validate_parser.set_defaults(func=_cmd_validate)

    ledger_parser = subparsers.add_parser("ledger")
    common(ledger_parser)
    ledger_parser.add_argument("--output")
    ledger_parser.add_argument("--render-version", choices=["r1", "r2", "r3"])
    ledger_parser.add_argument("--no-resume", action="store_true")
    ledger_parser.add_argument("--build-info", action="store_true")
    ledger_parser.add_argument("--cursor", type=int, default=0)
    ledger_parser.set_defaults(func=_cmd_ledger)

    query_parser = subparsers.add_parser("query")
    common(query_parser)
    query_parser.add_argument("--mode", choices=["fast", "authoritative"], default="fast")
    query_parser.add_argument("--budget-ms", type=int)
    query_parser.add_argument("--render-version", choices=["r1", "r2", "r3"])
    query_parser.add_argument("--no-resume", action="store_true")
    query_parser.add_argument("--verify", action="store_true")
    query_parser.add_argument("--output")
    query_parser.add_argument("--cursor", type=int, default=0)
    query_parser.set_defaults(func=_cmd_query)

    compare_parser = subparsers.add_parser("compare")
    common(compare_parser)
    compare_parser.add_argument("--budget-ms", type=int)
    compare_parser.add_argument("--output")
    compare_parser.add_argument("--cursor", type=int, default=0)
    compare_parser.set_defaults(func=_cmd_compare)

    diagnose_parser = subparsers.add_parser("diagnose")
    common(diagnose_parser)
    diagnose_parser.add_argument("--output")
    diagnose_parser.add_argument("--cursor", type=int, default=0)
    diagnose_parser.set_defaults(func=_cmd_diagnose)

    status_parser = subparsers.add_parser("snapshot-status")
    common(status_parser, faults=False)
    status_parser.add_argument("--render-version", choices=["r1", "r2", "r3"])
    status_parser.set_defaults(func=_cmd_snapshot_status)

    generate_parser = subparsers.add_parser("generate")
    common(generate_parser, faults=False)
    generate_parser.add_argument("--worker-id", required=True)
    generate_parser.add_argument("--version", choices=["r1", "r2", "r3"], required=True)
    generate_parser.add_argument("--param-version", type=int)
    generate_parser.add_argument("--batch-size", type=int)
    generate_parser.add_argument("--request-id", action="append")
    generate_parser.add_argument("--batch-prefix")
    generate_parser.add_argument("--duration-ms", type=int)
    generate_parser.add_argument("--output", required=True)
    generate_parser.set_defaults(func=_cmd_generate)

    convert_parser = subparsers.add_parser("convert")
    convert_parser.add_argument("--input", required=True)
    convert_parser.add_argument("--source-version", choices=["r1", "r2", "r3"])
    convert_parser.add_argument("--target-version", choices=["r1", "r2", "r3"], required=True)
    convert_parser.add_argument("--sidecar")
    convert_parser.add_argument("--output")
    convert_parser.add_argument("--worker-id")
    convert_parser.add_argument("--default-param-version", type=int)
    convert_parser.set_defaults(func=_cmd_convert)

    scenario_parser = subparsers.add_parser("scenario")
    scenario_parser.add_argument("--source-project", required=True)
    scenario_parser.add_argument("--work-dir", required=True)
    scenario_parser.add_argument("--commands", required=True)
    scenario_parser.add_argument("--out-dir", required=True)
    scenario_parser.set_defaults(func=_cmd_scenario)

    worker_parser = subparsers.add_parser("worker")
    worker_parser.add_argument("--project", default="sample")
    worker_parser.add_argument("--config", default="config.json")
    worker_parser.add_argument("--requests", default="requests.ndjson")
    worker_parser.add_argument("--worker-id", required=True)
    worker_parser.add_argument("--host", default="127.0.0.1")
    worker_parser.add_argument("--port", type=int)
    worker_parser.set_defaults(func=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(args_list)
    try:
        if args.command == "worker":
            from .worker import main as worker_main

            worker_argv = [
                "--project",
                args.project,
                "--config",
                args.config,
                "--requests",
                args.requests,
                "--worker-id",
                args.worker_id,
                "--host",
                args.host,
            ]
            if args.port is not None:
                worker_argv.extend(["--port", str(args.port)])
            return worker_main(worker_argv)
        return int(args.func(args))
    except ConfigValidation as exc:
        _print(
            {
                "ok": False,
                "code": "E_CONFIG_OR_INPUT_INVALID",
                "failures": exc.issues.to_dicts(),
            }
        )
        return 2
    except (OSError, ValueError, KeyError) as exc:
        _print({"ok": False, "code": type(exc).__name__, "message": str(exc)})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
