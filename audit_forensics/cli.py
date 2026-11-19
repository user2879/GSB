"""Interactive command-line interface for the offline forensics workbench."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from . import __version__
from .engine import AuditEngine, CommandResult
from .errors import AuditConfigError, E_RULES_INVALID, business_error
from .rules import load_rules


def _json_print(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _flatten_inputs(args: argparse.Namespace) -> list[str]:
    result: list[str] = list(getattr(args, "inputs", []) or [])
    for group in getattr(args, "input_groups", []) or []:
        result.extend(group)
    return result


def _engine_for_args(
    args: argparse.Namespace, *, load_rule_config: bool = False
) -> AuditEngine:
    workspace = Path(args.workspace)
    rules = None
    if load_rule_config:
        rules_path = args.rules
        if rules_path is None:
            candidate = workspace / "rules.json"
            if candidate.exists():
                rules_path = str(candidate)
        if rules_path:
            rules = load_rules(rules_path)
    return AuditEngine(workspace, rules=rules)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audit-forensics",
        description="Offline deterministic audit-event correlation and forensics",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    import_parser = subparsers.add_parser(
        "import",
        aliases=["ingest"],
        help="import one or more v1/v2 event streams",
    )
    import_parser.add_argument("--workspace", "-w", required=True)
    import_parser.add_argument("--rules", "--rules-config", "-r")
    import_parser.add_argument("--input", "-i", dest="input_groups", nargs="+", action="append")
    import_parser.add_argument("inputs", nargs="*")
    import_parser.add_argument("--workers", "--concurrency", type=int, default=1)
    import_parser.add_argument("--report")
    import_parser.add_argument("--graph")

    query_parser = subparsers.add_parser(
        "query-case",
        aliases=["case", "query"],
        help="query one case or list all cases",
    )
    query_parser.add_argument("--workspace", "-w", required=True)
    query_parser.add_argument("case_id_positional", nargs="?")
    query_parser.add_argument("--case-id", "--case", dest="case_id_option")
    query_parser.add_argument("--rules")

    export_parser = subparsers.add_parser(
        "export-report",
        aliases=["export", "report"],
        help="export deterministic JSON report and case graph",
    )
    export_parser.add_argument("--workspace", "-w", required=True)
    export_parser.add_argument("--output", "--report-output", "-o")
    export_parser.add_argument("--graph-output", "--graph")
    export_parser.add_argument("--rules")

    verify_parser = subparsers.add_parser(
        "verify-checkpoint",
        aliases=["verify", "checkpoint"],
        help="validate checkpoint, snapshot, and full case recomputation",
    )
    verify_parser.add_argument("--workspace", "-w", required=True)
    verify_parser.add_argument("--rules")
    return parser


def _run(args: argparse.Namespace) -> CommandResult:
    if args.command in ("import", "ingest"):
        if args.workers < 1:
            raise AuditConfigError("--workers must be at least 1")
        engine = _engine_for_args(args, load_rule_config=True)
        return engine.import_streams(
            _flatten_inputs(args),
            workers=args.workers,
            report_path=args.report,
            graph_path=args.graph,
        )
    if args.command in ("query-case", "case", "query"):
        engine = _engine_for_args(args)
        case_id = args.case_id_option or args.case_id_positional
        return engine.query_case(case_id)
    if args.command in ("export-report", "export", "report"):
        engine = _engine_for_args(args)
        return engine.export_report(
            output=args.output,
            graph_output=args.graph_output,
        )
    if args.command in ("verify-checkpoint", "verify", "checkpoint"):
        engine = _engine_for_args(args)
        return engine.verify_checkpoint()
    raise AuditConfigError(f"unknown command: {args.command}", code=E_RULES_INVALID)


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        result = _run(args)
    except AuditConfigError as exc:
        _json_print(
            {
                "status": "error",
                "exit_code": 2,
                "errors": [business_error(exc.code, str(exc))],
            }
        )
        return 2
    except Exception as exc:
        _json_print(
            {
                "status": "error",
                "exit_code": 1,
                "errors": [
                    business_error(
                        "E_INTERNAL",
                        f"unexpected internal error: {type(exc).__name__}: {exc}",
                    )
                ],
            }
        )
        return 1
    _json_print(result.payload)
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
