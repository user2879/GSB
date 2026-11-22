"""Command-line interface for the page aggregation component."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from .canonical import canonical_json, write_json_atomic, write_text_atomic
from .engine import assert_route_equivalence, build_execution
from .errors import DomainError
from .logging_utils import log_domain_error
from .replay import replay_database, validate_database
from .sample import generate_sample, inject_fault
from .service import (
    aggregate_database,
    evolve_database,
    prepare_execution,
    recover_database,
    _check_budget,
)


def _route_value(value: str) -> str | None:
    return None if value.lower() == "auto" else value.upper()


def _emit(value: Any, *, pretty: bool = False) -> None:
    if pretty:
        print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))
    else:
        print(canonical_json(value))


def _add_output_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json-out", help="write the canonical JSON response to this path")
    parser.add_argument("--pretty", action="store_true", help="pretty-print the JSON response")
    parser.add_argument("--log-path", help="structured JSONL log path for business errors")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pageagg",
        description="Deterministic two-route columnar page aggregation and SQLite ledger",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate a deterministic local sample")
    generate.add_argument("--out", default="data/sample", help="sample output directory")
    generate.add_argument("--pages", type=int, default=8, help="distinct page count")
    generate.add_argument("--rows", type=int, default=24, help="fixed rows per page")
    generate.add_argument("--force", action="store_true", help="replace a non-empty output directory")
    _add_output_options(generate)

    inject = subparsers.add_parser("inject", help="copy a sample and inject a deterministic fault")
    inject.add_argument("--catalog", required=True, help="source catalog.json")
    inject.add_argument("--out", required=True, help="fault copy output directory")
    inject.add_argument(
        "--kind",
        required=True,
        choices=["catalog_truncate", "nullmap_misalign"],
        help="fault kind",
    )
    inject.add_argument("--page-id", type=int, default=2)
    inject.add_argument("--column", default="value_float")
    _add_output_options(inject)

    aggregate = subparsers.add_parser("aggregate", help="execute and persist page aggregation")
    aggregate.add_argument("--catalog", required=True, help="page catalog.json")
    aggregate.add_argument("--db", required=True, help="SQLite ledger path")
    aggregate.add_argument("--budget", required=True, type=int, help="page cache slot budget")
    aggregate.add_argument("--route", choices=["auto", "resident", "rescan"], default="auto")
    aggregate.add_argument("--group-key", help="override the catalog group key")
    aggregate.add_argument("--batch-size", type=int, help="override catalog batch size")
    aggregate.add_argument(
        "--crash-during-batch",
        type=int,
        help="test hook: abruptly exit inside the named uncommitted batch transaction",
    )
    _add_output_options(aggregate)

    recover = subparsers.add_parser("recover", help="identify uncommitted batches after forced exit")
    recover.add_argument("--db", required=True)
    _add_output_options(recover)

    replay = subparsers.add_parser("replay", help="replay committed batches and localize mismatches")
    replay.add_argument("--db", required=True)
    replay.add_argument("--catalog", help="override the catalog path recorded in the ledger")
    _add_output_options(replay)

    validate = subparsers.add_parser("validate", help="independently validate ledger self-consistency")
    validate.add_argument("--db", required=True)
    validate.add_argument("--catalog", help="override the catalog path recorded in the ledger")
    _add_output_options(validate)

    evolve = subparsers.add_parser("evolve", help="replace the conclusion view after configuration evolution")
    evolve.add_argument("--db", required=True)
    evolve.add_argument("--catalog", required=True)
    evolve.add_argument("--budget", required=True, type=int)
    evolve.add_argument("--route", choices=["auto", "resident", "rescan"], default="auto")
    evolve.add_argument("--group-key")
    evolve.add_argument("--batch-size", type=int)
    evolve.add_argument("--dry-run", action="store_true")
    _add_output_options(evolve)

    compare = subparsers.add_parser(
        "compare-routes",
        help="force both routes and compare result/eviction bytes",
    )
    compare.add_argument("--catalog", required=True)
    compare.add_argument("--budget", required=True, type=int)
    compare.add_argument("--group-key")
    compare.add_argument("--batch-size", type=int)
    compare.add_argument("--out-dir", help="write raw canonical result/event files for byte comparison")
    _add_output_options(compare)

    ui = subparsers.add_parser("ui", help="open the Tkinter ledger dashboard")
    ui.add_argument("--db", required=True)
    ui.add_argument("--catalog", help="override the catalog path recorded in the ledger")
    ui.add_argument("--log-path", help="structured JSONL log path")

    return parser


def _write_response_if_requested(args: argparse.Namespace, value: Any) -> None:
    if getattr(args, "json_out", None):
        write_json_atomic(args.json_out, value)


def _run(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "generate":
        return generate_sample(
            args.out,
            page_count=args.pages,
            rows_per_page=args.rows,
            force=args.force,
        ) | {"ok": True, "command": "generate"}
    if args.command == "inject":
        result = inject_fault(
            args.catalog,
            args.out,
            args.kind,
            page_id=args.page_id,
            column=args.column,
        )
        return result | {"ok": True, "command": "inject"}
    if args.command == "aggregate":
        return aggregate_database(
            args.catalog,
            args.db,
            args.budget,
            force_route=_route_value(args.route),
            group_key_override=args.group_key,
            batch_size_override=args.batch_size,
            crash_during_batch=args.crash_during_batch,
        )
    if args.command == "recover":
        return recover_database(args.db)
    if args.command == "replay":
        return replay_database(args.db, catalog_path=args.catalog)
    if args.command == "validate":
        return validate_database(args.db, catalog_path=args.catalog)
    if args.command == "evolve":
        return evolve_database(
            args.db,
            args.catalog,
            args.budget,
            force_route=_route_value(args.route),
            group_key_override=args.group_key,
            batch_size_override=args.batch_size,
            dry_run=args.dry_run,
        )
    if args.command == "compare-routes":
        _check_budget(args.budget)
        prepared = prepare_execution(
            args.catalog,
            args.budget,
            group_key_override=args.group_key,
            batch_size_override=args.batch_size,
        )
        catalog = prepared.catalog
        resident = build_execution(
            catalog,
            args.budget,
            force_route="RESIDENT",
            batch_size=args.batch_size,
        )
        rescan = build_execution(
            catalog,
            args.budget,
            force_route="RESCAN",
            batch_size=args.batch_size,
        )
        equivalence = assert_route_equivalence(resident, rescan)
        if args.out_dir:
            out_dir = Path(args.out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            write_text_atomic(out_dir / "resident.result.json", resident.result_bytes().decode("utf-8"))
            write_text_atomic(out_dir / "rescan.result.json", rescan.result_bytes().decode("utf-8"))
            eviction_text = canonical_json(resident.eviction_events)
            write_text_atomic(out_dir / "eviction.events.json", eviction_text)
            write_text_atomic(out_dir / "eviction.events.from-rescan.json", canonical_json(rescan.eviction_events))
        return {
            "ok": True,
            "command": "compare-routes",
            "catalog_digest": catalog.digest,
            "budget": args.budget,
            "auto_route": prepared.execution.route.payload(),
            "equivalence": equivalence,
            "resident": resident.summary_payload(),
            "rescan": rescan.summary_payload(),
            "raw_result_sha256": hashlib.sha256(resident.result_bytes()).hexdigest(),
        }
    if args.command == "ui":
        from .ui import launch_ui

        launch_ui(args.db, catalog_path=args.catalog, log_path=args.log_path)
        return {"ok": True, "command": "ui", "closed": True}
    raise AssertionError(f"unhandled command: {args.command}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        response = _run(args)
        _write_response_if_requested(args, response)
        _emit(response, pretty=bool(getattr(args, "pretty", False)))
        return 0
    except DomainError as error:
        try:
            log_domain_error(error, path=getattr(args, "log_path", None), command=args.command)
        except Exception:
            pass
        response = error.to_dict() | {"command": args.command}
        try:
            _write_response_if_requested(args, response)
        except Exception:
            pass
        _emit(response, pretty=bool(getattr(args, "pretty", False)))
        return error.exit_code
    except (ValueError, FileExistsError, KeyError) as error:
        response = {
            "ok": False,
            "code": "E_PAGE",
            "category": "E_PAGE",
            "message": str(error),
            "page_id": None,
            "field": "arguments",
            "event_seq": None,
            "details": {},
            "command": args.command,
        }
        _emit(response, pretty=bool(getattr(args, "pretty", False)))
        return 10
    except Exception as error:
        response = {
            "ok": False,
            "code": "E_INTERNAL",
            "category": "E_INTERNAL",
            "message": f"{type(error).__name__}: {error}",
            "page_id": None,
            "field": None,
            "event_seq": None,
            "details": {},
            "command": args.command,
        }
        _emit(response, pretty=bool(getattr(args, "pretty", False)))
        return 1


if __name__ == "__main__":
    sys.exit(main())
