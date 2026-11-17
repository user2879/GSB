from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .errors import (
    DiagError,
    E_ENV,
    E_LEDGER,
    E_LATENCY,
    E_PORT,
    E_REPLAY,
    E_REV,
    E_SERVICE,
    E_SPEC,
    E_VIEW,
    StructuredLogger,
)
from .runtime import DiagnosticRuntime
from .spec import load_spec


EXIT_CODES = {
    E_SPEC: 2,
    E_REV: 2,
    E_PORT: 3,
    E_LEDGER: 4,
    E_VIEW: 5,
    E_REPLAY: 6,
    E_LATENCY: 7,
    E_SERVICE: 8,
    E_ENV: 9,
}


def _add_common(parser: argparse.ArgumentParser, *, include_budget: bool = False) -> None:
    parser.add_argument("--spec", required=True, help="path to the local JSON specification")
    parser.add_argument("--workdir", default=".diag_runtime", help="ledger, snapshots, and logs directory")
    if include_budget:
        parser.add_argument(
            "--latency-budget-ms",
            type=int,
            default=50,
            help="explicit UI-query budget in integer milliseconds",
        )
    parser.add_argument("--inject-cycle", action="store_true", help="inject a revision parent cycle")
    parser.add_argument(
        "--inject-port-conflict",
        "--inject-port-occupied",
        dest="inject_port_conflict",
        action="store_true",
        help="hold the first declared loopback port before starting simulators",
    )
    parser.add_argument(
        "--inject-ledger-corruption",
        action="store_true",
        help="corrupt one persisted node content digest before verification",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cluster-diag",
        description="Local deterministic multi-cluster configuration distribution diagnostics",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate the specification as a unit")
    _add_common(validate)

    commit = subparsers.add_parser("commit", help="incrementally commit configuration revisions")
    _add_common(commit, include_budget=True)
    commit.add_argument(
        "--revision",
        default="latest",
        help="'next', 'latest', or a revision version number",
    )

    query = subparsers.add_parser("query", help="perform a measured UI-path query")
    _add_common(query, include_budget=True)
    query.add_argument("--cluster", required=True)
    query.add_argument("--revision", type=int, default=None)

    verify = subparsers.add_parser("verify", help="run authoritative whole-tree recomputation")
    _add_common(verify, include_budget=True)

    recover = subparsers.add_parser("recover", help="restore from the session snapshot and replay")
    _add_common(recover, include_budget=True)
    recover.add_argument("--no-replay", action="store_true")

    run = subparsers.add_parser("run", help="commit, verify, and recover in one local run")
    _add_common(run, include_budget=True)
    run.add_argument("--revision", default="latest", help="'next', 'latest', or a revision number")
    run.add_argument("--no-verify", action="store_true")
    run.add_argument("--no-recover", action="store_true")

    gui = subparsers.add_parser("gui", help="open the Tkinter diagnostic console")
    _add_common(gui, include_budget=True)

    return parser


def _spec_for_args(args: argparse.Namespace) -> Any:
    return load_spec(
        args.spec,
        inject_cycle=bool(args.inject_cycle),
        cycle_code=E_REV,
    )


def _runtime_for_args(args: argparse.Namespace, spec: Any) -> DiagnosticRuntime:
    return DiagnosticRuntime(
        spec=spec,
        workdir=args.workdir,
        latency_budget_ms=getattr(args, "latency_budget_ms", 50),
        inject_port_conflict=bool(args.inject_port_conflict),
        inject_ledger_corruption=bool(args.inject_ledger_corruption),
        defer_semantic_ledger_checks=args.command == "verify",
    )


def _emit(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    sys.stdout.flush()


def _execute(args: argparse.Namespace, logger: StructuredLogger) -> dict[str, Any]:
    if args.command == "validate":
        spec = _spec_for_args(args)
        result = {
            "ok": True,
            "spec": str(spec.source_path),
            "clusters": list(spec.cluster_ids),
            "tasks": [task.id for task in spec.tasks],
            "revisions": [revision.version for revision in spec.revisions],
            "ports": [cluster.port for cluster in spec.clusters],
        }
        logger.info("SPEC_VALID", "specification validated", details={"spec": str(spec.source_path)})
        return result

    spec = _spec_for_args(args)
    if args.command == "gui":
        try:
            from .gui import run_gui
        except ImportError as exc:
            raise DiagError(E_ENV, f"Tkinter is unavailable: {exc}", field_path="tkinter") from exc
        with _runtime_for_args(args, spec) as runtime:
            run_gui(runtime)
            root_revision = runtime.ledger.root_revision()
        return {"ok": True, "gui_closed": True, "root_revision": root_revision}

    with _runtime_for_args(args, spec) as runtime:
        if args.command == "commit":
            if args.revision == "next":
                result = runtime.commit_next()
            else:
                result = runtime.commit_to(args.revision)
            return result
        if args.command == "query":
            return runtime.query(args.cluster, args.revision)
        if args.command == "verify":
            return runtime.authoritative()
        if args.command == "recover":
            return runtime.recover(replay=not args.no_replay)
        if args.command == "run":
            if args.revision == "next":
                commit_result: dict[str, Any] = runtime.commit_next()
            else:
                commit_result = runtime.commit_to(args.revision)
            verification = None if args.no_verify else runtime.authoritative()
            recovery = None if args.no_recover else runtime.recover(replay=True)
            return {
                "ok": True,
                "commit": commit_result,
                "authoritative": verification,
                "recovery": recovery,
                "root_revision": runtime.ledger.root_revision(),
            }
    raise DiagError(E_VIEW, f"unhandled command {args.command!r}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    workdir = Path(args.workdir)
    logger = StructuredLogger(workdir / "logs" / "events.jsonl")
    try:
        result = _execute(args, logger)
    except DiagError as exc:
        logger.error(exc)
        _emit({"ok": False, "command": args.command, "error": exc.as_dict()})
        return EXIT_CODES.get(exc.code, 1)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        logger.emit(
            E_SERVICE,
            "unexpected failure",
            details={"exception_type": type(exc).__name__, "exception": str(exc)},
        )
        _emit(
            {
                "ok": False,
                "command": args.command,
                "error": {
                    "code": E_SERVICE,
                    "message": "unexpected failure",
                    "details": {"exception_type": type(exc).__name__, "exception": str(exc)},
                },
            }
        )
        return EXIT_CODES[E_SERVICE]
    _emit({"ok": True, "command": args.command, "result": result})
    return 0
