"""Command-line entry points for validation, operations, and verification."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

from .engine import DiagnosticEngine
from .errors import GatewayError
from .ledger import LedgerStore
from .model import load_requests, load_spec, request_stream_digest
from .parameters import generate_parameters
from .routing import plan_routes

EXIT_CODES = {
    "E_SPEC": 2,
    "E_PORT": 3,
    "E_HANDSHAKE": 4,
    "E_RECEIPT": 5,
    "E_PREVIEW": 6,
    "E_REPLAY": 7,
    "E_SCHEDULER": 8,
    "E_LEDGER": 9,
}


def _atomic_write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    handle, temporary = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".tmp", dir=str(destination.parent)
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _build_engine(args: argparse.Namespace, store: LedgerStore) -> tuple[DiagnosticEngine, Any, Any]:
    spec = load_spec(args.spec)
    requests = load_requests(args.requests, spec)
    engine = DiagnosticEngine(
        spec,
        requests,
        store,
        startup_timeout=args.startup_timeout,
        receipt_timeout=args.receipt_timeout,
    )
    return engine, spec, requests


def _command_validate(args: argparse.Namespace) -> dict[str, Any]:
    spec = load_spec(args.spec)
    parameters = {
        backend.id: generate_parameters(backend).breakdown()
        for backend in sorted(spec.backends, key=lambda item: item.id)
    }
    result: dict[str, Any] = {
        "ok": True,
        "operation": "validate",
        "revision": spec.revision,
        "spec_fingerprint": spec.fingerprint,
        "parameter_digests": parameters,
        "rules": {
            rule.id: {
                "version": rule.version,
                "content_digest": rule.content_digest,
            }
            for rule in sorted(spec.rules, key=lambda item: item.id)
        },
    }
    if args.requests:
        requests = load_requests(args.requests, spec)
        plan = plan_routes(spec, requests)
        result.update(
            {
                "request_count": len(requests),
                "request_stream_digest": request_stream_digest(requests),
                "batch_count": len(plan.batches),
                "stable_count": sum(
                    decision.target == "stable" for decision in plan.decisions
                ),
                "canary_count": sum(
                    decision.target == "canary" for decision in plan.decisions
                ),
            }
        )
    return result


def _command_parameters(args: argparse.Namespace) -> dict[str, Any]:
    spec = load_spec(args.spec)
    return {
        "ok": True,
        "operation": "parameters",
        "revision": spec.revision,
        "backends": {
            backend.id: generate_parameters(backend).breakdown()
            for backend in sorted(spec.backends, key=lambda item: item.id)
        },
    }


def _command_run(args: argparse.Namespace) -> dict[str, Any]:
    with LedgerStore(args.db) as store:
        engine, _spec, _requests = _build_engine(args, store)
        result = engine.run().to_dict()
    if args.report:
        _atomic_write_json(args.report, result)
    return result


def _command_preview(args: argparse.Namespace) -> dict[str, Any]:
    with LedgerStore(args.db) as store:
        cache = store.load_preview_cache()
        if args.request_id is not None:
            entry = cache.get(args.request_id)
            if entry is None:
                raise GatewayError(
                    f"request {args.request_id} is absent from the active preview",
                    code="E_PREVIEW",
                    request_id=args.request_id,
                    field="presence",
                )
            return {
                "ok": True,
                "operation": "preview",
                "revision": cache.revision,
                "entry": entry.to_dict(),
                "stats": cache.stats,
            }
        result = cache.snapshot.to_dict()
        result.update({"ok": True, "operation": "preview"})
        return result


def _command_versions(args: argparse.Namespace) -> dict[str, Any]:
    with LedgerStore(args.db) as store:
        history = store.version_history()
        latest = history[-1]["revision"] if history else None
        return {
            "ok": True,
            "operation": "versions",
            "active_revision": latest,
            "history": history,
            "current_diff": store.activation_diff() if latest is not None else None,
        }


def _command_verify(args: argparse.Namespace) -> dict[str, Any]:
    with LedgerStore(args.db) as store:
        engine, _spec, _requests = _build_engine(args, store)
        preview = engine.preview() if args.mode in ("authority", "both") else None
        authority = (
            engine.verify_authoritative(preview).to_dict()
            if args.mode in ("authority", "both")
            else None
        )
        replay = (
            engine.verify_replay().to_dict()
            if args.mode in ("replay", "both")
            else None
        )
        return {
            "ok": True,
            "operation": "verify",
            "mode": args.mode,
            "authority": authority,
            "replay": replay,
        }


def _command_recover(args: argparse.Namespace) -> dict[str, Any]:
    with LedgerStore(args.db) as store:
        engine, _spec, _requests = _build_engine(args, store)
        result = engine.recover().to_dict()
    if args.report:
        _atomic_write_json(args.report, result)
    return result


def _command_gui(args: argparse.Namespace) -> dict[str, Any]:
    try:
        from .gui import launch_gui
    except ImportError as exc:
        raise GatewayError(
            "Tkinter is unavailable in this Python installation; use the Windows python.org installer or install the OS Tk runtime",
            code="E_UI",
            field="tkinter",
            details={"exception": type(exc).__name__, "message": str(exc)},
        ) from exc

    launch_gui(
        spec_path=args.spec,
        requests_path=args.requests,
        db_path=args.db,
    )
    return {"ok": True, "operation": "gui", "closed": True}


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--spec", required=True, help="path to the domain JSON spec")
    parser.add_argument("--requests", required=True, help="path to the NDJSON request stream")
    parser.add_argument("--db", required=True, help="path to the SQLite ledger")
    parser.add_argument(
        "--startup-timeout", type=float, default=8.0, help="backend readiness timeout"
    )
    parser.add_argument(
        "--receipt-timeout", type=float, default=20.0, help="per-batch receipt timeout"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gateway-diag",
        description="Deterministic loopback inference gateway diagnostics",
    )
    parser.add_argument("--debug", action="store_true", help="show Python tracebacks")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="strictly validate spec and requests")
    validate.add_argument("--spec", required=True)
    validate.add_argument("--requests")
    validate.set_defaults(handler=_command_validate)

    parameters = subparsers.add_parser("parameters", help="show parameter summaries")
    parameters.add_argument("--spec", required=True)
    parameters.set_defaults(handler=_command_parameters)

    run = subparsers.add_parser("run", help="execute or resume the current revision")
    _add_runtime_arguments(run)
    run.add_argument("--report", help="write the JSON report atomically")
    run.set_defaults(handler=_command_run)

    preview = subparsers.add_parser("preview", help="read the cached quick view")
    preview.add_argument("--db", required=True)
    preview.add_argument("--request-id", type=int)
    preview.set_defaults(handler=_command_preview)

    verify = subparsers.add_parser("verify", help="run authoritative and/or replay checks")
    _add_runtime_arguments(verify)
    verify.add_argument(
        "--mode", choices=("authority", "replay", "both"), default="both"
    )
    verify.set_defaults(handler=_command_verify)

    recover = subparsers.add_parser("recover", help="mark unknowns, rerun, and replay-verify")
    _add_runtime_arguments(recover)
    recover.add_argument("--report", help="write the JSON report atomically")
    recover.set_defaults(handler=_command_recover)

    versions = subparsers.add_parser("versions", help="show version replacement history")
    versions.add_argument("--db", required=True)
    versions.set_defaults(handler=_command_versions)

    gui = subparsers.add_parser("gui", help="open the interactive Tk application")
    gui.add_argument("--spec")
    gui.add_argument("--requests")
    gui.add_argument("--db")
    gui.set_defaults(handler=_command_gui)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = args.handler(args)
    except GatewayError as exc:
        if args.debug:
            raise
        _print(exc.to_dict())
        return EXIT_CODES.get(exc.code, 1)
    except Exception as exc:
        if args.debug:
            raise
        _print(
            {
                "ok": False,
                "code": "E_INTERNAL",
                "message": str(exc),
                "exception": type(exc).__name__,
            }
        )
        return 1
    _print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
