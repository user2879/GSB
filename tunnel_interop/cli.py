"""Command-line interface for validation, forwarding, evolution and recovery."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .errors import E_FIELD, E_FORMAT, E_REF, E_REPLAY, E_SNAPSHOT, E_SPEC, E_TIMEOUT, TunnelError
from .fragmenter import build_bundle
from .logging_utils import StructuredLogger
from .spec import load_spec
from .util import atomic_write_json
from .workflow import (
    evolution_report,
    forward_bundle,
    inject_anomaly,
    interop_report,
    resume_session,
    verify_all,
)


EXIT_CODES = {
    E_SPEC: 2,
    E_FORMAT: 2,
    E_FIELD: 2,
    E_REF: 2,
    E_TIMEOUT: 3,
    E_SNAPSHOT: 4,
    E_REPLAY: 5,
}


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--log-file", help="append structured JSON-lines events to this file")
    parser.add_argument("--out", help="write the command result as canonical JSON to this path")


def _emit(result: Any, out: str | None = None) -> None:
    if out:
        atomic_write_json(out, result)
    sys.stdout.write(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def _spec_summary(spec) -> dict[str, Any]:
    return {
        "ok": True,
        "source_path": spec.source_path,
        "versions": spec.versions.canonical(),
        "endpoint_count": len(spec.endpoints),
        "route_count": len(spec.routes),
        "fragment_policy": spec.fragment_policy.canonical(),
        "effective_payload_limit": spec.effective_payload_limit,
        "runtime": spec.runtime.canonical(),
        "fingerprints": {
            "endpoints": spec.endpoint_fingerprint(),
            "routes": spec.route_fingerprint(),
            "fragment_policy": spec.fragment_fingerprint(),
            "declaration": spec.declaration_fingerprint(),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tunnel-interop", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="load and strongly validate a local JSON declaration")
    validate.add_argument("--spec", required=True)
    _add_common(validate)

    interop = subparsers.add_parser("interop", help="exercise f1/f2/f3 normalization and migration")
    interop.add_argument("--spec", required=True)
    _add_common(interop)

    forward = subparsers.add_parser("forward", help="send real loopback UDP datagrams and commit ledger batches")
    forward.add_argument("--spec", required=True)
    forward.add_argument("--ledger", required=True)
    forward.add_argument("--scenario", choices=("normal", "reverse", "duplicate"), default="normal")
    forward.add_argument("--format", choices=("f1", "f2", "f3"))
    forward.add_argument("--message", action="append", dest="messages", help="message key to forward; repeatable")
    forward.add_argument("--ack-mode", choices=("normal", "missing-field"), default="normal")
    forward.add_argument("--drop-index", action="append", type=int, default=[], dest="drop_indices")
    forward.add_argument("--drop-all", action="store_true")
    forward.add_argument(
        "--crash-after-batch",
        type=int,
        help="commit an intent for this zero-based batch then terminate abruptly with os._exit(97)",
    )
    _add_common(forward)

    evolve = subparsers.add_parser("evolve", help="compute dependency-precise invalidation between two declarations")
    evolve.add_argument("--old", required=True)
    evolve.add_argument("--new", required=True)
    _add_common(evolve)

    resume = subparsers.add_parser("resume", help="open a ledger, replace an incompatible snapshot, and replay commits")
    resume.add_argument("--spec", required=True)
    resume.add_argument("--ledger", required=True)
    resume.add_argument("--rerun", action="store_true", help="rerun unknown uncommitted batches over real UDP")
    resume.add_argument("--scenario", choices=("normal", "reverse", "duplicate"), default="normal")
    _add_common(resume)

    inject = subparsers.add_parser("inject", help="create one of the three auditable anomaly inputs")
    inject.add_argument(
        "--kind",
        required=True,
        choices=("corrupt-spec", "dangling-route", "snapshot-digest"),
    )
    inject.add_argument("--spec", required=True)
    inject.add_argument("--ledger")
    inject.add_argument("--output-dir", required=True)
    _add_common(inject)

    verify = subparsers.add_parser("verify", help="run the full local acceptance matrix")
    verify.add_argument("--spec", required=True)
    verify.add_argument("--evolved-spec")
    verify.add_argument("--output-dir", required=True)
    verify.add_argument("--no-udp", action="store_true")
    _add_common(verify)

    serve = subparsers.add_parser("serve", help="serve one declared 127.0.0.1 UDP endpoint as a subprocess")
    serve.add_argument("--endpoint-id", required=True)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", required=True, type=int)
    serve.add_argument("--ack-mode", choices=("normal", "missing-field"), default="normal")
    serve.add_argument("--drop-indices", default="")
    serve.add_argument("--drop-all", action="store_true")
    serve.add_argument("--idle-timeout-ms", type=int, default=10_000)
    serve.add_argument("--ready-file", required=True)
    serve.add_argument("--log-file")

    return parser


def run_command(args: argparse.Namespace) -> Any:
    if args.command == "validate":
        return _spec_summary(load_spec(args.spec))
    if args.command == "interop":
        spec = load_spec(args.spec)
        result = interop_report(build_bundle(spec))
        return result
    if args.command == "forward":
        spec = load_spec(args.spec)
        bundle = build_bundle(spec, target_format=args.format)
        logger = StructuredLogger(args.log_file)
        return forward_bundle(
            spec=spec,
            bundle=bundle,
            ledger_path=args.ledger,
            scenario=args.scenario,
            logger=logger,
            message_keys=args.messages,
            ack_mode=args.ack_mode,
            drop_indices=set(args.drop_indices),
            drop_all=args.drop_all,
            crash_after_batch=args.crash_after_batch,
        )
    if args.command == "evolve":
        old_spec = load_spec(args.old)
        new_spec = load_spec(args.new)
        return evolution_report(old_spec, new_spec)
    if args.command == "resume":
        spec = load_spec(args.spec)
        logger = StructuredLogger(args.log_file)
        return resume_session(
            spec=spec,
            ledger_path=args.ledger,
            logger=logger,
            rerun=args.rerun,
            scenario=args.scenario,
        )
    if args.command == "inject":
        logger = StructuredLogger(args.log_file)
        logger.event("anomaly_injection_started", kind=args.kind, spec=args.spec, ledger=args.ledger)
        return inject_anomaly(
            kind=args.kind,
            spec_path=args.spec,
            output_dir=args.output_dir,
            ledger_path=args.ledger,
            logger=logger,
        )
    if args.command == "verify":
        logger = StructuredLogger(args.log_file)
        return verify_all(
            spec_path=args.spec,
            evolved_spec_path=args.evolved_spec,
            output_dir=args.output_dir,
            with_udp=not args.no_udp,
            logger=logger,
        )
    if args.command == "serve":
        from .udp_receiver import main as receiver_main

        command = [
            "--endpoint-id",
            args.endpoint_id,
            "--host",
            args.host,
            "--port",
            str(args.port),
            "--ack-mode",
            args.ack_mode,
            "--drop-indices",
            args.drop_indices,
            "--idle-timeout-ms",
            str(args.idle_timeout_ms),
            "--ready-file",
            args.ready_file,
        ]
        if args.drop_all:
            command.append("--drop-all")
        if args.log_file:
            command.extend(["--log-file", args.log_file])
        return receiver_main(command)
    raise AssertionError(f"unhandled command {args.command}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logger = StructuredLogger(getattr(args, "log_file", None))
    try:
        result = run_command(args)
        if args.command == "serve":
            return int(result)
        logger.event("command_ok", command=args.command)
        _emit(result, getattr(args, "out", None))
        return 0
    except TunnelError as exc:
        payload = exc.to_dict()
        payload["command"] = args.command
        logger.event(
            "command_error",
            command=args.command,
            code=exc.code,
            path=exc.path,
            message=exc.message,
            details=dict(exc.details),
        )
        _emit(payload, getattr(args, "out", None))
        return EXIT_CODES.get(exc.code, 1)
    except KeyboardInterrupt:
        payload = {
            "ok": False,
            "error": {"code": "E_INTERRUPTED", "path": "$", "message": "interrupted", "details": {}},
            "command": args.command,
        }
        logger.event("command_interrupted", command=args.command)
        _emit(payload, getattr(args, "out", None))
        return 130
    except SystemExit:
        raise
    except Exception as exc:
        payload = {
            "ok": False,
            "error": {
                "code": "E_INTERNAL",
                "path": "$",
                "message": "unexpected internal failure",
                "details": {"exception": type(exc).__name__, "repr": repr(exc)},
            },
            "command": args.command,
        }
        logger.event("command_internal_error", command=args.command, exception=type(exc).__name__, repr=repr(exc))
        _emit(payload, getattr(args, "out", None))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
