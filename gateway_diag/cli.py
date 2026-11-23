from __future__ import annotations

import argparse
import json
import sqlite3
import traceback
from pathlib import Path
from typing import Any

from .backend_client import BackendClient
from .backend_runtime import BackendRuntime
from .compare import validate_incremental_vs_full
from .errors import E_IO, E_SNAPSHOT, E_SPEC, GatewayError
from .ledger import Ledger
from .loaders import load_spec
from .logging_utils import StructuredLogger, default_log_path
from .snapshot import inspect_snapshot, load_snapshot
from .verify import verify_ledger
from .workflow import run_session


def _add_spec_args(parser: argparse.ArgumentParser, *, config_required: bool = False) -> None:
    group = parser.add_mutually_exclusive_group(required=config_required)
    group.add_argument("--config", help="combined backends+rules JSON")
    group.add_argument("--backends", help="standalone backends JSON")
    parser.add_argument("--rules", help="standalone rules JSON (used with --backends)")
    parser.add_argument("--requests", required=True, help="NDJSON request stream")


def _add_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--spawn-backends",
        action="store_true",
        help="launch the shipped deterministic mock on declared loopback ports",
    )
    parser.add_argument("--backend-timeout", type=float, default=15.0)
    parser.add_argument(
        "--crash-after-batches",
        type=int,
        default=None,
        help="fault injection: os._exit(137) after N committed batches",
    )


def _load(args: argparse.Namespace, *, config: str | None = None, backends: str | None = None, rules: str | None = None):
    return load_spec(
        config_path=config if config is not None else getattr(args, "config", None),
        backends_path=backends if backends is not None else getattr(args, "backends", None),
        rules_path=rules if rules is not None else getattr(args, "rules", None),
        requests_path=args.requests,
    )


def _logger_for(workdir: str | Path, log_file: str | None = None) -> StructuredLogger:
    path = Path(log_file) if log_file else default_log_path(workdir)
    return StructuredLogger(path)


def _runtime(args: argparse.Namespace, specs, logger: StructuredLogger):
    if not args.spawn_backends:
        return _NullRuntime()
    backends = []
    for spec in specs:
        backends.extend(spec.backends.values())
    return BackendRuntime(
        backends,
        workdir=args.workdir,
        logger=logger,
        timeout=args.backend_timeout,
    )


class _NullRuntime:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None


def _emit(payload: dict[str, Any], *, pretty: bool) -> None:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2 if pretty else None)
    print(text, flush=True)


def _inject_config_source(kind: str, config_path: str, workdir: Path) -> Path:
    source = Path(config_path)
    injections = workdir / "injections"
    injections.mkdir(parents=True, exist_ok=True)
    if kind == "spec":
        target = injections / "broken-spec.json"
        target.write_text('{"batch_limit": 4, "backends": [', encoding="utf-8")
        return target
    if kind == "ref":
        target = injections / "dangling-reference.json"
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise GatewayError(E_SPEC, "cannot prepare reference injection", field="config", details={"reason": str(exc)})
        rules = payload.get("rules") if isinstance(payload, dict) else None
        if not isinstance(rules, list) or not rules:
            raise GatewayError(E_SPEC, "reference injection requires a combined config with rules", field="rules")
        first_rule = rules[0]
        if not isinstance(first_rule, dict):
            first_rule = {}
            rules[0] = first_rule
        first_rule["canary_backend"] = "__missing_backend_reference__"
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return target
    raise GatewayError(E_SPEC, f"unknown config injection kind {kind!r}", field="inject")


def _tamper_ledger_digest(workdir: str | Path) -> Path:
    path = Path(workdir) / "ledger.sqlite3"
    if not path.exists():
        raise GatewayError(E_SNAPSHOT, "ledger does not exist for digest injection", field="ledger", entry=str(path))
    try:
        conn = sqlite3.connect(path)
        with conn:
            conn.execute(
                "UPDATE meta SET value=? WHERE key='content_digest'",
                ("sha256:" + "0" * 64,),
            )
        conn.close()
    except sqlite3.DatabaseError as exc:
        raise GatewayError(E_SNAPSHOT, "cannot inject ledger digest fault", field="content_digest", details={"reason": str(exc)})
    return path


def cmd_run(args: argparse.Namespace) -> dict[str, Any]:
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    logger = _logger_for(workdir, args.log_file)
    if args.inject == "ledger":
        _tamper_ledger_digest(workdir)
        logger.emit("fault_injected", level="WARNING", code=E_SNAPSHOT, field="content_digest")
        # Opening the ledger below is the observable failure path.
    config_path = args.config
    if args.inject in {"spec", "ref"}:
        if not args.config:
            raise GatewayError(E_SPEC, "config injection requires --config", field="inject")
        config_path = str(_inject_config_source(args.inject, args.config, workdir))
        logger.emit("fault_injected", level="WARNING", code=E_SPEC if args.inject == "spec" else "E_REF", field="config")
    spec = load_spec(
        config_path=config_path,
        backends_path=args.backends,
        rules_path=args.rules,
        requests_path=args.requests,
    )
    if args.crash_after_batches is not None and args.crash_after_batches <= 0:
        raise GatewayError(E_SPEC, "--crash-after-batches must be positive", field="crash_after_batches")
    client = BackendClient(timeout=args.backend_timeout)
    with _runtime(args, [spec], logger):
        return run_session(
            spec,
            workdir=workdir,
            client=client,
            logger=logger,
            crash_after_batches=args.crash_after_batches,
        )


def cmd_validate(args: argparse.Namespace) -> dict[str, Any]:
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    logger = _logger_for(workdir, args.log_file)
    before = load_spec(config_path=args.before, requests_path=args.requests)
    after = load_spec(config_path=args.after, requests_path=args.requests)
    client = BackendClient(timeout=args.backend_timeout)
    with _runtime(args, [before, after], logger):
        return validate_incremental_vs_full(
            before=before,
            after=after,
            workdir=workdir,
            client=client,
            logger=logger,
        )


def cmd_verify(args: argparse.Namespace) -> dict[str, Any]:
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    logger = _logger_for(workdir, args.log_file)
    spec = load_spec(
        config_path=args.config,
        backends_path=args.backends,
        rules_path=args.rules,
        requests_path=args.requests,
    )
    client = BackendClient(timeout=args.backend_timeout)
    with _runtime(args, [spec], logger):
        return verify_ledger(
            spec,
            workdir=workdir,
            client=client,
            logger=logger,
        )


def cmd_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    spec = load_spec(
        config_path=args.config,
        backends_path=args.backends,
        rules_path=args.rules,
        requests_path=args.requests,
    )
    ledger = Ledger(workdir, read_only=True)
    try:
        info = inspect_snapshot(workdir, ledger, spec)
    finally:
        ledger.close()
    snapshot = load_snapshot(workdir)
    return {"ok": True, "command": "snapshot", "comparison": info, "snapshot": snapshot}


def cmd_inject(args: argparse.Namespace) -> dict[str, Any]:
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    logger = _logger_for(workdir, args.log_file)
    if args.kind == "ledger-digest":
        path = _tamper_ledger_digest(workdir)
        logger.emit("fault_injected", level="WARNING", code=E_SNAPSHOT, field="content_digest")
        return {
            "ok": True,
            "command": "inject",
            "kind": args.kind,
            "target": str(path),
            "expected_code": E_SNAPSHOT,
        }
    if not args.config or not args.output:
        raise GatewayError(E_SPEC, "--config and --output are required for declaration injections", field="inject")
    if args.kind == "declaration":
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('{"batch_limit": 4, "backends": [', encoding="utf-8")
        expected = E_SPEC
        source = Path(args.config)
    elif args.kind == "reference":
        source = Path(args.config)
        target = Path(args.output)
        payload = json.loads(source.read_text(encoding="utf-8"))
        rules = payload.get("rules") if isinstance(payload, dict) else None
        if not isinstance(rules, list) or not rules:
            raise GatewayError(E_SPEC, "reference injection requires a combined config with rules", field="rules")
        if not isinstance(rules[0], dict):
            rules[0] = {}
        rules[0]["canary_backend"] = "__missing_backend_reference__"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        expected = "E_REF"
    else:
        raise GatewayError(E_SPEC, f"unknown injection kind {args.kind}", field="kind")
    logger.emit(
        "fault_injected",
        level="WARNING",
        code=expected,
        field="config",
        source=str(source),
        target=str(target),
    )
    return {
        "ok": True,
        "command": "inject",
        "kind": args.kind,
        "source": str(source),
        "target": str(target),
        "expected_code": expected,
    }


def cmd_gui(args: argparse.Namespace) -> dict[str, Any]:
    try:
        from .gui import launch_gui

        return launch_gui(
            config=args.config,
            requests=args.requests,
            workdir=args.workdir,
        )
    except ImportError as exc:
        raise GatewayError(
            "E_GUI",
            "tkinter is unavailable in this Python installation",
            field="tkinter",
            details={"reason": str(exc)},
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gateway-diag",
        description="Incremental routing diagnostics for a local deterministic inference gateway",
    )
    parser.add_argument("--pretty", action="store_true", help="pretty-print JSON output")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="create, update, or recover a session")
    _add_spec_args(run)
    run.add_argument("--workdir", required=True)
    run.add_argument("--log-file")
    run.add_argument(
        "--inject",
        choices=["spec", "ref", "ledger"],
        help="inject a declaration, dangling-reference, or ledger-digest fault",
    )
    _add_runtime_args(run)
    run.set_defaults(handler=cmd_run)

    validate = subparsers.add_parser("validate", help="compare incremental update to a full rebuild")
    validate.add_argument("--before", required=True, help="baseline combined JSON")
    validate.add_argument("--after", required=True, help="updated combined JSON")
    validate.add_argument("--requests", required=True)
    validate.add_argument("--workdir", required=True)
    validate.add_argument("--log-file")
    _add_runtime_args(validate)
    validate.set_defaults(handler=cmd_validate)

    verify = subparsers.add_parser("verify", help="independently verify a persisted ledger")
    _add_spec_args(verify)
    verify.add_argument("--workdir", required=True)
    verify.add_argument("--log-file")
    _add_runtime_args(verify)
    verify.set_defaults(handler=cmd_verify)

    snapshot = subparsers.add_parser("snapshot", help="inspect a session snapshot")
    _add_spec_args(snapshot)
    snapshot.add_argument("--workdir", required=True)
    snapshot.add_argument("--log-file")
    snapshot.set_defaults(handler=cmd_snapshot)

    inject = subparsers.add_parser("inject", help="create a fault-injected input")
    inject.add_argument("--kind", required=True, choices=["declaration", "reference", "ledger-digest"])
    inject.add_argument("--config")
    inject.add_argument("--output")
    inject.add_argument("--workdir", required=True)
    inject.add_argument("--log-file")
    inject.set_defaults(handler=cmd_inject)

    gui = subparsers.add_parser("gui", help="optional tkinter diagnostic dashboard")
    gui.add_argument("--config")
    gui.add_argument("--requests")
    gui.add_argument("--workdir", default=".gateway-diag")
    gui.set_defaults(handler=cmd_gui)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logger: StructuredLogger | None = None
    try:
        if getattr(args, "workdir", None):
            logger = _logger_for(args.workdir, getattr(args, "log_file", None))
            logger.emit("command_started", command=args.command)
        result = args.handler(args)
        if isinstance(result, dict) and "ok" not in result:
            result = {"ok": True, **result}
        if result is not None:
            _emit(result, pretty=args.pretty)
        if logger is not None:
            logger.emit("command_completed", command=args.command)
        return 0
    except GatewayError as exc:
        if logger is not None:
            logger.error(exc, event="business_error")
        payload = exc.as_dict()
        if logger is not None:
            payload["log"] = str(logger.path)
        _emit(payload, pretty=args.pretty)
        return 2
    except KeyboardInterrupt:
        if logger is not None:
            logger.emit("command_interrupted", level="WARNING")
        _emit({"ok": False, "code": "E_INTERRUPTED", "message": "interrupted"}, pretty=args.pretty)
        return 130
    except Exception as exc:
        if logger is not None:
            logger.emit(
                "internal_error",
                level="ERROR",
                code=E_IO,
                exception_type=type(exc).__name__,
                message=str(exc),
                traceback=traceback.format_exc(limit=20),
            )
        _emit(
            {
                "ok": False,
                "code": E_IO,
                "message": f"internal error: {type(exc).__name__}: {exc}",
                "log": str(logger.path) if logger is not None else None,
            },
            pretty=args.pretty,
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
