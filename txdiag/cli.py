"""Command line entry point for the transaction diagnostics workbench."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from .canonical import canonical_text, digest
from .engine import ExecutionResult, compare_canonical, execute
from .errors import DivergenceError, ReplayError, SpecError, TxDiagError
from .ledger import Ledger, ReplayHooks, declaration_digest, read_session_metadata
from .model import ISOLATION_ALIASES, RuntimeConfig, RangeThresholds, choose_lock_path
from .spec import load_spec, spec_summary


EXIT_CODES = {
    "OK": 0,
    "E_SPEC": 2,
    "E_TIMEOUT": 3,
    "E_DEADLOCK": 4,
    "E_DIVERGE": 5,
    "E_REPLAY": 6,
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="txdiag",
        description="Deterministic transaction isolation and lock-wait diagnostics",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="load a declaration, execute it, and persist the ledger")
    _add_common(run)
    run.add_argument("--path", choices=("auto", "row", "range", "both"), default="auto",
                     help="lock path; both writes two ledgers and compares canonical bytes")
    run.add_argument("--range-threshold", default="tx=8,rows=12,edges=10,total=24",
                     help="deterministic auto-range thresholds")
    run.add_argument("--trace-out", help="canonical round trace output path")

    replay = sub.add_parser("replay", help="re-execute committed rounds and compare them")
    _add_common(replay)
    replay.add_argument("--path", choices=("auto", "row", "range"), default=None,
                        help="ignored when the ledger records a path")
    replay.add_argument("--range-threshold", default="tx=8,rows=12,edges=10,total=24")

    recover = sub.add_parser("recover", help="inspect committed rounds and unknown inflight work")
    _add_common(recover)

    validate = sub.add_parser("validate", help="strictly validate a declaration only")
    _add_common(validate)

    gui = sub.add_parser("gui", help="launch the Tkinter diagnostics workbench")
    _add_common(gui)
    gui.add_argument("--path", choices=("auto", "row", "range"), default="auto")
    gui.add_argument("--range-threshold", default="tx=8,rows=12,edges=10,total=24")
    gui.add_argument("--check", action="store_true", help="open and destroy a Tk root")
    return parser


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--spec", required=True, help="transaction declaration JSON")
    parser.add_argument("--ops", help="external operation NDJSON (overrides operations_file)")
    parser.add_argument("--ledger", default="ledger/txdiag.sqlite", help="SQLite ledger path")
    parser.add_argument("--snapshot",
                        help="whole-session snapshot path (default: beside the ledger)")
    parser.add_argument("--max-running", type=int, default=4, help="global worker/transaction limit")
    parser.add_argument("--quota", action="append", default=[],
                        help="per-isolation quota, e.g. read_committed=2")
    parser.add_argument("--wait-timeout-ms", type=int, default=2000)
    parser.add_argument("--wait-budget-ms", type=int, default=10000)
    parser.add_argument("--inject", action="append", default=[],
                        help="fault injection: crash-after-round=N, crash-in-round=N, "
                             "hold-lock=TX:N, diverge, replay-tamper")
    parser.add_argument("--report", help="write a JSON run/replay report")
    parser.add_argument("--quiet", action="store_true")


def _quotas(items: list[str], max_running: int) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in items:
        name, sep, value = item.partition("=")
        if not sep:
            raise SpecError("quota must be isolation=count", txid="<config>", field="quota")
        canonical = ISOLATION_ALIASES.get(name.strip().lower())
        if canonical is None:
            raise SpecError("quota names an unknown isolation level", txid="<config>",
                            field="quota")
        try:
            count = int(value)
        except ValueError as exc:
            raise SpecError("quota count must be an integer", txid="<config>",
                            field="quota") from exc
        if count < 1:
            raise SpecError("quota count must be at least 1", txid="<config>",
                            field="quota")
        out[canonical] = count
    return out


def _thresholds(value: str) -> RangeThresholds:
    if not value:
        return RangeThresholds()
    values = {"tx": 8, "rows": 12, "edges": 10, "total": 24}
    for part in value.split(","):
        key, sep, raw = part.partition("=")
        if not sep or key not in values:
            raise SpecError("invalid range threshold item", txid="<config>",
                            field="range_threshold")
        try:
            values[key] = int(raw)
        except ValueError as exc:
            raise SpecError("range threshold must be an integer", txid="<config>",
                            field="range_threshold") from exc
    if any(value < 0 for value in values.values()):
        raise SpecError("range thresholds must be non-negative", txid="<config>",
                        field="range_threshold")
    return RangeThresholds(values["tx"], values["rows"], values["edges"], values["total"])


def _config(args: argparse.Namespace) -> RuntimeConfig:
    if args.max_running < 1:
        raise SpecError("max-running must be at least 1", txid="<config>", field="max_running")
    if args.wait_timeout_ms < 0 or args.wait_budget_ms < 0:
        raise SpecError("wait budgets must be non-negative", txid="<config>",
                        field="wait_timeout_ms")
    return RuntimeConfig(
        max_running=args.max_running,
        quotas=_quotas(args.quota, args.max_running),
        wait_timeout_ms=args.wait_timeout_ms,
        wait_budget_ms=args.wait_budget_ms,
        lock_path=getattr(args, "path", "auto") or "auto",
        thresholds=_thresholds(getattr(args, "range_threshold", "")),
        inject=tuple(args.inject),
    )


def _write_json(path: str | None, value: Any) -> None:
    if not path:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    os.replace(tmp, target)


def _snapshot_path(args: argparse.Namespace) -> str:
    if getattr(args, "snapshot", None):
        return str(args.snapshot)
    ledger = Path(args.ledger)
    return str(ledger.with_name(ledger.stem + ".snapshot.json"))


def _trace_document(result: ExecutionResult) -> dict[str, Any]:
    return {
        "format_version": 1,
        "status": result.status,
        "rounds": result.rounds,
        "final": {
            "transactions": result.result["transactions"],
            "final_versions": result.result["final_versions"],
            "final_locks": result.result["final_locks"],
            "deadlocks": [
                {key: item[key] for key in (
                    "round", "logical_ms", "cycle", "wait_graph_edges", "victim", "basis"
                )}
                for item in result.result["deadlocks"]
            ],
            "timeouts": [
                {key: item[key] for key in (
                    "round", "logical_ms", "txid", "row", "mode", "wait_ms", "code"
                )}
                for item in result.result["timeouts"]
            ],
        },
    }


def _run_single(args: argparse.Namespace, spec: Any, config: RuntimeConfig, path: str,
                ledger_path: str, snapshot_path: str) -> tuple[ExecutionResult, dict[str, Any]]:
    ledger = Ledger(ledger_path, snapshot_path, spec, config, path)
    opened = ledger.open()
    try:
        result = execute(spec, config, path, hooks=ledger)
        ledger.complete(result)
        session = ledger.session_info()
    finally:
        ledger.close()
    report = {
        "command": "run",
        "status": result.status,
        "path": path,
        "spec": spec_summary(spec),
        "config": config.as_dict(),
        "ledger": opened,
        "session": session,
        "execution": {
            "round_count": len(result.rounds),
            "canonical_sha256": digest(result.rounds),
            "worker_threads": result.worker_threads,
            "deadlocks": result.deadlocks,
            "timeouts": result.timeouts,
            "wait_budget": result.result["wait_budget"],
            "transactions": result.result["transactions"],
        },
        "isolations": result.isolation_conclusions,
        "range_plan_rounds": result.range_plan_rounds,
    }
    report["report_sha256"] = digest({k: v for k, v in report.items() if k != "report_sha256"})
    return result, report


def _run_both(args: argparse.Namespace, spec: Any, config: RuntimeConfig) -> dict[str, Any]:
    ledger = Path(args.ledger)
    snapshot = Path(_snapshot_path(args))
    row_ledger = str(ledger.with_name(ledger.stem + ".row" + ledger.suffix))
    range_ledger = str(ledger.with_name(ledger.stem + ".range" + ledger.suffix))
    row_snapshot = str(snapshot.with_name(snapshot.stem + ".row" + snapshot.suffix))
    range_snapshot = str(snapshot.with_name(snapshot.stem + ".range" + snapshot.suffix))
    row_cfg = RuntimeConfig(**{**config.__dict__, "lock_path": "row"})
    range_cfg = RuntimeConfig(**{**config.__dict__, "lock_path": "range"})
    row_result, row_report = _run_single(args, spec, row_cfg, "row", row_ledger, row_snapshot)
    range_result, range_report = _run_single(args, spec, range_cfg, "range", range_ledger,
                                             range_snapshot)
    comparison = compare_canonical(row_result, range_result)
    if not comparison["equal"]:
        raise DivergenceError(
            "row and range lock paths produced different canonical trajectories",
            txid=comparison.get("first_txid"), row=comparison.get("first_row"),
            field=comparison.get("first_field"), details=comparison,
        )
    row_trace = _trace_document(row_result)
    range_trace = _trace_document(range_result)
    if args.trace_out:
        base = Path(args.trace_out)
        _write_json(str(base.with_name(base.stem + ".row" + base.suffix)), row_trace)
        _write_json(str(base.with_name(base.stem + ".range" + base.suffix)), range_trace)
    return {
        "command": "run",
        "status": "COMPLETED_PATHS_MATCH" if row_result.status == "COMPLETED"
                  else row_result.status,
        "path": "both",
        "comparison": comparison,
        "row": row_report,
        "range": range_report,
        "canonical_bytes_equal": comparison["equal"],
    }


def _command_run(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    spec = load_spec(args.spec, args.ops)
    config = _config(args)
    if args.path == "both":
        report = _run_both(args, spec, config)
        if not report["canonical_bytes_equal"]:
            code = EXIT_CODES["E_DIVERGE"]
        elif report["row"]["execution"]["timeouts"]:
            code = EXIT_CODES["E_TIMEOUT"]
        else:
            code = EXIT_CODES["OK"]
        return code, report
    selected, selection = choose_lock_path(spec, config)
    single_cfg = RuntimeConfig(**{**config.__dict__, "lock_path": selected})
    result, report = _run_single(args, spec, single_cfg, selected, args.ledger,
                                 _snapshot_path(args))
    report["path_selection"] = selection
    if args.trace_out:
        _write_json(args.trace_out, _trace_document(result))
    report["trace_out"] = args.trace_out
    code = EXIT_CODES["E_TIMEOUT"] if result.timeouts else EXIT_CODES["OK"]
    return code, report


def _recorded_config(meta: dict[str, Any], fallback: RuntimeConfig) -> RuntimeConfig:
    raw = meta.get("config_json")
    if not isinstance(raw, dict):
        return fallback
    thresholds = raw.get("thresholds", {})
    if not isinstance(thresholds, dict):
        thresholds = {}
    return RuntimeConfig(
        max_running=int(raw.get("max_running", fallback.max_running)),
        quotas={str(k): int(v) for k, v in raw.get("quotas", {}).items()},
        wait_timeout_ms=int(raw.get("wait_timeout_ms", fallback.wait_timeout_ms)),
        wait_budget_ms=int(raw.get("wait_budget_ms", fallback.wait_budget_ms)),
        lock_path=str(meta.get("effective_path") or fallback.lock_path),
        thresholds=RangeThresholds(
            int(thresholds.get("tx_count", fallback.thresholds.tx_count)),
            int(thresholds.get("row_keys", fallback.thresholds.row_keys)),
            int(thresholds.get("dependency_edges", fallback.thresholds.dependency_edges)),
            int(thresholds.get("total", fallback.thresholds.total)),
        ),
        inject=(),
        logical_tick_ms=int(raw.get("logical_tick_ms", fallback.logical_tick_ms)),
    )


def _command_replay(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    spec = load_spec(args.spec, args.ops)
    fallback = _config(args)
    meta = read_session_metadata(args.ledger)
    if not meta:
        raise ReplayError("ledger does not exist or is empty", field="ledger")
    config = _recorded_config(meta, fallback)
    path = str(meta.get("effective_path") or "")
    if path not in ("row", "range"):
        path = choose_lock_path(spec, config)[0]
    ledger = Ledger(args.ledger, _snapshot_path(args), spec, config, path)
    opened = ledger.open(allow_replacement=False)
    info = ledger.session_info()
    snapshot = ledger.read_snapshot() or {}
    snapshot_state = snapshot.get("state", {}) if isinstance(snapshot, dict) else {}
    expected_final = snapshot_state.get("result") if info["complete"] else None
    hooks = ReplayHooks(ledger.expected_rounds, complete=info["complete"],
                        tamper="replay-tamper" in args.inject,
                        expected_final=expected_final)
    try:
        result = execute(spec, config, path, hooks=hooks)
    finally:
        ledger.close()
    summary = hooks.summary()
    complete = bool(info["complete"])
    status = "REPLAY_OK" if complete else "INCOMPLETE_REQUIRES_RERUN"
    report = {
        "command": "replay",
        "status": status,
        "path": path,
        "spec": spec_summary(spec),
        "ledger": opened,
        "session": info,
        "replay": summary,
        "exit_report_available": complete,
        "unknown_txids": [] if complete else info.get("unknown_txids", []),
        "note": None if complete else
                "only committed rounds were replayed; inflight work is unknown and must be rerun",
    }
    report["report_sha256"] = digest({k: v for k, v in report.items() if k != "report_sha256"})
    code = EXIT_CODES["OK"] if complete else EXIT_CODES["E_REPLAY"]
    return code, report


def _command_recover(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    spec = load_spec(args.spec, args.ops)
    fallback = _config(args)
    meta = read_session_metadata(args.ledger)
    if not meta:
        return EXIT_CODES["OK"], {
            "command": "recover", "status": "NO_LEDGER", "spec": spec_summary(spec)
        }
    config = _recorded_config(meta, fallback)
    path = str(meta.get("effective_path") or choose_lock_path(spec, config)[0])
    ledger = Ledger(args.ledger, _snapshot_path(args), spec, config, path)
    try:
        opened = ledger.open(allow_replacement=False)
        info = ledger.session_info()
    finally:
        ledger.close()
    report = {
        "command": "recover",
        "status": "COMPLETE" if info["complete"] else "INCOMPLETE_REQUIRES_RERUN",
        "ledger": opened,
        "session": info,
        "exit_report_available": info["complete"] and not info["rerun_required"],
    }
    code = EXIT_CODES["OK"] if info["complete"] else EXIT_CODES["E_REPLAY"]
    return code, report


def _command_validate(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    spec = load_spec(args.spec, args.ops)
    return EXIT_CODES["OK"], {
        "command": "validate", "status": "SPEC_OK", "spec": spec_summary(spec)
    }


def _command_gui(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    from .gui import launch
    spec = load_spec(args.spec, args.ops)
    config = _config(args)
    selected, selection = choose_lock_path(spec, config)
    config = RuntimeConfig(**{**config.__dict__, "lock_path": selected})
    launch(spec, config, args.path if args.path != "auto" else selected, args.ledger,
           _snapshot_path(args), args.check)
    return EXIT_CODES["OK"], {
        "command": "gui", "status": "GUI_OK" if args.check else "GUI_CLOSED",
        "spec": spec_summary(spec), "path": config.lock_path,
        "path_selection": selection,
    }


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    # A forced termination must never leave a stale earlier report that could be
    # mistaken for a report from the interrupted session.
    if args.report:
        Path(args.report).unlink(missing_ok=True)
    try:
        if args.command == "run":
            code, report = _command_run(args)
        elif args.command == "replay":
            code, report = _command_replay(args)
        elif args.command == "recover":
            code, report = _command_recover(args)
        elif args.command == "validate":
            code, report = _command_validate(args)
        elif args.command == "gui":
            code, report = _command_gui(args)
        else:
            raise SpecError("unknown command", txid="<cli>", field="command")
    except TxDiagError as exc:
        report = {"command": args.command, "status": exc.code, "error": exc.as_dict()}
        code = EXIT_CODES.get(exc.code, 1)
    except Exception as exc:  # deterministic machine-readable failure surface
        report = {
            "command": args.command,
            "status": "E_INTERNAL",
            "error": {"code": "E_INTERNAL", "message": str(exc),
                      "type": type(exc).__name__},
        }
        code = 1
    _write_json(args.report, report)
    if not args.quiet:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
