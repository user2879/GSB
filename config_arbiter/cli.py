from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .engine import ArbiterEngine
from .errors import AppError, ReplayError
from .ledger import Ledger
from .spec import load_spec


DEFAULT_SPEC = "samples/sample_spec.json"
DEFAULT_LEDGER = "artifacts/arbiter-ledger.sqlite3"


def _write_report(path: str | None, payload: dict[str, Any]) -> None:
    if not path:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _reset_ledger(path: str | Path) -> None:
    base = Path(path)
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(str(base) + suffix)
        if candidate.exists():
            candidate.unlink()
    index = 1
    while True:
        candidate = base.with_name(f"{base.name}.replaced.{index}")
        if not candidate.exists():
            break
        candidate.unlink()
        index += 1


def _print_view_results(report: dict[str, Any]) -> None:
    for run in report.get("views", []):
        summary = run["summary"]
        print(
            f"VIEW {summary['view_id']} v{summary['version']} "
            f"digest={summary['summary_digest'][:16]}"
        )
        for key in summary["keys"]:
            amount = key.get("summary", {}).get("decimal")
            bound = key.get("error_bound", {}).get("decision_bound_units")
            cause = key.get("attribution", {}).get("primary_cause")
            print(
                f"  {key['key']:<16} {key['mode']:<18} value={str(amount):<12} "
                f"adopted={','.join(key['adopted']) or '-':<24} "
                f"bound={bound:<6} cause={cause}"
            )


def _run(spec_path: str, ledger_path: str, report_path: str | None, reset: bool) -> int:
    if reset:
        _reset_ledger(ledger_path)
    spec = load_spec(spec_path)
    ledger = Ledger(ledger_path)
    engine: ArbiterEngine | None = None
    try:
        engine = ArbiterEngine(spec, ledger)
        engine.start_services()
        run_report = engine.run_all()
        replay_report = engine.replay_current()
        payload = {
            "ok": True,
            "mode": "run",
            "spec": str(spec.source_path),
            "ledger_path": str(ledger.path),
            "open_report": ledger.open_report.to_dict(),
            "run": run_report.to_dict(),
            "replay": replay_report,
            "snapshot": ledger.snapshot_report(),
            "current_display": engine.current_display(),
        }
        _print_view_results(payload["run"])
        print(
            f"REPLAY OK keys={replay_report['view_key_count']} "
            f"fields={replay_report['field_comparisons']} "
            f"content_digest={replay_report['content_digest'][:16]}"
        )
        _write_report(report_path, payload)
        return 0
    finally:
        if engine is not None:
            engine.close()
        else:
            ledger.close()


def _replay(spec_path: str, ledger_path: str, report_path: str | None) -> int:
    spec = load_spec(spec_path)
    ledger = Ledger(ledger_path)
    engine: ArbiterEngine | None = None
    try:
        engine = ArbiterEngine(spec, ledger)
        engine.start_services()
        replay_report = engine.replay_current()
        payload = {
            "ok": True,
            "mode": "replay",
            "replay": replay_report,
            "snapshot": ledger.snapshot_report(),
            "current_display": engine.current_display(),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        _write_report(report_path, payload)
        return 0
    finally:
        if engine is not None:
            engine.close()
        else:
            ledger.close()


def _demo(spec_path: str, ledger_path: str, report_path: str | None) -> int:
    _reset_ledger(ledger_path)
    spec = load_spec(spec_path)
    ledger = Ledger(ledger_path)
    engine: ArbiterEngine | None = None
    try:
        engine = ArbiterEngine(spec, ledger)
        engine.start_services()
        initial_run = engine.run_all()
        initial_replay = engine.replay_current()
        cn_current = engine.current_slices["cn-prod"]
        mutation = engine.change_environment_slice(
            "cn-prod",
            version=cn_current.version + 1,
            values={
                "cpu_limit": "1.3001",
                "memory_limit": "7.5",
                "replicas": "4",
            },
        )
        post_mutation_replay = engine.replay_current()
        current_display = engine.current_display()
        stale_rows = [row for row in ledger.all_results() if int(row["is_stale"]) == 1]
        mutated_environments = dict(spec.environments)
        mutated_environments["cn-prod"] = replace(
            mutated_environments["cn-prod"],
            slices=mutated_environments["cn-prod"].slices
            + (engine.current_slices["cn-prod"],),
        )
        mutated_spec = replace(
            spec,
            environments=MappingProxyType(mutated_environments),
        )
        engine.close()
        engine = None
        ledger = Ledger(ledger_path)
        engine = ArbiterEngine(mutated_spec, ledger)
        engine.start_services()
        restart_replay = engine.replay_current()
        restart_snapshot = ledger.snapshot_report()
        payload = {
            "ok": True,
            "mode": "demo",
            "spec": str(spec.source_path),
            "ledger_path": str(ledger.path),
            "open_report": ledger.open_report.to_dict(),
            "initial_run": initial_run.to_dict(),
            "initial_replay": initial_replay,
            "mutation": [item.to_dict() for item in mutation],
            "post_mutation_replay": post_mutation_replay,
            "restart_open_report": ledger.open_report.to_dict(),
            "restart_replay": restart_replay,
            "restart_snapshot": restart_snapshot,
            "current_display": current_display,
            "stale_result_count": len(stale_rows),
            "snapshot": ledger.snapshot_report(),
        }
        print("== initial arbitration ==")
        _print_view_results(payload["initial_run"])
        print(
            f"initial replay: keys={initial_replay['view_key_count']} "
            f"fields={initial_replay['field_comparisons']}"
        )
        print("== invalidation after cn-prod slice change ==")
        for item in payload["mutation"]:
            print(
                f"VIEW {item['view_id']} {item['previous_version']} -> "
                f"{item['new_version']} affected={','.join(item['affected_keys'])}"
            )
            for reason in item["reasons"]:
                print(f"  {json.dumps(reason, ensure_ascii=False, sort_keys=True)}")
        print(
            f"post-mutation replay: keys={post_mutation_replay['view_key_count']} "
            f"fields={post_mutation_replay['field_comparisons']}; "
            f"stale rows={len(stale_rows)}"
        )
        print(
            f"reopened ledger replay: keys={restart_replay['view_key_count']} "
            f"fields={restart_replay['field_comparisons']}; "
            f"replaced={ledger.open_report.replaced}"
        )
        _write_report(report_path, payload)
        return 0
    finally:
        if engine is not None:
            engine.close()
        else:
            ledger.close()


def _gui(spec_path: str, ledger_path: str) -> int:
    from .gui import launch_gui

    launch_gui(spec_path, ledger_path)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="config-arbiter",
        description="Configuration authority arbitration and invalidation diagnostics",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run all views once and verify replay")
    run.add_argument("--spec", default=DEFAULT_SPEC)
    run.add_argument("--ledger", default=DEFAULT_LEDGER)
    run.add_argument("--report")
    run.add_argument("--reset", action="store_true")

    replay = subparsers.add_parser("replay", help="verify the current ledger by recomputation")
    replay.add_argument("--spec", default=DEFAULT_SPEC)
    replay.add_argument("--ledger", default=DEFAULT_LEDGER)
    replay.add_argument("--report")

    demo = subparsers.add_parser("demo", help="run the bundled acceptance scenario")
    demo.add_argument("--spec", default=DEFAULT_SPEC)
    demo.add_argument("--ledger", default="artifacts/demo-ledger.sqlite3")
    demo.add_argument("--report", default="artifacts/acceptance_report.json")

    gui = subparsers.add_parser("gui", help="open the optional Tk diagnostics desktop")
    gui.add_argument("--spec", default=DEFAULT_SPEC)
    gui.add_argument("--ledger", default=DEFAULT_LEDGER)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            return _run(args.spec, args.ledger, args.report, args.reset)
        if args.command == "replay":
            return _replay(args.spec, args.ledger, args.report)
        if args.command == "demo":
            return _demo(args.spec, args.ledger, args.report)
        if args.command == "gui":
            return _gui(args.spec, args.ledger)
        parser.error(f"unknown command {args.command!r}")
        return 2
    except ReplayError as exc:
        print(
            json.dumps({"ok": False, "error": exc.to_dict()}, ensure_ascii=False, sort_keys=True),
            file=sys.stderr,
        )
        return 3
    except AppError as exc:
        print(
            json.dumps({"ok": False, "error": exc.to_dict()}, ensure_ascii=False, sort_keys=True),
            file=sys.stderr,
        )
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
