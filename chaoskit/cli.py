"""Command-line entry point for the chaos component (stdlib only)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

from .engine import ChaosEngine
from .errors import ChaosError
from .gui import launch, render_headless
from .ledger import LEDGER_FILENAME
from .replay import verify_identity, verify_timeline_entries
from .spec import load_spec
from .timeline import build_timeline, timeline_digest


def _print(payload: Dict[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False))


def cmd_validate(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec)
    _print({
        "ok": True,
        "spec_digest": spec.spec_digest,
        "services": sorted(spec.services),
        "injection_count": len(spec.injections),
    })
    return 0


def cmd_timeline(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec)
    entries = build_timeline(spec)
    _print({
        "seed": spec.seed,
        "timeline_digest": timeline_digest(entries),
        "entries": [entry.to_dict() for entry in entries],
    })
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec)
    events = []
    engine = ChaosEngine(spec, args.workdir, on_event=events.append)
    try:
        engine.setup()
        report = engine.run()
    finally:
        engine.teardown()
    _print({"events": events, "report": report})
    return 0


def cmd_recover(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec)
    engine = ChaosEngine(spec, args.workdir)
    try:
        engine.setup()
        report = engine.ledger.recovery_report(spec)
        continue_report = engine.run()
    finally:
        engine.teardown()
    _print({"recovery": report, "continued": continue_report})
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    from .ledger import Ledger
    from .replay import verify_observations

    spec = load_spec(args.spec)
    ledger = Ledger(Path(args.workdir) / LEDGER_FILENAME)
    try:
        verify_identity(spec, ledger)
        verify_timeline_entries(spec, ledger)
        if args.observations:
            replay_summaries = _load_replay_observations(args.replay_workdir or args.workdir)
            verify_observations(
                ledger, replay_summaries, spec.observation.sample_interval_ms
            )
        _print({"ok": True, "verified_committed": ledger.committed_indexes()})
    finally:
        ledger.close()
    return 0


def _load_replay_observations(workdir: str) -> Dict[int, list]:
    import sqlite3
    conn = sqlite3.connect(str(Path(workdir) / LEDGER_FILENAME))
    conn.row_factory = sqlite3.Row
    try:
        result: Dict[int, list] = {}
        rows = conn.execute("SELECT * FROM observations ORDER BY id").fetchall()
        for row in rows:
            result.setdefault(row["injection_idx"], []).append(json.loads(row["summary_json"]))
        return result
    finally:
        conn.close()


def cmd_gui(args: argparse.Namespace) -> int:
    ledger_path = str(Path(args.workdir) / LEDGER_FILENAME)
    if args.headless:
        print(render_headless(ledger_path))
        return 0
    launch(ledger_path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chaoskit", description="Local deterministic chaos engineering")
    sub = parser.add_subparsers(dest="command", required=True)

    p_validate = sub.add_parser("validate", help="strong-validate an experiment declaration")
    p_validate.add_argument("spec")
    p_validate.set_defaults(func=cmd_validate)

    p_timeline = sub.add_parser("timeline", help="print the deterministic timeline")
    p_timeline.add_argument("spec")
    p_timeline.set_defaults(func=cmd_timeline)

    p_run = sub.add_parser("run", help="run an experiment and write the ledger")
    p_run.add_argument("spec")
    p_run.add_argument("--workdir", required=True)
    p_run.set_defaults(func=cmd_run)

    p_recover = sub.add_parser("recover", help="resume from a committed ledger and finish")
    p_recover.add_argument("spec")
    p_recover.add_argument("--workdir", required=True)
    p_recover.set_defaults(func=cmd_recover)

    p_replay = sub.add_parser("replay", help="verify replay against the ledger")
    p_replay.add_argument("spec")
    p_replay.add_argument("--workdir", required=True)
    p_replay.add_argument("--replay-workdir")
    p_replay.add_argument("--observations", action="store_true")
    p_replay.set_defaults(func=cmd_replay)

    p_gui = sub.add_parser("gui", help="open the Tkinter diagnostic window")
    p_gui.add_argument("spec")
    p_gui.add_argument("--workdir", required=True)
    p_gui.add_argument("--headless", action="store_true")
    p_gui.set_defaults(func=cmd_gui)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ChaosError as exc:
        print(json.dumps(exc.to_dict(), sort_keys=True, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
