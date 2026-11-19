"""Command-line interface for arbitrary local declaration and script files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .declaration import FrameScript, SessionDeclaration, load_declaration, load_script
from .demo import run_demo
from .errors import ToolError
from .replay import replay_from_snapshot
from .session import H2Session
from .snapshot import SnapshotStore


ROOT = Path(__file__).resolve().parent.parent


def _load_inputs(declaration_path: str, script_path: str) -> tuple[SessionDeclaration, FrameScript]:
    declaration = load_declaration(declaration_path)
    script = load_script(script_path, declaration)
    return declaration, script


def _dump(value: Any, path: str | None) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="h2mux-cli",
        description="Offline HTTP/2 multiplexing state and replay verifier",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="strictly load and summarize inputs")
    validate.add_argument("--declaration", default=str(ROOT / "samples" / "session.json"))
    validate.add_argument("--script", default=str(ROOT / "samples" / "frames.json"))
    validate.add_argument("--output")

    run = subparsers.add_parser("run", help="advance a session and write the ledger")
    run.add_argument("--declaration", default=str(ROOT / "samples" / "session.json"))
    run.add_argument("--script", default=str(ROOT / "samples" / "frames.json"))
    run.add_argument("--strategy", choices=("weighted_round_robin", "window_priority"), default="weighted_round_robin")
    run.add_argument("--snapshot-at", type=int)
    run.add_argument("--snapshot", help="snapshot path; used with --snapshot-at")
    run.add_argument("--output")

    replay = subparsers.add_parser("replay", help="restore a snapshot and verify the remaining frame interval")
    replay.add_argument("--declaration", default=str(ROOT / "samples" / "session.json"))
    replay.add_argument("--script", default=str(ROOT / "samples" / "frames.json"))
    replay.add_argument("--snapshot", required=True)
    replay.add_argument("--strategy", choices=("weighted_round_robin", "window_priority"))
    replay.add_argument("--output")

    demo = subparsers.add_parser("demo", help="run the three built-in deterministic scenarios")
    demo.add_argument("--output-dir", default=str(ROOT / "artifacts" / "demo"))
    demo.add_argument("--output", help="optional summary output path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "demo":
            summary = run_demo(Path(args.output_dir))
            _dump(summary, args.output)
            return 0 if summary["all_checks_passed"] else 1
        declaration, script = _load_inputs(args.declaration, args.script)
        if args.command == "validate":
            _dump(
                {
                    "code": "OK",
                    "declaration_version": declaration.declaration_version,
                    "declaration_digest": declaration.content_digest,
                    "script_digest": script.source_digest,
                    "script_events": len(script.events),
                    "request_stream_ids": [item.stream_id for item in declaration.requests],
                },
                args.output,
            )
            return 0
        if args.command == "run":
            if args.snapshot_at is not None and not args.snapshot:
                raise ToolError("E_DECL", "--snapshot is required with --snapshot-at", path="--snapshot")
            session = H2Session(declaration, script, strategy=args.strategy)
            if args.snapshot_at is not None:
                session.run(stop_after_events=args.snapshot_at)
                snapshot_path = Path(args.snapshot)
                SnapshotStore(snapshot_path.parent).save(
                    session,
                    snapshot_path.name,
                    label=f"CLI stage at event {session.script_index}",
                )
            session.run()
            _dump(
                {
                    "code": "OK",
                    "summary": session.summary(),
                    "ledger": session.ledger.to_dict(),
                    "diagnostics": session.diagnostics(),
                },
                args.output,
            )
            return 0
        if args.command == "replay":
            report = replay_from_snapshot(
                args.snapshot,
                script,
                current_declaration=declaration,
                current_strategy=args.strategy,
            )
            _dump(report.to_dict(), args.output)
            return 0 if report.status in {"MATCH", "TRUNCATED"} else 1
        raise AssertionError(args.command)
    except ToolError as exc:
        sys.stderr.write(json.dumps(exc.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
