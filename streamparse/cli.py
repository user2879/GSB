from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Optional, TextIO

from .app import IncrementalApp
from .errors import AppError
from .reporting import write_report_atomic


def _emit(value: Any, stream: TextIO = sys.stdout) -> None:
    stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False))
    stream.write("\n")
    stream.flush()


def _command_lines(
    *,
    command: Optional[str],
    script: Optional[str],
    interactive: bool,
) -> Iterable[str]:
    if command:
        yield command
    if script:
        with open(script, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield line
    if interactive:
        for line in sys.stdin:
            if line.strip():
                yield line


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Budget-aware deterministic streaming instruction parser (JSONL commands)"
    )
    parser.add_argument("--sample-dir", default="sample_data")
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument("--command", help="one JSON command")
    parser.add_argument("--script", help="JSONL command file")
    parser.add_argument("--interactive", action="store_true", help="read JSONL commands from stdin")
    parser.add_argument("--prompt", action="store_true", help="write prompts to stderr")
    args = parser.parse_args(argv)

    try:
        app = IncrementalApp(args.sample_dir, workspace_root=args.workspace_root)
    except AppError as exc:
        _emit({"status": "error", "error": exc.as_failure()}, sys.stdout)
        return 2

    _emit(
        {
            "event": "ready",
            "global_version": app.latest_version,
            "sessions": sorted(app.dataset.sessions),
            "summary": app.summary(),
            "report_path": str(app.report_path),
        }
    )

    try:
        for raw in _command_lines(command=args.command, script=args.script, interactive=args.interactive):
            if args.prompt:
                sys.stderr.write("streamparse> ")
                sys.stderr.flush()
            try:
                command = json.loads(raw)
            except json.JSONDecodeError as exc:
                _emit(
                    {
                        "status": "error",
                        "business_codes": ["E_COMMAND_INVALID"],
                        "error": {"message": f"invalid command JSON: {exc.msg}", "line": exc.lineno},
                    }
                )
                continue
            if not isinstance(command, dict):
                _emit(
                    {
                        "status": "error",
                        "business_codes": ["E_COMMAND_INVALID"],
                        "error": {"message": "command must be a JSON object"},
                    }
                )
                continue
            response = app.execute(command)
            _emit(response)
            if str(command.get("cmd", "")).lower() in ("quit", "exit"):
                break
    finally:
        try:
            final = app.wait()
            if app.latest_report is not None:
                write_report_atomic(app.report_path, app.latest_report)
            _emit(
                {
                    "event": "diagnostic_report_written",
                    "path": str(app.report_path),
                    "global_version": app.latest_version,
                    "status": final.get("status"),
                }
            )
        finally:
            app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
