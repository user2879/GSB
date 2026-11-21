from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
from pathlib import Path
from typing import Sequence

from .dashboard import render_dashboard
from .errors import WorkbenchError
from .generator import generate_dataset
from .reporting import _atomic_write
from .runner import Workbench, compare_reports, load_report


def _print_json(value: object, *, stream: object = sys.stdout) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), file=stream)


def _cmd_generate(args: argparse.Namespace) -> int:
    manifest = generate_dataset(args.out, seed=args.seed, sample_count=args.samples)
    _print_json(manifest)
    return 0


def _install_stop_handlers(stop_event: threading.Event) -> None:
    def handler(signum: int, _frame: object) -> None:
        stop_event.set()
        print(
            json.dumps(
                {
                    "event": "graceful_stop_requested",
                    "signal": signal.Signals(signum).name,
                    "message": "finishing in-flight samples and committing current snapshot",
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )

    try:
        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)
    except ValueError:
        pass


def _cmd_run(args: argparse.Namespace) -> int:
    stop_event = threading.Event()
    _install_stop_handlers(stop_event)
    try:
        outcome = Workbench(
            args.samples,
            args.out,
            workers=args.workers,
            resume=args.resume,
            expected_version=args.expected_version,
            delay_ms=args.delay_ms,
            stop_after=args.stop_after,
            stop_event=stop_event,
        ).run()
    except WorkbenchError as exc:
        _print_json(
            {
                "ok": False,
                "code": exc.code,
                "stage": exc.stage,
                "exception_type": exc.exception_type,
                "message": exc.detail,
                "output": str(Path(args.out).resolve()),
            }
        )
        return 2
    summary = outcome.report.get("deterministic", {}).get("summary", {})
    snapshot = outcome.report.get("deterministic", {}).get("snapshot", {})
    _print_json(
        {
            "ok": outcome.exit_code == 0,
            "status": outcome.status,
            "exit_code": outcome.exit_code,
            "report": str(outcome.report_path.resolve()),
            "dashboard": str(outcome.dashboard_path.resolve()),
            "deterministic_digest": outcome.report.get("deterministic_digest"),
            "summary": summary,
            "snapshot": snapshot,
        }
    )
    return outcome.exit_code


def _cmd_compare(args: argparse.Namespace) -> int:
    comparison = compare_reports(args.left, args.right)
    if args.out:
        payload = (json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
        _atomic_write(Path(args.out), payload)
    _print_json(comparison)
    return 0 if comparison["equal"] else 1


def _cmd_dashboard(args: argparse.Namespace) -> int:
    report = load_report(args.report)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(output, render_dashboard(report).encode("utf-8"))
    _print_json({"ok": True, "dashboard": str(output.resolve())})
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
    import functools

    directory = Path(args.directory).resolve()
    if not directory.exists():
        _print_json({"ok": False, "message": f"directory does not exist: {directory}"})
        return 2
    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(directory))
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"serving {directory} at http://{args.host}:{args.port}/dashboard.html", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ner-workbench",
        description="Deterministic local Chinese NER scheduling and resource-budget diagnostics",
    )
    parser.add_argument("--version", action="version", version="ner-workbench 0.1.0")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate deterministic local samples")
    generate.add_argument("--out", type=Path, required=True)
    generate.add_argument("--seed", type=int, default=20261121)
    generate.add_argument("--samples", type=int, default=32)
    generate.set_defaults(func=_cmd_generate)

    run = subparsers.add_parser("run", help="run the workbench")
    run.add_argument("--samples", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--workers", type=int, default=None, help="override configured thread-pool size")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--expected-version", type=int, default=None)
    run.add_argument("--delay-ms", type=int, default=0, help="diagnostic per-task delay for interruption drills")
    run.add_argument("--stop-after", type=int, default=None, help="diagnostic graceful stop after N tasks")
    run.set_defaults(func=_cmd_run)

    compare = subparsers.add_parser("compare", help="compare deterministic report sections")
    compare.add_argument("left", type=Path)
    compare.add_argument("right", type=Path)
    compare.add_argument("--out", type=Path, default=None)
    compare.set_defaults(func=_cmd_compare)

    dashboard = subparsers.add_parser("dashboard", help="render a standalone HTML dashboard from a report")
    dashboard.add_argument("--report", type=Path, required=True)
    dashboard.add_argument("--out", type=Path, required=True)
    dashboard.set_defaults(func=_cmd_dashboard)

    serve = subparsers.add_parser("serve", help="serve a generated result directory locally")
    serve.add_argument("--directory", type=Path, required=True)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.set_defaults(func=_cmd_serve)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except WorkbenchError as exc:
        _print_json(
            {
                "ok": False,
                "code": exc.code,
                "stage": exc.stage,
                "exception_type": exc.exception_type,
                "message": exc.detail,
            }
        )
        return 2
    except ValueError as exc:
        _print_json({"ok": False, "code": "E_INPUT_INVALID", "message": str(exc)})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
