"""Command line interface for the deterministic rendering workbench."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys

from .config import load_config
from .dag import build_graph
from .errors import WorkbenchError
from .injection import load_injections
from .orchestrator import Orchestrator
from .reporting import compare_runs, export_graph_dot, write_report
from .store import SnapshotStore


def _resolve(base_dir: str, path: str | None) -> str | None:
    if path is None:
        return None
    return path if os.path.isabs(path) else os.path.normpath(
        os.path.join(base_dir, path)
    )


def _load(path: str):
    config = load_config(path)
    base_dir = os.path.dirname(os.path.abspath(path))
    # Re-resolve relative asset paths against the config file directory.
    import dataclasses

    config = dataclasses.replace(
        config,
        vocab_file=_resolve(base_dir, config.vocab_file),
        initial_image=_resolve(base_dir, config.initial_image),
        source_path=os.path.abspath(path),
    )
    return config, base_dir


def _install_signals(orch: Orchestrator) -> None:
    def handler(signum, frame):  # pragma: no cover - exercised via subprocess
        orch.request_shutdown()

    for name in ("SIGINT", "SIGTERM"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), handler)


def cmd_run(args: argparse.Namespace) -> int:
    config, base_dir = _load(args.config)
    injection_path = (
        os.path.abspath(args.injections)
        if args.injections and not os.path.isabs(args.injections)
        else args.injections
    )
    injections = load_injections(injection_path)
    store = SnapshotStore(args.store)
    output_png = args.output or os.path.join(args.store, "final.png")
    orch = Orchestrator(config, store, injections, output_png=output_png,
                        jitter=args.jitter)
    _install_signals(orch)
    try:
        report = orch.run()
    except WorkbenchError as exc:
        payload = {"schema": "dbench.report.v1", "fatal": exc.to_dict()}
        write_report(payload, args.report)
        print(json.dumps({"error": exc.to_dict()}, ensure_ascii=False))
        return 2
    report["store"] = os.path.abspath(args.store)
    report["output_png"] = os.path.abspath(output_png)
    report["injection_file"] = args.injections
    write_report(report, args.report)
    graph_path = args.graph or os.path.join(os.path.dirname(args.report),
                                            "dependency_graph.dot")
    export_graph_dot(report["dependency_graph"], graph_path)
    summary = {
        "report": os.path.abspath(args.report),
        "graph": os.path.abspath(graph_path),
        "png": os.path.abspath(output_png),
        "reuse": report["reuse"],
        "recompute": report["recompute"],
        "failures": report["failures"],
        "blocked_count": len(report["blocked"]),
        "injection_hits": report["injection_hits"],
        "recovery": report["recovery"],
        "shutdown": report["shutdown"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not report["failures"] else 3


def cmd_compare(args: argparse.Namespace) -> int:
    with open(args.report_a, encoding="utf-8") as handle:
        ra = json.load(handle)
    with open(args.report_b, encoding="utf-8") as handle:
        rb = json.load(handle)
    result = compare_runs(ra, rb, args.label_a, args.label_b)
    write_report(result, args.out)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["equivalent"] else 4


def cmd_verify(args: argparse.Namespace) -> int:
    config, _ = _load(args.config)
    store = SnapshotStore(args.store)
    nodes = build_graph(config)
    problems = []
    for stage in nodes:
        try:
            store.check_continuity(stage)
        except WorkbenchError as exc:
            problems.append(exc.to_dict())
    result = {"ok": not problems, "problems": problems}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not problems else 5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dbench",
                                     description="Deterministic rendering workbench")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run (or incrementally update) a render")
    run.add_argument("--config", required=True)
    run.add_argument("--store", required=True)
    run.add_argument("--report", required=True)
    run.add_argument("--injections")
    run.add_argument("--output")
    run.add_argument("--graph")
    run.add_argument("--jitter", action="store_true",
                     help="add scheduling jitter (testing only)")
    run.set_defaults(func=cmd_run)

    cmp_ = sub.add_parser("compare", help="compare two diagnostic reports")
    cmp_.add_argument("--report-a", required=True)
    cmp_.add_argument("--report-b", required=True)
    cmp_.add_argument("--label-a", default="incremental")
    cmp_.add_argument("--label-b", default="full")
    cmp_.add_argument("--out", required=True)
    cmp_.set_defaults(func=cmd_compare)

    verify = sub.add_parser("verify-snapshots",
                            help="check snapshot version continuity")
    verify.add_argument("--config", required=True)
    verify.add_argument("--store", required=True)
    verify.set_defaults(func=cmd_verify)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except WorkbenchError as exc:
        print(json.dumps({"error": exc.to_dict()}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    sys.exit(main())
