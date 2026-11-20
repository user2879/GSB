"""Command-line entry points for scheduling, verification and replay."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from .demo import run_demo
from .errors import BusinessError, E_IO, E_REPLAY, E_SNAPSHOT
from .injection import prepare_injected_inputs, write_digest_mismatch_snapshot
from .manifest import load_manifests
from .persistence import Persistence, StructuredLogger
from .reporting import invalidation_lines, json_line, metrics_summary, pod_lines, print_lines, session_lines
from .restore import replay_payload, restore_payload, session_from_payload
from .scheduler import WorkbenchSession
from .util import pretty_json


DEFAULT_WORKDIR = ".sched-diag"


def _add_manifest_args(parser: argparse.ArgumentParser, *, optional: bool = False) -> None:
    required = not optional
    parser.add_argument("--nodes", required=required, help="node manifest JSON")
    parser.add_argument("--pods", required=required, help="pod queue JSON")
    parser.add_argument("--plugins", default=None, help="optional plugin weight JSON")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m sched_diag")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="load manifests and run a full schedule")
    _add_manifest_args(run_parser)
    run_parser.add_argument("--workdir", default=DEFAULT_WORKDIR)
    run_parser.add_argument("--inject", choices=("none", "manifest", "ref"), default="none")
    run_parser.add_argument("--snapshot", default=None)
    run_parser.add_argument("--json", action="store_true")
    run_parser.add_argument("--selected-only", action="store_true")

    incremental = subparsers.add_parser("incremental", help="full schedule followed by one label mutation")
    _add_manifest_args(incremental)
    incremental.add_argument("--workdir", default=DEFAULT_WORKDIR)
    incremental.add_argument("--inject", choices=("none", "manifest", "ref"), default="none")
    incremental.add_argument("--node", required=True)
    incremental.add_argument("--key", required=True)
    value_group = incremental.add_mutually_exclusive_group(required=True)
    value_group.add_argument("--value")
    value_group.add_argument("--delete", action="store_true")
    incremental.add_argument("--snapshot", default=None)
    incremental.add_argument("--json", action="store_true")

    verify = subparsers.add_parser("verify-incremental", help="assert incremental equals full recomputation")
    _add_manifest_args(verify)
    verify.add_argument("--workdir", default=DEFAULT_WORKDIR)
    verify.add_argument("--node", required=True)
    verify.add_argument("--key", required=True)
    verify_group = verify.add_mutually_exclusive_group(required=True)
    verify_group.add_argument("--value")
    verify_group.add_argument("--delete", action="store_true")
    verify.add_argument("--json", action="store_true")

    graph = subparsers.add_parser("graph", help="validate and print the dependency graph")
    _add_manifest_args(graph)
    graph.add_argument("--workdir", default=DEFAULT_WORKDIR)
    graph.add_argument("--json", action="store_true")

    snapshot = subparsers.add_parser("snapshot", help="schedule and persist a named snapshot")
    _add_manifest_args(snapshot)
    snapshot.add_argument("--workdir", default=DEFAULT_WORKDIR)
    snapshot.add_argument("--name", required=True)
    snapshot.add_argument("--json", action="store_true")

    listing = subparsers.add_parser("list-snapshots", help="list immutable snapshots")
    listing.add_argument("--workdir", default=DEFAULT_WORKDIR)
    listing.add_argument("--json", action="store_true")

    restore = subparsers.add_parser("restore", help="whole-session restore and replay verification")
    restore.add_argument("--workdir", default=DEFAULT_WORKDIR)
    restore_source = restore.add_mutually_exclusive_group(required=True)
    restore_source.add_argument("--name")
    restore_source.add_argument("--snapshot-file")
    restore.add_argument("--revision", type=int, default=None)
    _add_manifest_args(restore, optional=True)
    restore.add_argument("--inject", choices=("none", "snapshot-digest"), default="none")
    restore.add_argument("--json", action="store_true")

    reopen = subparsers.add_parser("reopen", help="restore the current SQLite session and replay it")
    reopen.add_argument("--workdir", default=DEFAULT_WORKDIR)
    reopen.add_argument("--json", action="store_true")

    demo = subparsers.add_parser("demo", help="run the deterministic three-scenario demo")
    demo.add_argument("--workdir", default="artifacts/demo-state")
    demo.add_argument("--keep", action="store_true", help="do not reset the demo work directory")
    demo.add_argument("--json", action="store_true")

    gui = subparsers.add_parser("gui", help="open the Tkinter diagnostic workbench")
    _add_manifest_args(gui, optional=True)
    gui.add_argument("--workdir", default=DEFAULT_WORKDIR)
    gui.add_argument("--inject", choices=("none", "manifest", "ref", "snapshot-digest"), default="none")
    return parser


def _logger_for(workdir: str | Path) -> StructuredLogger:
    return StructuredLogger(Path(workdir) / "logs" / "workbench.jsonl")


def _load_session(
    args: argparse.Namespace,
    persistence: Persistence,
) -> tuple[WorkbenchSession, dict[str, Any]]:
    nodes_path, pods_path, plugins_path, injection = prepare_injected_inputs(
        args.nodes, args.pods, args.plugins, args.workdir, getattr(args, "inject", "none")
    )
    if injection["kind"] != "none":
        persistence.log("WARN", "fault_injection", **injection)
    nodes, pods, plugins = load_manifests(nodes_path, pods_path, plugins_path)
    session = WorkbenchSession(nodes, pods, plugins)
    session.full_schedule()
    persistence.save_current(session)
    persistence.log(
        "INFO",
        "full_schedule",
        digest=session.content_digest(),
        versions=session.versions(),
        pods=len(session.pods),
        stats=len(session.stats),
    )
    return session, injection


def _apply_label_change(session: WorkbenchSession, args: argparse.Namespace) -> dict[str, Any]:
    value = None if args.delete else args.value
    return session.edit_node_label(args.node, args.key, value)


def _render_session(session: WorkbenchSession, *, selected_only: bool = False) -> None:
    print_lines(session_lines(session, title="SCHEDULE", include_all_scores=not selected_only))


def _restore_current_session(args: argparse.Namespace, persistence: Persistence) -> WorkbenchSession | None:
    if getattr(args, "nodes", None) and getattr(args, "pods", None):
        nodes, pods, plugins = load_manifests(args.nodes, args.pods, args.plugins)
        current = WorkbenchSession(nodes, pods, plugins)
        current.full_schedule()
        return current
    payload = persistence.load_current()
    if payload is None:
        return None
    return session_from_payload(payload, source=str(persistence.db_path) + "#/current_session")


def _cmd_run(args: argparse.Namespace) -> int:
    persistence = Persistence(args.workdir)
    session, _ = _load_session(args, persistence)
    verify = session.verify_full_consistency()
    snapshot_summary = None
    if args.snapshot:
        snapshot_summary = persistence.save_snapshot(session, args.snapshot)
    if args.json:
        print(
            json_line(
                {
                    **metrics_summary(session),
                    "verification": verify,
                    "snapshot": snapshot_summary,
                }
            )
        )
    else:
        _render_session(session, selected_only=args.selected_only)
        print(f"VERIFY full ok={verify['ok']} pods={verify['podCount']} stats={verify['statCount']}")
        if snapshot_summary:
            print(
                f"SNAPSHOT name={snapshot_summary['name']} revision={snapshot_summary['revision']} "
                f"backup={snapshot_summary['backupPath']}"
            )
    return 0


def _cmd_incremental(args: argparse.Namespace, *, verify_only: bool = False) -> int:
    persistence = Persistence(args.workdir)
    session, _ = _load_session(args, persistence)
    report = _apply_label_change(session, args)
    verify = session.verify_full_consistency()
    snapshot_summary = None
    if not verify_only and getattr(args, "snapshot", None):
        snapshot_summary = persistence.save_snapshot(session, args.snapshot)
    persistence.save_current(session)
    persistence.log(
        "INFO",
        "incremental_verify" if verify_only else "incremental_schedule",
        digest=session.content_digest(),
        affectedPods=report.get("pods", []),
        affectedStats=len(report.get("stats", [])),
        verification=verify,
    )
    if args.json:
        print(
            json_line(
                {
                    **metrics_summary(session),
                    "invalidation": report,
                    "recomputedPods": session.last_recomputed_pods,
                    "unaffectedCheck": session._unaffected_check,
                    "verification": verify,
                    "snapshot": snapshot_summary,
                }
            )
        )
    else:
        print_lines(invalidation_lines(report))
        print(f"RECOMPUTED_PODS {','.join(session.last_recomputed_pods)}")
        print(f"UNAFFECTED_CHECK {session._unaffected_check}")
        for pod_id in session.last_recomputed_pods:
            print_lines(pod_lines(session, pod_id, include_all_scores=True))
        print(f"VERIFY incremental ok={verify['ok']} pods={verify['podCount']} stats={verify['statCount']}")
        if snapshot_summary:
            print(
                f"SNAPSHOT name={snapshot_summary['name']} revision={snapshot_summary['revision']} "
                f"backup={snapshot_summary['backupPath']}"
            )
    return 0


def _cmd_graph(args: argparse.Namespace) -> int:
    persistence = Persistence(args.workdir)
    nodes, pods, plugins = load_manifests(args.nodes, args.pods, args.plugins)
    session = WorkbenchSession(nodes, pods, plugins)
    summary = session.graph.validate()
    payload = {
        "ok": True,
        **summary,
        "signature": session.graph.signature(),
        "versions": session.versions(),
    }
    persistence.log("INFO", "dependency_graph_validated", **payload)
    print(json_line(payload) if args.json else pretty_json(payload))
    return 0


def _cmd_snapshot(args: argparse.Namespace) -> int:
    persistence = Persistence(args.workdir)
    session, _ = _load_session(args, persistence)
    summary = persistence.save_snapshot(session, args.name)
    print(json_line(summary) if args.json else pretty_json(summary))
    return 0


def _cmd_list_snapshots(args: argparse.Namespace) -> int:
    persistence = Persistence(args.workdir)
    snapshots = persistence.list_snapshots()
    if args.json:
        print(json_line({"snapshots": snapshots}))
    else:
        for item in snapshots:
            print(
                f"{item['name']}@r{item['revision']} digest={item['digest']} "
                f"versions={item['versions']} readOnly={item['readOnly']} path={item['backupPath']}"
            )
        if not snapshots:
            print("NO_SNAPSHOTS")
    return 0


def _load_restore_payload(args: argparse.Namespace, persistence: Persistence) -> tuple[dict[str, Any], str]:
    if args.snapshot_file:
        src = args.snapshot_file
        payload = persistence.load_snapshot_file(src)
    else:
        payload = persistence.load_snapshot(args.name, args.revision)
        src = f"snapshot:{args.name}" + (f"@r{args.revision}" if args.revision is not None else "")
    if args.inject == "snapshot-digest":
        altered = write_digest_mismatch_snapshot(payload, args.workdir)
        persistence.log("WARN", "fault_injection", kind="snapshot-digest", file=str(altered))
        payload = persistence.load_snapshot_file(altered)
        src = str(altered)
    return payload, src


def _cmd_restore(args: argparse.Namespace) -> int:
    persistence = Persistence(args.workdir)
    payload, source = _load_restore_payload(args, persistence)
    current = _restore_current_session(args, persistence)
    restored, report = restore_payload(payload, current_session=current, source=source)
    replay = replay_payload(payload, source=source)
    persistence.save_current(restored)
    persistence.log(
        "INFO",
        "snapshot_restore",
        source=source,
        replaceRequired=report["replaceRequired"],
        changedNodes=report["changedNodes"],
        changedPods=report["changedPods"],
        replay=replay,
    )
    if args.json:
        print(json_line({"restore": report, "replay": replay}))
    else:
        print(
            f"RESTORE replaceRequired={report['replaceRequired']} versionsMatch={report['versionsMatch']} "
            f"digestMatch={report['digestMatch']} replaced={report['replaced']}"
        )
        print(f"RESTORE changedNodes={report['changedNodes']}")
        print(f"RESTORE changedPods={report['changedPods']}")
        print(
            f"REPLAY ok={replay['ok']} invalidationPods={replay['invalidation']['pods']} "
            f"steps={replay['invalidation']['steps']} fullRecomputeMatched={replay['fullRecomputeMatched']}"
        )
        print(f"REPLAY comparedPods={replay['comparedPods']}")
    return 0


def _cmd_reopen(args: argparse.Namespace) -> int:
    persistence = Persistence(args.workdir)
    payload = persistence.load_current()
    if payload is None:
        raise BusinessError(E_SNAPSHOT, "SQLite current session is empty")
    restored, report = restore_payload(payload, current_session=None, source=str(persistence.db_path))
    replay = replay_payload(payload, source=str(persistence.db_path))
    persistence.save_current(restored)
    persistence.log("INFO", "session_reopen", digest=payload["contentDigest"], replay=replay)
    if args.json:
        print(json_line({"restore": report, "replay": replay}))
    else:
        print(f"REOPEN digest={payload['contentDigest']} versions={restored.versions()}")
        print(
            f"REPLAY ok={replay['ok']} comparedPods={replay['comparedPods']} "
            f"fullRecomputeMatched={replay['fullRecomputeMatched']}"
        )
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    if args.json:
        lines: list[str] = []
        result = run_demo(args.workdir, reset=not args.keep, printer=lines.append)
        print(json_line(result))
    else:
        run_demo(args.workdir, reset=not args.keep)
    return 0


def _cmd_gui(args: argparse.Namespace) -> int:
    try:
        from .gui import launch_gui
    except ImportError as exc:
        raise BusinessError(
            E_IO,
            "Tkinter is not available in this Python installation; use the official Windows Python 3.11+ installer",
            details={"error": str(exc)},
        ) from None
    return launch_gui(args)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            return _cmd_run(args)
        if args.command == "incremental":
            return _cmd_incremental(args)
        if args.command == "verify-incremental":
            return _cmd_incremental(args, verify_only=True)
        if args.command == "graph":
            return _cmd_graph(args)
        if args.command == "snapshot":
            return _cmd_snapshot(args)
        if args.command == "list-snapshots":
            return _cmd_list_snapshots(args)
        if args.command == "restore":
            return _cmd_restore(args)
        if args.command == "reopen":
            return _cmd_reopen(args)
        if args.command == "demo":
            return _cmd_demo(args)
        if args.command == "gui":
            return _cmd_gui(args)
        parser.error(f"unknown command {args.command}")
        return 2
    except BusinessError as exc:
        try:
            _logger_for(getattr(args, "workdir", DEFAULT_WORKDIR)).error(
                exc, event="command_failed", command=args.command, argv=list(argv) if argv is not None else sys.argv[1:]
            )
        except Exception:
            pass
        if getattr(args, "json", False):
            print(json_line({"error": exc.to_dict()}), file=sys.stderr)
        else:
            location = f" location={exc.location}" if exc.location else ""
            print(f"ERROR {exc.code} {exc.message}{location}", file=sys.stderr)
            if exc.details:
                print("DETAILS " + json_line(exc.details), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
