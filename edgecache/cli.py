"""Command-line interface for validation, demo, diagnostics and workers."""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import socket
from pathlib import Path
from typing import Any, Sequence

from .errors import EdgeError
from .gui import build_view_model, run_gui
from .ledger import Ledger
from .orchestrator import PROJECT_ROOT, EdgeCluster
from .snapshot import SnapshotStore
from .spec import load_spec


SAMPLE_V1 = PROJECT_ROOT / "samples" / "demo_spec.json"
SAMPLE_V2 = PROJECT_ROOT / "samples" / "demo_spec_v2.json"
SAMPLE_V3 = PROJECT_ROOT / "samples" / "demo_spec_v3.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m edgecache.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="strongly validate a local specification")
    validate.add_argument("--spec", required=True)

    demo = subparsers.add_parser("demo", help="run the deterministic three-scenario demonstration")
    demo.add_argument("--workspace", default=str(PROJECT_ROOT / "runtime" / "demo"))
    demo.add_argument("--port-base", type=int, default=46000)
    demo.add_argument("--handshake-timeout", type=float, default=2.0)
    demo.add_argument("--gui", action="store_true", help="open the Tkinter workbench after the run")
    demo.add_argument(
        "--inject",
        choices=["content_truncation", "cycle", "snapshot_digest_mismatch"],
        help="inject one of the required deterministic anomaly classes",
    )

    gui = subparsers.add_parser("gui", help="open the read-only diagnostic workbench")
    gui.add_argument("--db", required=True)

    view = subparsers.add_parser("view", help="print the current stable ledger view as JSON")
    view.add_argument("--db", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "validate":
            spec = load_spec(args.spec)
            _emit(
                {
                    "ok": True,
                    "source": spec.source_path,
                    "content_version": spec.content_version,
                    "content_digest": spec.content_digest,
                    "policy_version": spec.policy_version,
                    "policy_digest": spec.policy_digest,
                    "paths": [record.path for record in spec.content.records],
                    "topological_order": spec.content.topological_order(),
                    "nodes": [node.node_id for node in spec.policy.nodes],
                }
            )
            return 0
        if args.command == "view":
            _emit(build_view_model(args.db))
            return 0
        if args.command == "gui":
            run_gui(args.db)
            return 0
        if args.command == "demo":
            workspace = _prepare_workspace(Path(args.workspace))
            if args.inject:
                envelope = _run_injection(args.inject, workspace, args.port_base, args.handshake_timeout)
                _emit(envelope)
                return 2
            report = _run_demo(workspace, args.port_base, args.handshake_timeout)
            report_path = workspace / "report.json"
            report_path.write_text(
                json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            report["report_path"] = str(report_path)
            _emit(report)
            if args.gui:
                run_gui(workspace / "ledger.sqlite3")
            return 0
        raise EdgeError(code="E_STATE", message="unsupported command")
    except EdgeError as exc:
        _emit(exc.envelope())
        return 2
    except KeyboardInterrupt:
        _emit({"ok": False, "code": "E_PROCESS", "message": "interrupted by user"})
        return 130


def _emit(value: dict[str, Any]) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


def _prepare_workspace(path: Path) -> Path:
    path = path.resolve()
    marker = path / ".edgecache-workspace"
    if path.exists() and any(path.iterdir()) and not marker.exists():
        raise EdgeError(
            code="E_STATE",
            message="refusing to clear a non-empty directory that is not an EdgeCache demo workspace",
            field_path="workspace",
            details={"path": str(path)},
        )
    if path.exists():
        for child in path.iterdir():
            if child.is_dir():
                shutil.rmtree(child, onerror=_make_writable_and_retry)
            else:
                try:
                    child.chmod(stat_mode_writable())
                except OSError:
                    pass
                child.unlink()
    path.mkdir(parents=True, exist_ok=True)
    marker.write_text("edgecache demo workspace\n", encoding="utf-8")
    return path


def stat_mode_writable() -> int:
    return 0o600 if os.name != "nt" else 0o666


def _make_writable_and_retry(function: Any, path: str, _exc_info: Any) -> None:
    try:
        os.chmod(path, 0o700 if os.path.isdir(path) else 0o600)
        function(path)
    except OSError:
        pass


def _run_demo(workspace: Path, port_base: int, handshake_timeout: float) -> dict[str, Any]:
    report: dict[str, Any] = {
        "ok": True,
        "scenario": "edgecache-three-scenario-demo",
        "workspace": str(workspace),
        "port_base": port_base,
        "scenarios": [],
        "verification": {},
    }
    cluster = EdgeCluster(
        spec_path=SAMPLE_V1,
        db_path=workspace / "ledger.sqlite3",
        runtime_dir=workspace / "runtime",
        port_base=port_base,
        handshake_timeout=handshake_timeout,
    )
    try:
        started = cluster.start()
        cold = started["bootstrap"]
        report["scenarios"].append(
            {
                "name": "cold_start_fill",
                "ledger_version": cold["ledger_version"],
                "content_version": cold["content_version"],
                "policy_version": cold["policy_version"],
                "entries_by_node": _stable_entries(cold["entries_by_node"]),
                "evictions": _stable_evictions(cold["evictions"]),
            }
        )

        v2 = load_spec(SAMPLE_V2, cycle_error_code="E_SPEC")
        propagation = cluster.evolve_content(v2, trigger_path="/data/root")
        report["scenarios"].append(
            {
                "name": "derived_invalidation",
                "ledger_version": propagation["ledger_version"],
                "content_version": propagation["content_version"],
                "policy_version": propagation["policy_version"],
                "trigger_path": "/data/root",
                "affected_paths": propagation["affected_paths"],
                "propagation": propagation["propagation"],
                "trace_digest": propagation["trace_digest"],
                "entries_by_node": _stable_entries(propagation["entries_by_node"]),
                "evictions": _stable_evictions(propagation["evictions"]),
            }
        )

        snapshot = cluster.save_snapshot()
        v3 = load_spec(SAMPLE_V3, cycle_error_code="E_SPEC")
        newer = cluster.evolve_content(v3, trigger_path="/data/root")
        restored = cluster.restore_snapshot(snapshot["snapshot_path"])
        report["scenarios"].append(
            {
                "name": "snapshot_restore_replay",
                "snapshot_ledger_version": snapshot["ledger_version"],
                "snapshot_content_version": snapshot["content_version"],
                "intervening_ledger_version": newer["ledger_version"],
                "restored_ledger_version": restored["restored_ledger_version"],
                "restored_content_version": restored["content_version"],
                "version_set_compared": restored["comparison"],
                "replay": restored["replay"],
                "node_entry_counts": restored["node_entry_counts"],
            }
        )
        view = cluster.ledger.view_model()
        current_entries = view["entries"]
        current_versions = {entry["content_version"] for entry in current_entries}
        current_policies = {view["policy_version"]}
        report["verification"] = {
            "propagation_order": propagation["affected_paths"],
            "propagation_trace": propagation["propagation"],
            "trace_digest": propagation["trace_digest"],
            "single_content_version_in_current_round": len(current_versions) <= 1
            and (not current_versions or view["content_version"] in current_versions),
            "single_policy_version_in_current_round": current_policies == {view["policy_version"]},
            "snapshot_required_whole_replacement": restored["comparison"]["requires_replace"],
            "replay_ok": restored["replay"]["ok"],
            "current_ledger_version": view["ledger_version"],
            "stale_ledger_versions": [item["ledger_version"] for item in view["versions"] if item["stale"]],
            "unknown_rounds": view["unknown_rounds"],
        }
    finally:
        close = cluster.close()
        report["process_exit_reports"] = close["exit_reports"]
    return report


def _stable_entries(entries_by_node: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    return {
        node: [
            {
                "path": entry["path"],
                "etag": entry["etag"],
                "remaining_ttl_ms": entry["remaining_ttl_ms"],
                "entry_version": entry["entry_version"],
                "stale": entry["stale"],
            }
            for entry in sorted(entries, key=lambda item: (item["path"], item["etag"]))
        ]
        for node, entries in sorted(entries_by_node.items())
    }


def _stable_evictions(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "node_id": event["node_id"],
            "event_seq": event["event_seq"],
            "path": event["path"],
            "etag": event["etag"],
            "reason": event["reason"],
            "rule": event.get("rule"),
            "entry_version": event.get("entry_version"),
        }
        for event in sorted(events, key=lambda item: (item["node_id"], item["event_seq"]))
    ]


def _run_injection(kind: str, workspace: Path, port_base: int, handshake_timeout: float) -> dict[str, Any]:
    if kind == "cycle":
        path = _write_cycle_spec(workspace)
        try:
            load_spec(path, cycle_error_code="E_CYCLE")
        except EdgeError as exc:
            return exc.envelope(injection=kind, injected_file=str(path))
        raise EdgeError(code="E_STATE", message="cycle injection unexpectedly validated")
    if kind == "content_truncation":
        path = _write_truncated_content_spec(workspace)
        try:
            load_spec(path)
        except EdgeError as exc:
            return exc.envelope(injection=kind, injected_file=str(path))
        raise EdgeError(code="E_STATE", message="content truncation injection unexpectedly validated")

    # Snapshot digest mismatch needs a real snapshot first.  The demonstration
    # is deliberately short: cold fill, save, close, tamper, load.
    cluster = EdgeCluster(
        spec_path=SAMPLE_V1,
        db_path=workspace / "ledger.sqlite3",
        runtime_dir=workspace / "runtime",
        port_base=port_base,
        handshake_timeout=handshake_timeout,
    )
    snapshot_path: Path | None = None
    try:
        cluster.start()
        snapshot = cluster.save_snapshot()
        snapshot_path = Path(snapshot["snapshot_path"])
    finally:
        cluster.close()
    assert snapshot_path is not None
    document = json.loads(snapshot_path.read_text(encoding="utf-8"))
    document["content_digest"] = "0" * 64
    tampered = workspace / "snapshot-content-digest-mismatch.json"
    tampered.write_text(json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    ledger = Ledger(workspace / "ledger.sqlite3")
    try:
        try:
            SnapshotStore(workspace / "runtime" / "snapshots").load(tampered, ledger)
        except EdgeError as exc:
            return exc.envelope(injection=kind, injected_file=str(tampered))
        raise EdgeError(code="E_STATE", message="snapshot digest mismatch injection unexpectedly validated")
    finally:
        ledger.close()


def _write_cycle_spec(workspace: Path) -> Path:
    document = json.loads(SAMPLE_V1.read_text(encoding="utf-8"))
    for record in document["content"]:
        if record["path"] == "/data/left":
            record["derived_from"] = ["/page/home"]
            break
    target = workspace / "injected-cycle.json"
    target.write_text(json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return target


def _write_truncated_content_spec(workspace: Path) -> Path:
    document = json.loads(SAMPLE_V1.read_text(encoding="utf-8"))
    for record in document["content"]:
        if record["path"] == "/data/root":
            content = base64.b64decode(record["content_b64"])
            record["content_b64"] = base64.b64encode(content[:-2]).decode("ascii")
            break
    target = workspace / "injected-truncated-content.json"
    target.write_text(json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return target


def find_free_port_base(count: int = 4, start: int = 43000) -> int:
    for base in range(start, 65000 - count):
        sockets: list[socket.socket] = []
        try:
            for port in range(base, base + count):
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sockets.append(sock)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("127.0.0.1", port))
            return base
        except OSError:
            continue
        finally:
            for sock in sockets:
                sock.close()
    raise EdgeError(code="E_PORT", message="unable to find a contiguous loopback high-port range")


if __name__ == "__main__":
    raise SystemExit(main())
