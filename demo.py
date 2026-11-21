#!/usr/bin/env python3
"""End-to-end local demonstration using only relative project modules.

The script materializes the two checked-in JSON specs with three free loopback
ports, runs the same two revisions twice, compares deterministic artifacts
byte-for-byte, performs a live post-snapshot replay, and writes a report.
"""

from __future__ import annotations

import argparse
import json
import shutil
import socket
from pathlib import Path
from typing import Any

from modeldash.gui.view import format_text_view
from modeldash.nodes import NodeSupervisor, SocketDispatcher
from modeldash.replay import ReplayVerifier
from modeldash.scheduler import Scheduler
from modeldash.spec import load_spec, validate_spec
from modeldash.storage import Ledger
from modeldash.util import StructuredLog, canonical_json, digest_json, ensure_directory


ROOT = Path(__file__).resolve().parent


def pick_free_ports(count: int) -> list[int]:
    ports: list[int] = []
    probe_ports: list[socket.socket] = []
    try:
        for port in range(47831, 48000):
            probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                probe.close()
                continue
            probe_ports.append(probe)
            ports.append(port)
            if len(ports) == count:
                break
    finally:
        for probe in probe_ports:
            probe.close()
    if len(ports) != count:
        raise RuntimeError("unable to reserve three loopback ports in 47831..47999")
    return ports


def materialize_specs(base_dir: Path) -> tuple[Path, Path]:
    ports = pick_free_ports(3)
    sources = [ROOT / "examples" / "demo_v1.json", ROOT / "examples" / "demo_v2.json"]
    targets: list[Path] = []
    for source in sources:
        raw = json.loads(source.read_text(encoding="utf-8"))
        for node, port in zip(raw["nodes"], ports):
            node["port"] = port
        validate_spec(raw)
        target = base_dir / source.name
        target.write_text(json.dumps(raw, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        targets.append(target)
    return targets[0], targets[1]


def stable_artifacts(summary: Any) -> dict[str, Any]:
    tasks = []
    for task in summary.tasks:
        tasks.append(
            {
                "task_id": task["task_id"],
                "state": task["state"],
                "error_code": task["error_code"],
                "retries_total": task["retries_total"],
                "output_summary": task["output_summary"],
                "completed_shards": [
                    {
                        "shard_index": shard["shard_index"],
                        "output_summary": shard["output_summary"],
                        "attempts": shard["attempts"],
                    }
                    for shard in sorted(task["completed_shards"], key=lambda row: row["shard_index"])
                ],
            }
        )
    receipts = [
        {
            "receipt_key": item["receipt_key"],
            "node_id": item["node_id"],
            "task_id": item["task_id"],
            "config_version": item["config_version"],
            "output_summary": item["output_summary"],
            "status": item["status"],
            "decay_count": item["decay_count"],
            "untrust_score": item["untrust_score"],
        }
        for item in sorted(summary.receipts, key=lambda row: row["receipt_key"])
    ]
    node_untrust = [
        {
            "node_id": item["node_id"],
            "rank": item["rank"],
            "score": item["score"],
            "untrusted_receipts": item["untrusted_receipts"],
            "receipt_count": item["receipt_count"],
            "max_decay": item["max_decay"],
        }
        for item in summary.node_untrust
    ]
    preemptions = [
        {
            "task_id": item["task_id"],
            "by_task_id": item["by_task_id"],
            "requested_shard": item["requested_shard"],
            "suspended_shard": item["suspended_shard"],
            "status": item["status"],
        }
        for item in summary.preemptions
    ]
    decisions = [
        {
            "high_task_id": item["high_task_id"],
            "victim_task_id": item["victim_task_id"],
            "verdict": item["verdict"],
            "reason": item["reason"],
            "preemptable": item["preemptable"],
        }
        for item in summary.decisions
    ]
    return {
        "deterministic_digest": summary.deterministic_digest,
        "tasks": tasks,
        "receipts": receipts,
        "node_untrust": node_untrust,
        "preemptions": preemptions,
        "decisions": decisions,
        "node_output_digest": digest_json(
            {
                "tasks": [(item["task_id"], item["output_summary"], item["completed_shards"]) for item in tasks],
                "receipts": [(item["receipt_key"], item["output_summary"]) for item in receipts],
            }
        ),
        "untrust_ranking_bytes": canonical_json(node_untrust),
        "untrust_ranking_digest": digest_json(node_untrust),
    }


def run_revision(spec_path: Path, db_path: Path, data_dir: Path) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    spec = load_spec(spec_path)
    logger = StructuredLog(data_dir / "logs" / "workbench.jsonl")
    ledger = Ledger(db_path)
    try:
        activation = ledger.activate_spec(spec)
        snapshots: list[dict[str, Any]] = []

        def snapshot_hook(state: dict[str, Any], label: str) -> None:
            snapshots.append(ledger.create_snapshot(activation.session_id, state, label))

        supervisor = NodeSupervisor(spec, data_dir, logger)
        try:
            supervisor.start_all()
            summary = Scheduler(
                spec,
                ledger,
                activation.session_id,
                SocketDispatcher(spec, logger),
                recovered_state=activation.state,
                logger=logger,
                snapshot_hook=snapshot_hook,
            ).run()
        finally:
            supervisor.stop_all()
        snapshots.append(
            ledger.create_snapshot(
                activation.session_id,
                ledger.latest_state(activation.session_id) or {},
                "session-final",
            )
        )
        metadata = {
            "activation": activation.action,
            "session_id": activation.session_id,
            "stale_summary": activation.stale_summary,
            "snapshots": snapshots,
            "spec_version": spec.spec_version,
            "current_version": spec.current_version,
        }
        return summary, snapshots, metadata
    finally:
        ledger.close()


def replay_final(spec_path: Path, db_path: Path, data_dir: Path) -> dict[str, Any]:
    spec = load_spec(spec_path)
    logger = StructuredLog(data_dir / "logs" / "workbench.jsonl")
    ledger = Ledger(db_path)
    supervisor = None
    try:
        session_id = ledger.current_session_id()
        if session_id is None:
            raise RuntimeError("no active session to replay")
        snapshots = ledger.list_snapshots(session_id)
        start = next((item for item in snapshots if item["label"] == "session-start"), None)
        snapshot_id = start["snapshot_id"] if start else (snapshots[-1]["snapshot_id"] if snapshots else None)
        supervisor = NodeSupervisor(spec, data_dir, logger)
        supervisor.start_all()
        summary = ReplayVerifier(
            spec,
            ledger,
            session_id,
            snapshot_id=snapshot_id,
            dispatcher=SocketDispatcher(spec, logger),
            redispatch=True,
        ).verify()
        return summary.to_dict()
    finally:
        if supervisor is not None:
            supervisor.stop_all()
        ledger.close()


def diagnostics(db_path: Path) -> str:
    ledger = Ledger(db_path)
    try:
        return format_text_view(ledger.diagnostic_view())
    finally:
        ledger.close()


def execute_repeat(
    repeat_index: int,
    base_dir: Path,
    spec_v1: Path,
    spec_v2: Path,
) -> dict[str, Any]:
    run_dir = base_dir / f"run-{repeat_index}"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    ensure_directory(run_dir / "logs")
    db_path = run_dir / "ledger.sqlite3"

    v1_summary, v1_snapshots, v1_metadata = run_revision(spec_v1, db_path, run_dir)
    v1_artifacts = stable_artifacts(v1_summary)
    v2_summary, v2_snapshots, v2_metadata = run_revision(spec_v2, db_path, run_dir)
    v2_artifacts = stable_artifacts(v2_summary)
    replay = replay_final(spec_v2, db_path, run_dir)
    diagnostic_text = diagnostics(db_path)
    (run_dir / "diagnostic.txt").write_text(diagnostic_text, encoding="utf-8")

    return {
        "repeat": repeat_index,
        "v1": v1_artifacts,
        "v2": v2_artifacts,
        "v1_metadata": v1_metadata,
        "v2_metadata": v2_metadata,
        "replay": replay,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(ROOT / "runtime" / "demo"))
    parser.add_argument("--repeat", type=int, default=2)
    args = parser.parse_args(argv)
    if args.repeat < 2:
        parser.error("--repeat must be at least 2 for the byte-identity check")
    base_dir = ensure_directory(Path(args.data_dir))
    spec_v1, spec_v2 = materialize_specs(base_dir)
    results = [execute_repeat(index + 1, base_dir, spec_v1, spec_v2) for index in range(args.repeat)]

    first = results[0]
    comparisons = {}
    for revision in ("v1", "v2"):
        expected = canonical_json(first[revision])
        comparisons[revision] = all(canonical_json(result[revision]) == expected for result in results[1:])
        if not comparisons[revision]:
            raise SystemExit(f"E_DEMO: {revision} deterministic artifacts differ between runs")

    report = {
        "ok": True,
        "specs": [str(spec_v1), str(spec_v2)],
        "repeat": args.repeat,
        "byte_identical": comparisons,
        "reports": results,
        "first_diagnostic": diagnostics(base_dir / "run-1" / "ledger.sqlite3"),
    }
    report_path = base_dir / "demo_report.json"
    report_path.write_text(json.dumps(report, sort_keys=True, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"DEMO_OK repeats={args.repeat} byte_identical={comparisons}")
    print(f"v1 preemptions={len(first['v1']['preemptions'])} ranking={first['v1']['node_untrust']}")
    print(f"v1 decisions={first['v1']['decisions']}")
    print(f"v2 activation={first['v2_metadata']['activation']} ranking={first['v2']['node_untrust']}")
    print(f"replay={first['replay']}")
    print(f"report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
