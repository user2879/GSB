"""Repeatable local acceptance demonstration."""

from __future__ import annotations

from contextlib import closing
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import threading
from typing import Any

from .arbitration import WaterCandidate, arbitrate_budget, arbitrate_water
from .client import NodeClient, NodeProcess
from .errors import GSBError
from .fixed import analyze_fixed_error, water_level
from .invalidation import plan_invalidation
from .ledger import Ledger, compare_record_sets, inspect_snapshot, restore_snapshot
from .model import Spec
from .protocol import JsonLineChannel
from .runtime import ClusterRuntime
from .samples import canonical_json_bytes, generate_samples, sample_digest
from .spec import load_spec, parse_spec


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _record_summary(records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        shard_id: {
            "aggregate_digest": record["aggregate_digest"],
            "lease_id": record["lease"]["lease_id"],
            "utilization_fp": record["water"]["utilization_fp"],
            "content_digest": record["content_digest"],
        }
        for shard_id, record in records.items()
    }


def _expect_error(expected_code: str, action: Any) -> dict[str, Any]:
    try:
        action()
    except GSBError as exc:
        _assert(exc.code == expected_code, f"expected {expected_code}, got {exc.code}: {exc}")
        return exc.as_dict()
    raise AssertionError(f"expected {expected_code}, but no error was raised")


def _fake_node_server(port: int, reply: bytes | None) -> threading.Thread:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", port))
    listener.listen(4)
    stop = threading.Event()

    def serve() -> None:
        try:
            listener.settimeout(5)
            connection, _ = listener.accept()
            with closing(connection):
                channel = JsonLineChannel(connection)
                try:
                    channel.receive(2.0, stage="fake_request")
                except GSBError:
                    return
                if reply is not None:
                    connection.sendall(reply)
                else:
                    stop.wait(2.0)
        finally:
            listener.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    return thread


def _free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


def _error_coverage(spec: Spec, workdir: Path) -> dict[str, Any]:
    coverage: dict[str, Any] = {}

    invalid_spec = spec.as_dict()
    invalid_spec["nodes"][0]["host"] = "localhost"
    coverage["E_SPEC"] = _expect_error("E_SPEC", lambda: parse_spec(invalid_spec))

    bad_dependency = spec.as_dict()
    bad_dependency["shards"][1]["dependencies"] = ["does-not-exist"]
    coverage["E_DEPEND"] = _expect_error("E_DEPEND", lambda: parse_spec(bad_dependency))

    occupied = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    occupied.bind(("127.0.0.1", 0))
    occupied.listen(1)
    occupied_port = int(occupied.getsockname()[1])
    port_node = spec.nodes[0].__class__(
        id="port-node",
        host="127.0.0.1",
        port=occupied_port,
        seed=1,
        concurrency=1,
        lease_limit=1,
        quota=1,
        used=0,
        fixed_scale=1000,
    )
    try:
        coverage["E_PORT"] = _expect_error(
            "E_PORT", lambda: NodeProcess(port_node, startup_timeout_s=2.0).start()
        )
    finally:
        occupied.close()

    timeout_port = _free_port()
    timeout_thread = _fake_node_server(timeout_port, None)
    timeout_node = port_node.__class__(
        id="timeout-node",
        host="127.0.0.1",
        port=timeout_port,
        seed=1,
        concurrency=1,
        lease_limit=1,
        quota=1,
        used=0,
        fixed_scale=1000,
    )
    coverage["E_TIMEOUT"] = _expect_error(
        "E_TIMEOUT", lambda: NodeClient(timeout_node, read_timeout_s=0.2).handshake()
    )
    timeout_thread.join(timeout=3)

    proto_port = _free_port()
    proto_thread = _fake_node_server(proto_port, b'{"ok":true}\n')
    proto_node = port_node.__class__(
        id="proto-node",
        host="127.0.0.1",
        port=proto_port,
        seed=1,
        concurrency=1,
        lease_limit=1,
        quota=1,
        used=0,
        fixed_scale=1000,
    )
    coverage["E_PROTO"] = _expect_error(
        "E_PROTO", lambda: NodeClient(proto_node, read_timeout_s=0.5).handshake()
    )
    proto_thread.join(timeout=3)

    expected = {"shard": {"aggregate_digest": "aaa"}}
    actual = {"shard": {"aggregate_digest": "bbb"}}
    coverage["E_DIVERGE"] = _expect_error(
        "E_DIVERGE", lambda: compare_record_sets(actual, expected)
    )
    return coverage


def run_demo(spec_path: str | Path, evolved_path: str | Path, workdir: str | Path) -> dict[str, Any]:
    work = Path(workdir)
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    spec = load_spec(spec_path)
    evolved = load_spec(evolved_path)
    fixes = analyze_fixed_error(len(spec.nodes), spec.fixed_scale)
    node_a = spec.node_map()["node-a"]
    node_b = spec.node_map()["node-b"]

    level_a = water_level(
        "node-a", 1, 3, spec.fixed_scale, lease_limit_units=node_a.lease_limit
    )
    level_b = water_level(
        "node-b", 2, 6, spec.fixed_scale, lease_limit_units=node_b.lease_limit
    )
    candidate_a = WaterCandidate("raw-a", "node-a", 3, level_a)
    candidate_b = WaterCandidate("raw-b", "node-b", 3, level_b)
    water_decision = arbitrate_water(
        candidate_a,
        candidate_b,
        topo_rank=spec.topo_rank(),
        error_model=fixes,
    )
    _assert(water_decision["error_affected"], "water boundary must be flagged as error affected")
    _assert(water_decision["decision"] == "raw-a", "stable topo/id arbitration failed")

    budget_a = WaterCandidate("raw-a", "node-a", 4, level_a)
    budget_b = WaterCandidate("raw-b", "node-b", 4, level_b)
    budget_decision = arbitrate_budget(
        4,
        budget_a,
        budget_b,
        topo_rank=spec.topo_rank(),
        error_model=fixes,
        budget_error_bound_units=0,
    )
    _assert(budget_decision["error_affected"], "budget boundary must be flagged")
    _assert(budget_decision["decision"] == "raw-a", "budget stable arbitration failed")

    ledger_path = work / "live.db"
    initial_ledger = Ledger(ledger_path, session_version=1)
    with ClusterRuntime(spec, ledger=initial_ledger, session_version=1) as runtime:
        initial = runtime.execute()
    initial_records = initial.records
    _assert(len(initial_records) == len(spec.shards), "not every shard committed")
    _assert(initial.lease_stats["global_units"] == 0, "resource units leaked after the run")

    repeat_rows_a = generate_samples(node_a, spec.shard_map()["raw-a"])
    repeat_rows_b = generate_samples(node_a, spec.shard_map()["raw-a"])
    _assert(
        canonical_json_bytes(repeat_rows_a) == canonical_json_bytes(repeat_rows_b),
        "deterministic sample bytes differ",
    )
    split_signature = sample_digest(repeat_rows_a)
    _assert(split_signature == sample_digest(repeat_rows_b), "sample digest is not deterministic")

    snapshot_path = work / "session-1.snapshot.db"
    snapshot_info = initial_ledger.create_snapshot(snapshot_path)
    restore_target = work / "restore-target.db"
    stale = Ledger(restore_target, session_version=9)
    stale.close()
    restore_report = restore_snapshot(snapshot_path, restore_target, expected_version=1)
    _assert(restore_report["whole_replacement"], "snapshot was not restored as a whole")
    _assert(restore_report["version_mismatch"], "stale session version was not replaced")
    _assert(inspect_snapshot(restore_target)["record_count"] == len(initial_records), "restore lost records")

    crash_ledger_path = work / "crash.db"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1]) + os.pathsep + env.get("PYTHONPATH", "")
    crashed = subprocess.run(
        [
            sys.executable,
            "-m",
            "gsb.crash_helper",
            "--ledger",
            str(crash_ledger_path),
            "--session-version",
            "1",
            "--shard-id",
            "crash-shard",
            "--units",
            "6",
        ],
        env=env,
        check=False,
    )
    _assert(crashed.returncode == 70, "crash helper did not force-terminate as intended")
    recovered_ledger = Ledger(crash_ledger_path, session_version=1)
    recovery = recovered_ledger.recover()
    _assert(recovery["reclaimed_units"] == 6, "crash recovery did not reclaim quota")
    _assert(recovery["unknown_shards"][0]["requires_rerun"], "unknown shard not marked for rerun")
    _assert(recovery["exit_report_available"] is False, "forced termination exit report was claimed")
    recovered_ledger.close()

    plan = plan_invalidation(spec, evolved)
    _assert(plan.invalid_shards == ("raw-a", "join-ab", "report"), "unexpected invalidation set")
    _assert(plan.preserved_shards == ("raw-b", "raw-d"), "unexpected preserved set")

    incremental_ledger = Ledger(work / "incremental.db", session_version=2)
    with ClusterRuntime(evolved, ledger=incremental_ledger, session_version=2) as runtime:
        incremental = runtime.recompute_subset(initial_records, plan)
    incremental_ledger.close()

    full_ledger = Ledger(work / "full-rebuild.db", session_version=2)
    with ClusterRuntime(evolved, ledger=full_ledger, session_version=2) as runtime:
        full = runtime.execute()
    full_ledger.close()
    comparison = compare_record_sets(incremental.records, full.records)
    _assert(comparison["consistent"], "incremental rerun diverged from full rebuild")
    _assert(incremental.preserved["preserved"], "byte preservation failed")

    error_coverage = _error_coverage(spec, work)
    report = {
        "status": "PASS",
        "spec": str(Path(spec_path)),
        "evolved_spec": str(Path(evolved_path)),
        "fixed_scale": spec.fixed_scale,
        "error_model": fixes.as_dict(),
        "water_decision": water_decision,
        "budget_decision": budget_decision,
        "initial_run": {
            "topological_order": list(spec.topological_order()),
            "records": _record_summary(initial_records),
            "waits": initial.waits,
            "lease_events": initial.lease_events,
            "lease_stats": initial.lease_stats,
        },
        "snapshot": {
            "created": snapshot_info,
            "restore": restore_report,
        },
        "crash_recovery": recovery,
        "invalidation": plan.as_dict(),
        "incremental_rerun": {
            "recomputed": list(incremental.recomputed_shards),
            "preserved": incremental.preserved,
            "records": _record_summary(incremental.records),
        },
        "full_rebuild_consistency": comparison,
        "error_coverage": error_coverage,
        "limitations": [
            "No exit report is produced for a forced process termination.",
            "Tkinter is optional; the acceptance demo is headless and does not require a display.",
        ],
    }
    (work / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    initial_ledger.close()
    return report
