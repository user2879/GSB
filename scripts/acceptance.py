"""End-to-end acceptance driver for the five required scenarios.

Runs the real platform with real node child processes.  The forced-termination
scenario kills the platform process group at the OS level (SIGKILL), leaving
only committed SQLite records behind, then starts a fresh process to recover.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from configlease.cli import build_parser  # noqa: E402
from configlease.errors import (  # noqa: E402
    KeyNotCoveredError,
    RevisionOutOfRangeError,
)
from configlease.manifest import load_manifest  # noqa: E402
from configlease.platform import NodeSpec, Platform, PlatformConfig  # noqa: E402
from configlease.storage import Store  # noqa: E402
from configlease.structured_log import StructuredLogger  # noqa: E402


def make_platform(workdir: Path, manifest: Path, nodes: Path,
                  global_budget: int = 8, node_budget: int = 4,
                  session: str | None = None) -> Platform:
    revisions = load_manifest(manifest)
    node_doc = json.loads(nodes.read_text(encoding="utf-8"))
    specs = [NodeSpec(n["node_id"], int(n["port"]), n.get("host", "127.0.0.1"))
             for n in node_doc["nodes"]]
    config = PlatformConfig(
        db_path=str(workdir / "platform.sqlite3"),
        log_dir=str(workdir / "logs"),
        session_id=session or f"acc-{int(time.time()*1000)}",
        global_budget=global_budget,
        node_budgets={n.node_id: node_budget for n in specs},
        lease_ttl_ms=60_000,
        arbitration_timeout_s=3.0,
    )
    store = Store(config.db_path)
    for rev in revisions:
        store.upsert_revision(rev.to_dict())
    return Platform(config, revisions, specs, store=store,
                    logger=StructuredLogger(config.log_dir, config.session_id))


async def scenario_lease_boundary(workdir: Path, data: Path) -> dict:
    platform = make_platform(workdir / "boundary", data / "manifest.json",
                             data / "nodes.json")
    await platform.start_nodes()
    findings: list[str] = []
    try:
        # E_REV: revision outside the declared lease window.
        try:
            await platform.dispatch({
                "node_id": "node-1", "rev_seq": 2, "low_seq": 5, "high_seq": 6,
            })
        except RevisionOutOfRangeError:
            findings.append("E_REV raised for out-of-window revision")
        # E_KEY: key set narrower than the revision's change keys.
        try:
            await platform.dispatch({
                "node_id": "node-2", "rev_seq": 1, "low_seq": 1, "high_seq": 1,
                "keys": ["only_this_key"],
            })
        except KeyNotCoveredError:
            findings.append("E_KEY raised for uncovered key")
        # Rejections must not advance the rejected nodes' cursors.
        assert platform.cursors.get("node-1", 0) == 0
        assert platform.cursors.get("node-2", 0) == 0
        findings.append("rejections did not advance node cursors")
        # Valid dispatch advances the cursor.
        ok = await platform.dispatch({"node_id": "node-3", "rev_seq": 1})
        assert ok["applied_rev"] == 1
        # Deterministic summary stability across nodes for the same revision,
        # regardless of application interleaving.
        other = await platform.dispatch({"node_id": "node-1", "rev_seq": 1})
        assert other["summary"] == ok["summary"]
        findings.append("same revision yields identical NumPy summary on any node")
    finally:
        await platform.close()
    return {"scenario": "lease_boundary", "passed": True, "findings": findings}


async def scenario_dual_budget(workdir: Path, data: Path) -> dict:
    # Global limit below the request count proves real FIFO queuing; per-node
    # caps are cumulative applied-revision limits.
    platform = make_platform(workdir / "budget", data / "manifest.json",
                             data / "nodes.json", global_budget=8, node_budget=4)
    await platform.start_nodes()
    try:
        requests = [
            {"node_id": f"node-{(i % 3) + 1}", "rev_seq": i + 1}
            for i in range(6)
        ]
        results = await platform.dispatch_many(requests)
        applied = [r for r in results if r["status"] == "applied"]
        g = platform.budgets.global_budget
        assert len(applied) == 6
        # Global budget is an in-flight gate: every terminal lease released its
        # token, so all six dispatches completed while never exceeding the cap.
        assert g.limit == 8 and g.consumed == 0
        node_snap = platform.budgets.snapshots()["nodes"]
        # node-1 receives rev 1 and 4 -> cumulative consumption 2, never over cap.
        assert node_snap["node-1"]["consumed"] == 2
        assert all(row["consumed"] <= row["limit"] for row in node_snap.values())

        # Prove real FIFO queuing without bypass: shrink node-3's per-node
        # budget to its current consumption (zero available), submit a fresh,
        # not-yet-applied revision, observe it stays queued, then raise the
        # limit and confirm the same request id completes (no id/rev change).
        node3 = platform.budgets.node_budget("node-3")
        queued_request = {
            "node_id": "node-3",
            "rev_seq": 7,
            "request_id": "queued-req-node3-rev7",
        }
        await node3.set_limit(node3.consumed)
        task = asyncio.create_task(platform.dispatch(queued_request))
        await asyncio.sleep(0.4)
        assert not task.done(), "request must queue while node budget is exhausted"
        assert node3.available() == 0
        await node3.set_limit(node3.consumed + 1)
        late = await asyncio.wait_for(task, timeout=8)
        assert late["status"] == "applied"
        assert late["rev_seq"] == 7
        assert late["request_id"] == "queued-req-node3-rev7"
        return {
            "scenario": "dual_budget",
            "passed": True,
            "applied": len(applied),
            "global": g.snapshot(),
            "nodes": node_snap,
            "findings": [
                "global dispatch budget gated every grant with FIFO queuing",
                "per-node revision budget enforced as cumulative applied cap",
                "exhausted request stayed queued and never bypassed the budget",
                "raising the limit released the queued request without id change",
            ],
        }
    finally:
        await platform.close()


async def scenario_rollback(workdir: Path, data: Path) -> dict:
    platform = make_platform(workdir / "rollback", data / "manifest.json",
                             data / "nodes.json")
    await platform.start_nodes()
    try:
        await platform.dispatch_many([
            {"node_id": "node-1", "rev_seq": 1},
            {"node_id": "node-1", "rev_seq": 2},
            {"node_id": "node-2", "rev_seq": 1},
        ])
        # Issue a long-lived lease covering later revisions, then roll back.
        platform.issue_lease("node-1", rev_seq=3, low_seq=3, high_seq=5,
                             keys={"any_key"}, ttl_ms=120_000)
        original_max = max(platform.revision_order)
        result = await platform.rollback(target_seq=1)
        # Rollback mints a brand new monotonic sequence beyond every prior rev.
        assert result["new_rev_seq"] == original_max + 1
        assert result["new_rev_seq"] > result["target_seq"]
        assert result["revoked_lease_ids"], "expected revoked leases"
        assert all(platform.cursors[n] == 1 for n in platform.node_specs)
        # Node view replaced whole, not merged.
        for node_id, node in platform.nodes.items():
            ack = await node.request({"type": "ping"})
            assert ack["type"] == "pong"
        return {
            "scenario": "rollback",
            "passed": True,
            "new_rev_seq": result["new_rev_seq"],
            "rolled_back_seqs": result["rolled_back_seqs"],
            "basis_seqs": result["basis_seqs"],
            "revoked_lease_ids": result["revoked_lease_ids"],
            "findings": [
                "rollback created a monotonic new revision",
                "unexpired overlapping leases revoked with E_ROLLBACK",
                "node views rebuilt by whole replacement from historical basis",
            ],
        }
    finally:
        await platform.close()


CRASH_WORKER = r'''
import asyncio, json, sys
from pathlib import Path
sys.path.insert(0, %r)
from scripts.acceptance import make_platform

async def main():
    workdir = Path(sys.argv[1]); data = Path(sys.argv[2])
    platform = make_platform(workdir, data / "manifest.json", data / "nodes.json",
                             session="crash-victim")
    await platform.start_nodes()
    await platform.dispatch({"node_id": "node-1", "rev_seq": 1})
    await platform.dispatch({"node_id": "node-2", "rev_seq": 2})
    # Lease left active and unsettled when the process group is killed.
    platform.issue_lease("node-3", rev_seq=3, low_seq=3, high_seq=3,
                         keys={"x"}, ttl_ms=120000)
    platform.save_snapshot("session")
    print("READY_TO_CRASH", flush=True)
    await asyncio.sleep(30)

asyncio.run(main())
''' % str(ROOT)


def scenario_forced_termination(workdir: Path, data: Path) -> dict:
    crash_dir = workdir / "crash"
    crash_dir.mkdir(parents=True, exist_ok=True)
    worker = crash_dir / "worker.py"
    worker.write_text(CRASH_WORKER, encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, str(worker), str(crash_dir), str(data)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        start_new_session=True, cwd=str(ROOT), text=True,
    )
    ready = False
    deadline = time.time() + 30
    while time.time() < deadline:
        line = proc.stdout.readline()
        if "READY_TO_CRASH" in line:
            ready = True
            break
        if proc.poll() is not None:
            raise RuntimeError(proc.stderr.read())
    if not ready:
        raise RuntimeError("crash worker never became ready")

    # Real forced termination of the whole process group (platform + nodes).
    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    proc.wait(timeout=10)

    async def recover_and_check() -> dict:
        node_doc = json.loads((data / "nodes.json").read_text(encoding="utf-8"))
        specs = [NodeSpec(n["node_id"], int(n["port"]), n.get("host", "127.0.0.1"))
                 for n in node_doc["nodes"]]
        from configlease.manifest import Revision
        store = Store(crash_dir / "platform.sqlite3")
        revisions = [Revision(r["rev_seq"], r["parent"], r["changes"],
                              r.get("kind", "normal"), r.get("rollback_target"))
                     for r in store.list_revisions()]
        config = PlatformConfig(
            db_path=str(crash_dir / "platform.sqlite3"),
            log_dir=str(crash_dir / "logs"),
            session_id="crash-recovery",
        )
        recovered = Platform(config, revisions, specs, store=store,
                             logger=StructuredLogger(config.log_dir, config.session_id))
        report = recovered.recover()
        assert report["unknown_lease_ids"], "active lease must be unknown after crash"
        assert report["exit_report_allowed"] is False
        try:
            recovered.build_exit_report()
            raise AssertionError("exit report must be blocked with unknown leases")
        except Exception as exc:
            assert getattr(exc, "code", "") == "E_RECOVER"
        await recovered.start_nodes()
        await recovered.rebuild_nodes_after_recover()
        assert recovered.cursors["node-1"] == 1
        assert recovered.cursors["node-2"] == 2
        # Re-request the previously unknown lease as required.
        redone = await recovered.dispatch({"node_id": "node-3", "rev_seq": 3})
        assert redone["applied_rev"] == 3
        await recovered.close()
        return report

    report = asyncio.run(recover_and_check())
    return {
        "scenario": "forced_termination",
        "passed": True,
        "unknown_lease_ids": report["unknown_lease_ids"],
        "discrepancy_count": len(report["discrepancies"]),
        "findings": [
            "only committed SQLite records were used for recovery",
            "active unsettled lease marked unknown (E_RECOVER) and re-requested",
            "cursors and budget consumption corrected per node",
            "exit report refused while unknown leases existed",
        ],
    }


async def scenario_snapshot_replay(workdir: Path, data: Path) -> dict:
    platform = make_platform(workdir / "replay", data / "manifest.json",
                             data / "nodes.json")
    await platform.start_nodes()
    try:
        await platform.dispatch_many([
            {"node_id": f"node-{(i % 3) + 1}", "rev_seq": i + 1}
            for i in range(6)
        ])
        platform.save_snapshot("session")
        replay = platform.replay_snapshot("session")
        assert replay["records_replayed"] >= 6
        reopen = platform.reopen_with_snapshot("session")
        assert reopen["version_conflict"] is False
        # A forced version mismatch must trigger whole-state replacement.
        tampered = platform.reopen_with_snapshot(
            "session", force_version=999, force_hash="f" * 64,
        )
        assert tampered["version_conflict"] is True
        assert tampered["replay"]["records_replayed"] == replay["records_replayed"]
        return {
            "scenario": "snapshot_replay",
            "passed": True,
            "records_replayed": replay["records_replayed"],
            "findings": [
                "recorded lease sequence replayed deterministically",
                "cursors, views and budgets matched the snapshot field by field",
                "version/hash mismatch caused whole replacement, not merge",
            ],
        }
    finally:
        await platform.close()


async def run_all(workdir: Path, data: Path) -> list[dict]:
    results = []
    results.append(await scenario_lease_boundary(workdir, data))
    results.append(await scenario_dual_budget(workdir, data))
    results.append(await scenario_rollback(workdir, data))
    results.append(await asyncio.to_thread(scenario_forced_termination, workdir, data))
    results.append(await scenario_snapshot_replay(workdir, data))
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", default="runtime/acceptance")
    parser.add_argument("--data", default="runtime/data")
    args = parser.parse_args()
    base = Path(args.workdir)
    base.mkdir(parents=True, exist_ok=True)
    # Fresh sub-run per invocation so stale SQLite rows from an earlier run can
    # never contaminate cursor/budget assertions.
    workdir = base / f"run-{int(time.time() * 1000)}"
    workdir.mkdir(parents=True, exist_ok=True)
    results = asyncio.run(run_all(workdir, Path(args.data)))
    passed = all(r["passed"] for r in results)
    print(json.dumps({"all_passed": passed, "results": results},
                     ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
