"""Repeatable end-to-end demonstration covering all acceptance scenarios."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from .engine import Engine, NodeKey
from .errors import DiagError, canonical_json
from .ledger import Ledger
from .model import load_spec
from .orchestrator import Orchestrator
from .snapshot import build_snapshot, load_snapshot, save_snapshot


class CrashSimulated(Exception):
    pass


def run_demo(samples: Path, data: Path, local: bool = False,
             crash: bool = False) -> dict:
    if data.exists():
        shutil.rmtree(data)
    data.mkdir(parents=True)
    report: dict = {"mode": "local" if local else "subprocess",
                    "crash_requested": crash, "steps": []}
    spec = load_spec(samples / "clusters.json", samples / "tasks.json",
                     samples / "revisions.json")
    ledger = Ledger(data / "ledger.db")
    orch = None if local else Orchestrator(spec)
    if orch is not None:
        orch.start_all()
    engine = Engine(spec, ledger, orch)
    try:
        # 1. Initial incremental apply of the full chain.
        apply_result = engine.apply_chain(
            None, batch_id="demo-apply-1",
            progress=_quiet("initial-apply"))
        report["steps"].append({"step": "initial_apply",
                                "result": _summary(apply_result)})
        head = spec.revisions[-1].version
        chain_versions = set(spec.ancestors_including(head))
        snapshot = build_snapshot(
            spec, head,
            [n for n in ledger.list_nodes("committed")
             if n["version"] in chain_versions],
            {"stage": "apply", "head": head})
        save_snapshot(data / "snapshot.json", snapshot)

        # 2. Authority rebuild from the root and node-by-node comparison.
        rebuild_result = engine.rebuild(1, use_services=not local,
                                        progress=_quiet("rebuild"))
        report["steps"].append({"step": "authority_rebuild",
                                "result": rebuild_result})

        # 3. Amended chain -> invalidation subset redistribution.
        amended_spec = load_spec(samples / "clusters.json",
                                 samples / "tasks.json",
                                 samples / "revisions_amended.json")
        engine.spec = amended_spec
        previous = [{"version": r.version, "values": r.values,
                     "strategy": r.strategy, "derives": list(r.derives)}
                    for r in spec.revisions]
        amended_result = engine.apply_chain(
            previous, batch_id="demo-apply-amended",
            progress=_quiet("amended-apply"))
        report["steps"].append({"step": "invalidation_recompute",
                                "result": _summary(amended_result)})
        engine.spec = spec

        # 4. Optional forced-termination simulation. A staging batch is left
        # open and the in-flight node is recorded in the durable crash journal
        # with status 'unknown'; no exit report is produced.
        if crash:
            ledger.begin_batch("demo-staging-crash")
            key = NodeKey(spec.tasks[0].task_id, spec.tasks[0].targets[0], head)
            in_flight = engine.compute_node_locally(key)
            ledger.insert_unknown(in_flight)
            raise CrashSimulated()
    except CrashSimulated:
        report["steps"].append({"step": "forced_termination",
                                "status": "simulated"})
    finally:
        if orch is not None:
            orch.stop_all()
        ledger.close()

    # 5. Recovery in a fresh process-like context.
    ledger2 = Ledger(data / "ledger.db")
    engine2 = Engine(spec, ledger2, None)
    recovery = engine2.recover()
    report["steps"].append({"step": "recover", "result": {
        "requires_rerun": recovery["requires_rerun"],
        "unknown_nodes": recovery["unknown_nodes"],
        "exit_report_available": recovery["exit_report_available"]}})
    if crash:
        rerun = engine2.rerun_unknown()
        report["steps"].append({"step": "rerun_unknown", "result": rerun})

    # 6. Rollback to a historical root revision.
    rollback = engine2.rollback(3, progress=_quiet("rollback"))
    report["steps"].append({"step": "rollback", "result": {
        "new_root": rollback["new_root"],
        "isolated_count": len(rollback["isolated_nodes"]),
        "isolated_sample": rollback["isolated_nodes"][:3],
        "digest_all_comparable": all(
            item["after_digest"] is not None
            for item in rollback["digest_comparison"])}})

    # 7. Replay verification against a snapshot of the current root chain.
    rollback_chain = set(spec.ancestors_including(rollback["new_root"]))
    rollback_snapshot = build_snapshot(
        spec, rollback["new_root"],
        [n for n in ledger2.list_nodes("committed")
         if n["version"] in rollback_chain],
        {"stage": "post-rollback", "root": rollback["new_root"]})
    save_snapshot(data / "snapshot_rollback.json", rollback_snapshot)
    snapshot = load_snapshot(data / "snapshot_rollback.json", spec)
    replay = engine2.replay(snapshot, use_services=False,
                            progress=_quiet("replay"))
    report["steps"].append({"step": "replay", "result": replay})
    report["acceptance"] = {
        "incremental_matches_rebuild": rebuild_result["consistent"],
        "invalidation_subset_only": amended_result["invalidation"][
            "invalid_set"] and amended_result["unchanged_verified"][
            "all_byte_identical"],
        "rollback_isolates_stale": bool(rollback["isolated_nodes"]),
        "replay_reproducible": replay["reproducible"],
    }
    ledger2.close()
    return report


def _summary(result: dict) -> dict:
    invalidation = result.get("invalidation", {})
    return {
        "head": result.get("head"),
        "invalid_count": len(invalidation.get("invalid_set", [])),
        "recomputed": result.get("recomputed", []),
        "unchanged_byte_identical": result.get(
            "unchanged_verified", {}).get("all_byte_identical"),
        "persist": result.get("persist"),
    }


def _quiet(label):
    def callback(index, total, key: NodeKey):
        return None
    return callback
