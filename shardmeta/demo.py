"""End-to-end reproducible demonstration of the shardmeta component."""
from __future__ import annotations

import argparse
import json
import os
from typing import Any

from .adoption import decisions_canonical_blob
from .clock import ManualClock
from .engine import AccountingEngine
from .evolution import (
    compare_field_by_field,
    filter_unaffected,
    propagate_invalidation,
)
from .generator import generate_sample_bundle
from .ledger import Ledger
from .metadata import decode_file, normalize_files
from .node import NodeOrchestrator
from .spec import load_bundle
from .ui import launch_or_smoke


def _resolve_runtime_leases(request, lease_by_key_idx):
    if request.op in ("renew", "abort") and request.lease_id == "resolved-at-runtime":
        target = (request.object_key, request.shard_index)
        lease = lease_by_key_idx.get(target)
        if lease is None:
            raise RuntimeError(f"no active lease for {target}")
        lease_id = lease.lease_id if hasattr(lease, "lease_id") else lease["lease_id"]
        from dataclasses import replace
        return replace(request, lease_id=lease_id)
    return request


def run_demo(sample_dir: str, ledger_path: str, *, use_subprocess: bool = True,
             smoke_ui: bool = True, simulate_crash: bool = True) -> dict[str, Any]:
    paths = generate_sample_bundle(sample_dir)
    nodes, manifests, raw_requests = load_bundle(
        paths["nodes"], paths["objects"], paths["requests"],
    )

    report: dict[str, Any] = {"samples": paths, "steps": []}

    # --- Cross-format normalization ------------------------------------- #
    decoded = [decode_file(paths[name], fmt)
               for name, fmt in (("c1", 1), ("c2", 2), ("c3", 3))]
    normalized = normalize_files(decoded)
    report["normalization"] = {
        "record_count": len(normalized),
        "by_format": {
            f"fmt{record.source_format}": {
                "merge_id": record.merge_id,
                "filled_fields": sorted(record.filled_fields),
                "fill_basis": record.fill_basis,
            }
            for record in normalized
        },
    }

    # --- Ledger + optional real node subprocesses ----------------------- #
    if os.path.exists(ledger_path):
        os.remove(ledger_path)
    ledger = Ledger(ledger_path)
    orchestrator = NodeOrchestrator(nodes) if use_subprocess else None
    if orchestrator is not None:
        orchestrator.start_all()
        report["steps"].append("started node subprocesses on 127.0.0.1")

    # Real subprocesses own their own monotonic clock, so the engine uses the
    # real clock too and grants short genuine TTLs. In-process mode uses a
    # manual deterministic clock and never sleeps to emulate expiry.
    clock = ManualClock(start_ms=1_000_000) if not use_subprocess else None
    engine = AccountingEngine(nodes, manifests, ledger, orchestrator=orchestrator,
                              clock=clock, lease_ttl_ms=200 if use_subprocess else 300)
    engine.ingest_metadata_candidates(normalized, "sample-metadata")

    lease_by_key_idx: dict[tuple[str, int], Any] = {}
    results: list[dict] = []
    crash_seq: int | None = None

    for request in raw_requests:
        request = _resolve_runtime_leases(request, lease_by_key_idx)
        if simulate_crash and request.seq == 7 and orchestrator is not None:
            target_node = manifests[request.object_key].node_of(request.shard_index)
            orchestrator.kill_node(target_node)
            engine.mark_unknown_and_rerun(request, "node force-killed before commit")
            crash_seq = request.seq
            results.append({"seq": request.seq, "op": request.op,
                            "status": "unknown-rerun", "node": target_node})
            # Restart from committed ledger state, then rerun the request.
            recovery_leases = [
                lease.to_dict()
                for lease in engine._leases.values()
                if lease.node_id == target_node
            ]
            orchestrator.restart_node(target_node, recovery_leases=recovery_leases)
            restored = engine.restore_state_from_ledger()
            report.setdefault("crash_recovery", {})["restored"] = restored
            result = engine.process_request(request)
            results.append({"seq": request.seq, "op": request.op,
                            "status": "rerun-ok", "detail_keys":
                            sorted(result.detail.keys())})
        else:
            result = engine.process_request(request)
            if result.op == "grant":
                lease = result.detail["lease"]
                lease_by_key_idx[(lease["object_key"], lease["shard_index"])] = lease
            elif result.op in ("renew", "abort"):
                lease = result.detail["lease"]
                lease_by_key_idx[(lease["object_key"], lease["shard_index"])] = lease
            results.append({"seq": request.seq, "op": request.op,
                            "status": "ok", "elapsed_ms": result.elapsed_ms})

        # In manual mode advance logical time; real mode relies on monotonic time.
        if clock is not None:
            clock.advance(130)

    report["request_results"] = results
    report["crash_recovery"] = report.get("crash_recovery", {})
    report["crash_recovery"]["crash_seq"] = crash_seq
    report["crash_recovery"]["unknown_requests"] = [
        dict(row) for row in ledger.fetch_unknown()
    ]

    # --- Real expiry demonstration (monotonic comparison, no fake delay) - #
    if clock is not None:
        clock.advance(10_000)
    else:
        # Real subprocess clock: wait beyond the longest genuine TTL so that
        # expiry is detected by monotonic comparison, not artificial logic.
        import time
        time.sleep(0.6)
    final_expiry = engine.reap_expired_leases()

    # Count expiry events already committed during request processing too.
    committed_expiry: dict[str, int] = {}
    for row in ledger.fetch_events():
        if row["op"] == "expire":
            committed_expiry[row["node_id"]] = committed_expiry.get(row["node_id"], 0) + 1
    report["expiry"] = {
        "committed_during_processing": committed_expiry,
        "final_explicit_reap": {
            node_id: len(ids) for node_id, ids in final_expiry.items()
        },
        "total_expired_lease_events": sum(committed_expiry.values())
        + sum(len(ids) for ids in final_expiry.values()),
    }

    # --- Cached fast reads (after real expiry reaping) ------------------ #
    engine.reap_expired_leases()
    fast = {key: engine.fast_read(key) for key in sorted(manifests)}
    report["fast_reads"] = {
        key: {"elapsed_ms": value["elapsed_ms"],
              "adopted_count": len(value["view"]["adopted"]),
              "lease_watermark": value["view"]["lease_watermark"]}
        for key, value in fast.items()
    }

    # --- Authoritative replay + divergence check ------------------------ #
    authoritative = engine.authoritative_replay()
    engine.verify_against_ledger()
    report["adoption_blob_sha256"] = __import__("hashlib").sha256(
        decisions_canonical_blob(engine.adoption)).hexdigest()
    report["authoritative_blob_sha256"] = __import__("hashlib").sha256(
        decisions_canonical_blob(authoritative)).hexdigest()
    report["divergence"] = "none"

    # --- Evolution / invalidation + incremental vs full rebuild --------- #
    new_nodes = {
        node_id: _bump_node(spec, 1 if node_id == "node-1" else 0)
        for node_id, spec in nodes.items()
    }
    invalidation = propagate_invalidation(
        old_nodes=nodes, new_nodes=new_nodes, manifests=manifests,
        candidates=engine._candidates,
    )
    unaffected = filter_unaffected(engine._candidates, invalidation)
    # Compare the adopted winners for unaffected shards: incremental view vs a
    # full from-blank authoritative rebuild, field by field.
    incremental_winners = [
        decision.winner for decisions in engine.adoption.values()
        for decision in decisions
        if (decision.object_key, decision.shard_index)
        not in set(invalidation.affected_shards)
    ]
    rebuilt_decisions = engine.authoritative_replay()
    rebuilt_winners = [
        decision.winner for decisions in rebuilt_decisions.values()
        for decision in decisions
        if (decision.object_key, decision.shard_index)
        not in set(invalidation.affected_shards)
    ]
    mismatches = compare_field_by_field(incremental_winners, rebuilt_winners)
    report["evolution"] = {
        "affected_keys": invalidation.affected_keys,
        "affected_shards": [list(item) for item in invalidation.affected_shards],
        "propagation_steps": [step.as_dict() for step in invalidation.steps],
        "incremental_vs_full_mismatches": mismatches,
        "unaffected_candidate_count": len(unaffected),
    }

    # --- Ledger snapshot identity --------------------------------------- #
    digest = ledger.refresh_digest()
    report["ledger"] = {
        "path": ledger_path,
        "format_version": int(ledger.get_meta("format_version")),
        "content_digest": digest,
        "cursor": ledger.progress_cursor(),
        "event_count": len(ledger.fetch_events()),
    }

    # --- Headless Tk smoke test ----------------------------------------- #
    if smoke_ui:
        report["tk_smoke"] = launch_or_smoke(
            engine, sorted(manifests), smoke=True,
        )

    if orchestrator is not None:
        orchestrator.stop_all()
    ledger.close()
    return report


def _bump_node(spec, delta: int):
    from dataclasses import replace
    return replace(spec, meta_version=spec.meta_version + delta)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="shardmeta reproducible demo")
    parser.add_argument("--samples", default="samples")
    parser.add_argument("--ledger", default="samples/shardmeta-ledger.sqlite3")
    parser.add_argument("--report", default="samples/demo-report.json")
    parser.add_argument("--no-subprocess", action="store_true",
                        help="run nodes in-process instead of real subprocesses")
    parser.add_argument("--no-crash", action="store_true",
                        help="skip forced node termination")
    parser.add_argument("--no-ui", action="store_true")
    args = parser.parse_args(argv)

    root = os.path.dirname(os.path.abspath(args.report))
    os.makedirs(root, exist_ok=True)
    report = run_demo(
        args.samples, args.ledger,
        use_subprocess=not args.no_subprocess,
        smoke_ui=not args.no_ui,
        simulate_crash=not args.no_crash,
    )
    with open(args.report, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps({
        "report": args.report,
        "ledger": report["ledger"],
        "divergence": report["divergence"],
        "crash_seq": report["crash_recovery"].get("crash_seq"),
        "normalized_records": report["normalization"]["record_count"],
    }, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
