"""Hard-kill recovery and replay-verification demo.

Phase A runs the full pipeline and replays every candidate, committing
receipts atomically.  Phase B simulates a forced termination *during* an
in-flight replay by inserting an ``unknown`` replay row (the exact state a
kill leaves behind: intent committed, receipt never committed).  Phase C
reopens the same SQLite snapshot, reruns only the unknown candidate through
the real loopback service, and cross-checks every committed receipt; any
first field divergence raises ``E_REPLAY``.
"""

from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from slowquery_diag.engine import DiagnosticEngine  # noqa: E402
from slowquery_diag.scheduler import ReplayJob  # noqa: E402

WORKDIR = os.path.join(ROOT, "workdir")


def main() -> int:
    os.makedirs(WORKDIR, exist_ok=True)
    declaration = os.path.join(ROOT, "examples", "declaration.json")
    ledger_path = os.path.join(WORKDIR, "crash_ledger.db")
    catalog_path = os.path.join(WORKDIR, "crash_catalog.json")
    for suffix in ("", "-wal", "-shm", ".readonly.bak"):
        path = ledger_path + suffix
        if os.path.exists(path):
            os.remove(path)

    engine = DiagnosticEngine(ledger_path, scale_exponent=1)
    result = engine.run(declaration)
    engine.build_replay_catalog(result.bundle, catalog_path)
    quota = {source.source_id: 1 for source in result.bundle.sources}
    config = {
        "catalog_path": catalog_path,
        "global_concurrency": 2,
        "per_source_quota": quota,
        "socket_timeout_ms": 2000,
    }
    result = engine.run(declaration, replay=True, replay_config=config)
    scheme = engine.scheme_version
    committed_before = engine.ledger.load_replay_jobs(scheme)
    print(f"phase A: committed {sum(1 for r in committed_before if r['status'] == 'committed')} replay receipts")

    # Simulate SIGKILL while one replay is in flight.  The intent row is
    # written in its own transaction before the socket call; the receipt
    # transaction never happens, so on disk the row stays status=unknown.
    victim = result.bundle.candidates[0]
    victim_source = victim.target_sources[-1]
    crash_candidate_id = victim.candidate_id + "_inflight"
    engine.ledger.write_replay_intent(ReplayJob(
        candidate_id=crash_candidate_id,
        source_id=victim_source,
        rewritten_sql=victim.rewritten_sql,
    ))
    print(f"phase B: simulated hard kill during replay "
          f"{crash_candidate_id}@{victim_source} -> status=unknown")
    engine.close()

    # Reopen: only fully committed rows are trusted; unknown rows get rerun.
    recovered = DiagnosticEngine(ledger_path, scale_exponent=1)
    recovered.scheme_version = scheme
    unknown = recovered.ledger.unknown_replay_jobs(scheme)
    print(f"phase C: {len(unknown)} unknown replay job(s) require rerun")
    # The recovery catalog must contain the in-flight (candidate, source)
    # key, mirroring a service catalog built before the kill.
    with open(catalog_path, encoding="utf-8") as handle:
        catalog_payload = json.load(handle)
    catalog_payload["replay_catalog"].append({
        "candidate_id": crash_candidate_id,
        "source_id": victim_source,
        "est_latency_us": 4200,
        "est_rows_scanned": 320,
        "stall_ms": 0,
    })
    with open(catalog_path, "w", encoding="utf-8") as handle:
        json.dump(catalog_payload, handle, indent=2, sort_keys=True)
    rerun_results = recovered.recover_pending_replays(config)
    for item in rerun_results:
        print(f"  rerun {item.job.candidate_id}@{item.job.source_id} ok={item.ok}")

    fresh_receipts = [
        {"candidate_id": item.job.candidate_id,
         "source_id": item.job.source_id, "ack": item.ack}
        for item in rerun_results if item.ok
    ]
    # Also verify all previously committed receipts against fresh replays.
    all_jobs = [
        ReplayJob(row["candidate_id"], row["source_id"], row["rewritten_sql"])
        for row in recovered.ledger.load_replay_jobs(scheme)
        if row["status"] == "committed"
    ]
    from slowquery_diag.scheduler import ReplayScheduler, LoopbackServiceProcess
    process = LoopbackServiceProcess(catalog_path)
    port = process.start()
    try:
        scheduler = ReplayScheduler(
            port=port, global_concurrency=2, per_source_quota=quota,
            socket_timeout_ms=2000,
        )
        verify_results = scheduler.run_all(all_jobs)
    finally:
        process.stop()
    fresh_receipts.extend(
        {"candidate_id": item.job.candidate_id,
         "source_id": item.job.source_id, "ack": item.ack}
        for item in verify_results if item.ok
    )
    recovered.verify_committed_replays(fresh_receipts)
    print("phase C: E_REPLAY check passed - every replayed field matches the ledger")
    recovered.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
