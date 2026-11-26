from __future__ import annotations

import threading
import time

from topic_ledger.engine import TopicLedgerEngine, direct_dispatch
from topic_ledger.samplegen import build_sample_bundle


def test_midflight_lease_recovery_fences_late_receipts_and_releases_slots(tmp_path) -> None:
    bundle = build_sample_bundle()
    bundle["inference"]["lease_ttl_ms"] = 5
    entered_lock = threading.Lock()
    entered_count = 0
    all_entered = threading.Event()

    def slow_dispatch(node_id: str, request: dict):
        nonlocal entered_count
        with entered_lock:
            entered_count += 1
            if entered_count == 4:
                all_entered.set()
        time.sleep(0.25)
        return direct_dispatch(node_id, request)

    engine = TopicLedgerEngine(
        topology=bundle["topology"],
        inference=bundle["inference"],
        vocab=bundle["vocab"],
        documents=bundle["documents"],
        state_dir=tmp_path / "midflight",
        dispatcher=slow_dispatch,
        clock="real",
        global_concurrency_override=4,
    )

    command_thread = threading.Thread(
        target=engine.handle_command,
        args=({"command_id": "slow-run", "op": "run_sampling", "batch_id": "midflight"},),
        daemon=True,
    )
    command_thread.start()
    assert all_entered.wait(timeout=3.0)
    time.sleep(0.03)

    recovery = engine.recover_leases()
    assert len(recovery["expired"]) == 4
    assert engine.slots.metrics()["global_held"] == 0

    command_thread.join(timeout=5.0)
    assert not command_thread.is_alive()
    report = engine.canonical_report()
    assert len(report["pending"]) == 4
    assert report["code_counts"]["E_LEASE_EXPIRED"] == 4
    assert report["shard_receipts"] == {}
    assert engine.slots.metrics()["global_held"] == 0
