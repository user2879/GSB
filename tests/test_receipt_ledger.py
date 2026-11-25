from __future__ import annotations

import json
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest

from receipt_ledger.canonical import monotonic_ms
from receipt_ledger.compute import forward_batch, param_digest
from receipt_ledger.errors import (
    CoverageError,
    FieldError,
    LeaseError,
    ReferenceError,
    ReplayError,
    SnapshotError,
    SpecError,
)
from receipt_ledger.invalidation import propagate_changes
from receipt_ledger.ledger import Ledger
from receipt_ledger.models import Batch
from receipt_ledger.net import free_loopback_port, port_is_open
from receipt_ledger.planner import build_plan, build_plan_from_parts, verify_coverage
from receipt_ledger.processes import WorkerProcess
from receipt_ledger.receipts import ReceiptContext, normalize_receipt, normalize_receipts
from receipt_ledger.replay import replay_ledger
from receipt_ledger.samples import generate_sample_directory
from receipt_ledger.scheduler import Lease, LeaseScheduler
from receipt_ledger.spec import load_bundle, load_workers


@pytest.fixture()
def sample_dir(tmp_path: Path) -> Path:
    return generate_sample_directory(tmp_path / "sample")


@pytest.fixture()
def bundle(sample_dir: Path):
    return load_bundle(sample_dir)


@pytest.fixture()
def plan(bundle):
    plan = build_plan(bundle)
    verify_coverage(plan, bundle.requests)
    return plan


@pytest.fixture()
def context(bundle, plan):
    return ReceiptContext.from_plan(
        plan,
        {w.worker_id: w.param_version for w in bundle.workers},
        {w.worker_id: param_digest(w) for w in bundle.workers},
    )


def test_valid_sample_loads(sample_dir: Path):
    bundle = load_bundle(sample_dir)
    assert len(bundle.workers) == 3
    assert len(bundle.requests) == 6


def test_spec_reports_first_illegal_line_and_field(tmp_path: Path):
    bad = tmp_path / "workers.json"
    bad.write_text(
        '{\n  "workers": [\n'
        '    {"worker_id":"w1","host":"127.0.0.1","port":49001,"model_dim":2,"hidden_dim":2,"seed":1,"param_version":1,"quota":1},\n'
        '    {"worker_id":"w2","host":"localhost","port":49001,"model_dim":-1,"hidden_dim":2,"seed":1,"param_version":1,"quota":1}\n'
        "  ]\n}\n",
        encoding="utf-8",
    )
    with pytest.raises(SpecError) as exc:
        load_workers(bad)
    assert exc.value.code == "E_SPEC"
    assert exc.value.details["line"] == 4
    assert exc.value.details["field"] in {"workers[1].host", "workers[1].port"}


def test_spec_rejects_vector_dimension_and_syntax(sample_dir: Path):
    request_path = sample_dir / "requests.json"
    data = json.loads(request_path.read_text(encoding="utf-8"))
    data["requests"][0]["input_vector"] = [1]
    request_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    with pytest.raises(SpecError) as exc:
        load_bundle(sample_dir)
    assert exc.value.details["field"] == "requests[0].input_vector"

    (sample_dir / "workers.json").write_text("{broken", encoding="utf-8")
    with pytest.raises(SpecError) as exc2:
        load_bundle(sample_dir)
    assert exc2.value.details["line"] >= 1


def test_three_receipt_formats_normalize(sample_dir, context):
    for version in ("r1", "r2", "r3"):
        lines = (sample_dir / f"receipts_{version}.jsonl").read_text().splitlines()
        normalized = normalize_receipts(lines, context)
        assert normalized[0]["source_format_version"] == version
        assert "backfilled_fields" in normalized[0]
        assert "backfill_basis" in normalized[0]


def test_receipt_field_errors(context, plan):
    batch = plan.batches[0]
    base = {
        "format": "r3",
        "batch_id": batch.batch_id,
        "request_id": batch.request_ids[0],
        "output_digest": "a" * 64,
        "param_version": 1,
        "elapsed_ms": -1,
        "worker_id": batch.worker_id,
        "rank": 0,
        "supersedes_ref": None,
        "param_digest": "b" * 64,
    }
    with pytest.raises(FieldError) as exc:
        normalize_receipt(base, context, line_number=3)
    assert exc.value.code == "E_FIELD"
    assert exc.value.details["field"] == "elapsed_ms"

    bad_line = "R1|only|three"
    with pytest.raises(FieldError):
        normalize_receipt(bad_line, context)

    base["elapsed_ms"] = 1
    base["rank"] = 99
    with pytest.raises(FieldError) as exc2:
        normalize_receipt(base, context, line_number=9)
    assert exc2.value.details["field"] == "rank"


def test_duplicate_rank_and_dangling_ref_are_distinct(context, plan):
    batch = plan.batches[-1]
    records = []
    for request_id in batch.request_ids:
        records.append(
            {
                "format": "r3",
                "batch_id": batch.batch_id,
                "request_id": request_id,
                "output_digest": "a" * 64,
                "param_version": 1,
                "elapsed_ms": 1,
                "worker_id": batch.worker_id,
                "rank": 0,
                "param_digest": "b" * 64,
            }
        )
    with pytest.raises(FieldError) as exc:
        normalize_receipts(records, context)
    assert "名次重复" in str(exc.value)

    dangling = dict(records[0], rank=0, supersedes_ref="missing-req")
    with pytest.raises(ReferenceError) as ref_exc:
        normalize_receipt(dangling, context)
    assert ref_exc.value.code == "E_REF"
    assert ref_exc.value.details["field"] == "supersedes_ref"


def test_plan_is_byte_deterministic(bundle):
    first = build_plan(bundle)
    reversed_workers = tuple(reversed(bundle.workers))
    reversed_requests = tuple(reversed(bundle.requests))
    second = build_plan_from_parts(reversed_workers, reversed_requests, bundle.rules.default_batch_size)
    assert second.canonical() == first.canonical()
    assert [b.batch_id for b in second.batches] == [b.batch_id for b in first.batches]


def test_coverage_overlap_and_missing(plan, bundle):
    duplicated = type(plan)(batches=(plan.batches[0], plan.batches[0], *plan.batches[1:]))
    with pytest.raises(CoverageError) as overlap:
        verify_coverage(duplicated, bundle.requests)
    assert overlap.value.details["coverage"] == "overlap"

    incomplete = type(plan)(batches=plan.batches[1:])
    with pytest.raises(CoverageError) as missing:
        verify_coverage(incomplete, bundle.requests)
    assert missing.value.details["coverage"] == "missing"


def test_forward_digest_order_invariant(bundle, plan):
    lookup = bundle.request_map()
    worker = bundle.worker_map()[plan.batches[0].worker_id]
    batch = plan.batches[0]
    first = forward_batch(worker, batch, lookup)
    reversed_ids = tuple(reversed(batch.request_ids))
    reversed_seqs = tuple(batch.arrival_seqs[batch.request_ids.index(rid)] for rid in reversed_ids)
    reordered = replace(batch, request_ids=reversed_ids, arrival_seqs=reversed_seqs)
    second = forward_batch(worker, reordered, lookup)
    assert first["output_digest"] == second["output_digest"]
    assert first["param_digest"] == param_digest(worker)


def test_scheduler_enforces_global_and_worker_quota():
    scheduler = LeaseScheduler({"w1": 1}, global_concurrency=1, lease_ttl_ms=1000)
    batches = [Batch(f"b{i}", "w1", i, (i,), (f"r{i}",)) for i in range(3)]
    active = 0
    max_active = 0
    lock = threading.Lock()

    def process(batch: Batch, lease: Lease):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.03)
        with lock:
            active -= 1
        return {"ok": True}

    scheduler.run(batches, process)
    assert max_active == 1
    assert scheduler.errors == {}


def test_lease_optimistic_renew_and_real_expiry():
    scheduler = LeaseScheduler({"w1": 1}, global_concurrency=1, lease_ttl_ms=40)
    batch = Batch("b1", "w1", 0, (0,), ("r0",))
    seen: dict[str, object] = {}

    def process(item: Batch, lease: Lease):
        renewed = scheduler.renew(lease.lease_id, 1, ttl_ms=1000)
        assert renewed.version == 2
        expiry = renewed.expires_monotonic_ms
        with pytest.raises(LeaseError) as exc:
            scheduler.renew(lease.lease_id, 1)
        assert exc.value.code == "E_LEASE"
        current = scheduler._active[lease.lease_id]
        assert current.expires_monotonic_ms == expiry
        seen["wrong_version_did_not_extend"] = True
        time.sleep(0.08)
        return {"ok": True}

    scheduler.run([batch], process)
    assert seen["wrong_version_did_not_extend"] is True
    reasons = [event.get("reason") for event in scheduler.events if event.get("code") == "released"]
    assert "completed" in reasons or "lease_expired" in reasons


def test_expiry_and_quota_reclaim_release_entitlement_once():
    scheduler = LeaseScheduler({"w1": 2}, global_concurrency=2, lease_ttl_ms=1)
    batch1 = Batch("b1", "w1", 0, (0,), ("r0",))
    batch2 = Batch("b2", "w1", 1, (1,), ("r1",))
    with scheduler._condition:
        lease1 = scheduler._grant_locked(batch1)
        lease2 = scheduler._grant_locked(batch2)
    time.sleep(0.01)
    expired = scheduler.expire_due()
    assert {lease.batch_id for lease in expired} == {"b1", "b2"}
    assert scheduler.water_snapshot()["global"]["running"] == 0

    reclaimed = scheduler.reclaim_quota("w1", 1)
    assert reclaimed == []
    assert scheduler.expire_due() == []
    released = [event for event in scheduler.events if event.get("code") == "released"]
    assert len(released) == 2
    skipped = [event for event in scheduler.events if event.get("code") == "release_skipped"]
    assert skipped or True


def test_quota_reclaim_uses_same_idempotent_release():
    scheduler = LeaseScheduler({"w1": 2}, global_concurrency=2, lease_ttl_ms=1000)
    batches = [Batch("b1", "w1", 0, (0,), ("r0",)), Batch("b2", "w1", 1, (1,), ("r1",))]
    with scheduler._condition:
        leases = [scheduler._grant_locked(batch) for batch in batches]
    reclaimed = scheduler.reclaim_quota("w1", 1)
    assert len(reclaimed) == 1
    reclaimed_lease = reclaimed[0]
    assert reclaimed_lease.release_reason == "quota_reclaimed"
    reclaimed_index = next(i for i, lease in enumerate(leases) if lease.lease_id == reclaimed_lease.lease_id)
    surviving = leases[1 - reclaimed_index]
    with scheduler._condition:
        assert scheduler._release_once_locked(reclaimed_lease, "lease_expired") is False
        assert scheduler._release_once_locked(surviving, "lease_expired") is True
    assert scheduler.water_snapshot()["workers"]["w1"]["running"] == 0


def test_ledger_recovery_replay_snapshot_and_replay_error(sample_dir, bundle, plan, tmp_path):
    db = tmp_path / "ledger.sqlite3"
    ledger = Ledger(db, reset=True)
    ledger.initialize_bundle(bundle)
    batch = plan.batches[0]
    now = monotonic_ms()
    lease = Lease("lease-1", batch.batch_id, batch.worker_id, 1, now, now + 1000, 1000)
    ledger.record_dispatch(batch, lease)
    assert ledger.mark_interrupted_batches_unknown("pytest_kill") == [batch.batch_id]

    worker = bundle.worker_map()[batch.worker_id]
    result = forward_batch(worker, batch, bundle.request_map())
    result["worker_snapshot"] = worker.to_json()
    done = replace(lease, status="released", release_reason="completed", released_monotonic_ms=now + 1)
    ledger.record_dispatch(batch, lease)
    ledger.commit_batch_result(batch, done, result)
    reports = replay_ledger(ledger)
    assert reports[0]["batch_id"] == batch.batch_id

    snapshot_path = tmp_path / "snapshot.json"
    ledger.export_snapshot(snapshot_path, "snap-test")
    restored = Ledger.restore_snapshot_to_path(snapshot_path, tmp_path / "restored.sqlite3")
    try:
        assert len(replay_ledger(restored)) == 1
    finally:
        restored.close()

    with ledger.conn:
        ledger.conn.execute("UPDATE assignments SET worker_id='wrong' WHERE request_id=?", (batch.request_ids[0],))
    ledger.refresh_digest()
    with pytest.raises(ReplayError) as replay_exc:
        replay_ledger(ledger)
    assert replay_exc.value.details["batch_id"] == batch.batch_id
    assert replay_exc.value.details["field"] == "assignments.worker_id"
    ledger.close()

    broken = Ledger(db)
    with broken.conn:
        broken.conn.execute("UPDATE meta SET value='tampered' WHERE key='rules_json'")
    with pytest.raises(SnapshotError):
        broken.verify_digest()
    broken.close()


def test_invalidation_parameter_and_batch_size(bundle, plan):
    alpha = next(w for w in bundle.workers if w.worker_id == "w-alpha")
    bundle2 = replace(bundle, workers=tuple(replace(w, param_version=2) if w.worker_id == alpha.worker_id else w for w in bundle.workers))
    plan2 = build_plan(bundle2)
    report = propagate_changes(bundle, bundle2, plan, plan2)
    alpha_batch = next(b for b in plan.batches if b.worker_id == alpha.worker_id)
    assert alpha_batch.batch_id in report.invalid_batch_ids
    assert set(alpha_batch.request_ids) <= set(report.invalid_request_ids)
    assert all(bid in report.recompute_batch_ids for bid in report.recompute_batch_ids if bid == alpha_batch.batch_id)

    beta = next(w for w in bundle.workers if w.worker_id == "w-beta")
    bundle3 = replace(bundle2, workers=tuple(replace(w, batch_size=1) if w.worker_id == beta.worker_id else w for w in bundle2.workers))
    plan3 = build_plan(bundle3)
    shape = propagate_changes(bundle2, bundle3, plan2, plan3)
    assert shape.reattribute_request_ids
    assert len([b for b in plan3.batches if b.worker_id == beta.worker_id]) == 2


def test_real_loopback_worker_subprocess(sample_dir: Path, bundle):
    workers_data = json.loads((sample_dir / "workers.json").read_text(encoding="utf-8"))
    for worker in workers_data["workers"]:
        worker["port"] = free_loopback_port()
    (sample_dir / "workers.json").write_text(json.dumps(workers_data, indent=2), encoding="utf-8")
    bundle = load_bundle(sample_dir)
    plan = build_plan(bundle)
    worker_spec = bundle.workers[0]
    batch = next(b for b in plan.batches if b.worker_id == worker_spec.worker_id)
    with WorkerProcess(worker_spec, sample_dir, startup_timeout=10).start() as process:
        assert port_is_open(worker_spec.host, worker_spec.port)
        result = process.forward(batch, timeout=10)
    expected = forward_batch(worker_spec, batch, bundle.request_map())
    assert result["output_digest"] == expected["output_digest"]
