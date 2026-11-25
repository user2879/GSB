"""端到端编排：声明 -> 计划 -> 真实回环调度 -> 三格式回执 -> 账本。"""

from .clock import MonotonicClock
from .invalidation import compute_invalidation, compare_with_full_recompute
from .ledger import Ledger, digest_of
from .leases import LeaseManager
from .plan import build_plan, plan_fingerprint, stable_hash
from .processes import WorkerPool
from .receipts import normalize_batch, encode_receipt
from .scheduler import Scheduler


def make_receipt_records(spec, plan, batch_result):
    """为批内每个请求生成 r1/r2/r3 三种历史回执并归一校验。"""
    supersession = spec.supersession_map()
    known_ids = spec.all_request_ids()
    rank_index = {request_id: rank for rank, request_id in enumerate(plan.request_ids)}
    context = {
        "batch_size": spec.batch_size,
        "known_request_ids": known_ids,
        "worker_id": plan.worker_id,
        "param_version": batch_result["param_version"],
        "rank_of": lambda request_id: rank_index[request_id],
        "dup_ref_of": lambda request_id: supersession.get(request_id),
        "location": f"/batches/{plan.batch_id}",
    }
    raw_r1 = [
        {"batch_id": plan.batch_id,
         "request_id": item["request_id"],
         "output_digest": item["output_digest"]}
        for item in batch_result["results"]
    ]
    normalized = normalize_batch(raw_r1, context)

    records = []
    for norm in normalized:
        for fmt in (1, 2, 3):
            payload = encode_receipt(norm, fmt)
            records.append({
                "fmt": fmt,
                "payload": payload,
                "normalized": norm.to_dict(),
            })
    return records, normalized


def run_pipeline(spec, spec_dict, db_path, clock=None, on_event=None,
                 snapshot_note="full_run"):
    clock = clock or MonotonicClock()
    plans = build_plan(spec)
    delivery_index = {d.request_id: d for d in spec.deliveries}

    events = []
    if on_event is None:
        on_event = events.append

    ledger = Ledger(db_path)
    spec_digest = stable_hash(spec_dict, digest_size=16)
    snapshot_version = ledger.begin_snapshot(
        spec_digest, clock.now_ms(), plan_fingerprint(plans), snapshot_note)

    lease_manager = LeaseManager(spec, clock, on_lease_change=on_event)
    lease_manager.start_reaper()
    try:
        with WorkerPool(spec, spec_dict) as pool:
            scheduler = Scheduler(spec, pool, lease_manager, clock, on_event=on_event)
            results = scheduler.run_plan(plans, delivery_index)
            audit = scheduler.audit_report()
    finally:
        lease_manager.stop_reaper()

    all_normalized = []
    for plan, result in zip(plans, results):
        records, normalized = make_receipt_records(spec, plan, result)
        all_normalized.extend(normalized)
        lease_snapshot = lease_manager.snapshot_leases()
        ledger.commit_batch(snapshot_version, result, lease_snapshot, records)

    replay = ledger.replay_verify(snapshot_version)
    ledger.close()
    return {
        "snapshot_version": snapshot_version,
        "plans": [plan.to_dict() for plan in plans],
        "results": results,
        "normalized": [item.to_dict() for item in all_normalized],
        "audit": audit,
        "events": events,
        "replay": replay,
        "db_path": db_path,
    }


def run_invalidation(old_spec, old_spec_dict, new_spec, new_spec_dict,
                    old_outcome, db_path, clock=None, on_event=None):
    """失效传播 + 仅失效子集真实重算 + 与全量重算逐项比较。"""
    clock = clock or MonotonicClock()
    report = {}
    from .plan import build_plan as _build
    old_plans = _build(old_spec)
    new_plans = _build(new_spec)
    steps, untouched = compute_invalidation(old_spec, new_spec, old_plans, new_plans)
    report["steps"] = [step.to_dict() for step in steps]

    prior_index = {item["batch_id"]: item for item in old_outcome["results"]}
    reused = []
    for batch_id in sorted(untouched):
        plan = untouched[batch_id]
        prior = prior_index[batch_id]
        reused.append({
            "batch_id": batch_id,
            "batch_seq": plan.batch_seq,
            "worker_id": plan.worker_id,
            "results": prior["results"],
            "reused": True,
        })

    invalid_plans = [plan for plan in new_plans if plan.batch_id not in untouched]
    delivery_index = {d.request_id: d for d in new_spec.deliveries}

    ledger = Ledger(db_path)
    snapshot_version = ledger.begin_snapshot(
        digest_of(new_spec_dict), clock.now_ms(),
        plan_fingerprint(new_plans), "incremental_run")
    lease_manager = LeaseManager(new_spec, clock)
    lease_manager.start_reaper()
    try:
        with WorkerPool(new_spec, new_spec_dict) as pool:
            scheduler = Scheduler(new_spec, pool, lease_manager, clock,
                                  on_event=on_event)
            partial = scheduler.run_plan(invalid_plans, delivery_index) if invalid_plans else []
    finally:
        lease_manager.stop_reaper()

    for result in partial:
        plan = next(plan for plan in new_plans if plan.batch_id == result["batch_id"])
        records, _ = make_receipt_records(new_spec, plan, result)
        ledger.commit_batch(snapshot_version, result,
                            lease_manager.snapshot_leases(), records)
    for item in reused:
        plan = untouched[item["batch_id"]]
        synthetic = {
            "batch_id": item["batch_id"],
            "batch_seq": item["batch_seq"],
            "worker_id": item["worker_id"],
            "param_version": new_spec.worker(item["worker_id"]).param_version,
            "elapsed_ms": 0,
            "results": item["results"],
            "request_ids": list(plan.request_ids),
            "arrival_seqs": list(plan.arrival_seqs),
        }
        records, _ = make_receipt_records(new_spec, plan, synthetic)
        ledger.commit_batch(snapshot_version, synthetic,
                            lease_manager.snapshot_leases(), records)

    full_outcome = run_pipeline(
        new_spec, new_spec_dict,
        db_path.replace(".sqlite3", ".full.sqlite3"), clock=clock)
    comparison = compare_with_full_recompute(
        partial + reused, full_outcome["results"])

    reused_pairs = [
        {"batch_id": item["batch_id"],
         "digest_equal": prior_index[item["batch_id"]]["results"] == item["results"]}
        for item in reused
    ]
    report.update({
        "snapshot_version": snapshot_version,
        "invalidated_batches": [plan.batch_id for plan in invalid_plans],
        "reused_batches": reused_pairs,
        "comparison_with_full_recompute": comparison,
        "replay": ledger.replay_verify(snapshot_version),
    })
    ledger.close()
    return report
