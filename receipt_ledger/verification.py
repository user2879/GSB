from __future__ import annotations

import json
import shutil
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from .canonical import monotonic_ms
from .compute import forward_batch, param_digest
from .errors import LedgerError
from .injections import inject_dangling_reference, inject_snapshot_digest_mismatch, inject_spec_corruption
from .invalidation import propagate_changes
from .ledger import Ledger
from .net import free_loopback_port, port_is_open
from .planner import build_plan, build_plan_from_parts, verify_coverage
from .processes import WorkerPool
from .receipts import ReceiptContext, normalize_receipts, read_jsonl
from .replay import replay_ledger
from .scheduler import Lease, LeaseScheduler
from .spec import load_bundle


def _expect_code(code: str, fn: Any, *args: Any) -> dict[str, Any]:
    try:
        fn(*args)
    except LedgerError as exc:
        if exc.code != code:
            raise AssertionError(f"expected {code}, got {exc.code}: {exc}") from exc
        return {"ok": True, "code": exc.code, "details": exc.details, "message": str(exc)}
    raise AssertionError(f"expected {code}, but no error was raised")


def _max_water(scheduler: LeaseScheduler, stop: threading.Event, observed: dict[str, Any]) -> None:
    observed["max_global"] = 0
    observed["max_worker"] = {}
    while not stop.is_set():
        water = scheduler.water_snapshot()
        observed["max_global"] = max(observed["max_global"], water["global"]["running"])
        for worker_id, item in water["workers"].items():
            observed["max_worker"][worker_id] = max(observed["max_worker"].get(worker_id, 0), item["running"])
        time.sleep(0.001)


def run_verification(
    sample_dir: str | Path = "samples/local",
    runtime_dir: str | Path = ".runtime/verify",
    *,
    start_processes: bool = True,
) -> dict[str, Any]:
    sample_path = Path(sample_dir)
    runtime = Path(runtime_dir)
    if runtime.exists():
        shutil.rmtree(runtime)
    (runtime / "snapshots").mkdir(parents=True)
    result: dict[str, Any] = {"started_monotonic_ms": monotonic_ms(), "stages": {}}

    bundle = load_bundle(sample_path)
    plan = build_plan(bundle)
    verify_coverage(plan, bundle.requests)
    result["stages"]["spec"] = {"workers": len(bundle.workers), "requests": len(bundle.requests), "batches": len(plan.batches)}

    shuffled_requests = tuple(reversed(bundle.requests))
    reordered = build_plan_from_parts(tuple(reversed(bundle.workers)), shuffled_requests, bundle.rules.default_batch_size)
    if reordered.canonical() != plan.canonical():
        raise AssertionError("计划在字典/派发顺序变化后未保持逐字节一致")
    result["stages"]["deterministic_plan"] = {"canonical_bytes": len(plan.canonical()), "batch_ids": [b.batch_id for b in plan.batches]}

    digests = {worker.worker_id: param_digest(worker) for worker in bundle.workers}
    versions = {worker.worker_id: worker.param_version for worker in bundle.workers}
    context = ReceiptContext.from_plan(plan, versions, digests)
    normalized_counts = {}
    for version in ("r1", "r2", "r3"):
        normalized = normalize_receipts(read_jsonl(sample_path / f"receipts_{version}.jsonl"), context)
        normalized_counts[version] = {
            "count": len(normalized),
            "backfilled_fields": normalized[0]["backfilled_fields"],
            "source_format_version": normalized[0]["source_format_version"],
        }
    result["stages"]["receipt_normalization"] = normalized_counts

    request_map = bundle.request_map()
    worker_map = bundle.worker_map()
    order_checks = []
    for batch in plan.batches:
        first = forward_batch(worker_map[batch.worker_id], batch, request_map)
        reversed_ids = tuple(reversed(batch.request_ids))
        reversed_seqs = tuple(batch.arrival_seqs[batch.request_ids.index(request_id)] for request_id in reversed_ids)
        reversed_batch = replace(
            batch,
            request_ids=reversed_ids,
            arrival_seqs=reversed_seqs,
            stable_ranks=dict(batch.stable_ranks),
        )
        second = forward_batch(worker_map[batch.worker_id], reversed_batch, request_map)
        order_checks.append(first["output_digest"] == second["output_digest"])
    if not all(order_checks):
        raise AssertionError("同批请求顺序变化改变了输出摘要")
    result["stages"]["order_invariance"] = {"batches_checked": len(order_checks), "all_equal": True}

    pool: WorkerPool | None = None
    if start_processes:
        loopback_dir = runtime / "loopback-sample"
        shutil.copytree(sample_path, loopback_dir, dirs_exist_ok=True)
        worker_declaration = json.loads((loopback_dir / "workers.json").read_text(encoding="utf-8"))
        for worker_decl in worker_declaration["workers"]:
            worker_decl["port"] = free_loopback_port()
        (loopback_dir / "workers.json").write_text(
            json.dumps(worker_declaration, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        service_bundle = load_bundle(loopback_dir)
        service_worker_map = service_bundle.worker_map()
        pool = WorkerPool(service_bundle.workers, loopback_dir).start_all()
        try:
            health_result = pool.forward(plan.batches[0].worker_id, plan.batches[0])
            local_result = forward_batch(worker_map[plan.batches[0].worker_id], plan.batches[0], request_map)
            if health_result["output_digest"] != local_result["output_digest"]:
                raise AssertionError("回环工作进程 NumPy 结果与本地重算不一致")
            worker_receipts = normalize_receipts(health_result.get("receipt_lines", []), context)
            if len(worker_receipts) != len(plan.batches[0].request_ids):
                raise AssertionError("工作进程回传的 r3 回执数量与批内请求数不一致")
            result["stages"]["loopback_workers"] = {
                "started": len(pool.processes),
                "host": "127.0.0.1",
                "first_worker_pid": health_result.get("worker_pid"),
                "output_matches_numpy": True,
                "r3_receipt_lines": len(health_result.get("receipt_lines", [])),
                "receipt_normalization_ok": True,
            }

            ledger = Ledger(runtime / "ledger.sqlite3", reset=True)
            ledger.initialize_bundle(service_bundle)
            scheduler = LeaseScheduler(
                {worker.worker_id: worker.quota for worker in bundle.workers},
                bundle.rules.global_concurrency,
                bundle.rules.lease_ttl_ms,
            )
            observed: dict[str, Any] = {"max_global": 0, "max_worker": {}}
            stop_monitor = threading.Event()
            monitor = threading.Thread(target=_max_water, args=(scheduler, stop_monitor, observed), daemon=True)
            monitor.start()

            def process_batch(batch: Any, lease: Any) -> dict[str, Any]:
                assert pool is not None
                computed = pool.forward(batch.worker_id, batch)
                computed["worker_snapshot"] = service_worker_map[batch.worker_id].to_json()
                return computed

            def finish_batch(batch: Any, lease: Any, computed: dict[str, Any]) -> None:
                ledger.commit_batch_result(batch, lease, computed, source_format="r3")

            scheduler.run(
                plan.batches,
                process_batch,
                on_grant=ledger.record_dispatch,
                on_finish=finish_batch,
            )
            stop_monitor.set()
            monitor.join(timeout=1)
            if scheduler.errors:
                raise AssertionError(f"初始调度存在失败批次: {scheduler.errors}")
            if observed["max_global"] > bundle.rules.global_concurrency:
                raise AssertionError("全局并发上限被突破")
            for worker in bundle.workers:
                if observed["max_worker"].get(worker.worker_id, 0) > worker.quota:
                    raise AssertionError(f"工作进程 {worker.worker_id} 配额被突破")
            queued = [event for event in scheduler.events if event["code"] == "wait_finished"]
            event_global = 0
            event_workers: dict[str, int] = {}
            event_max_global = 0
            event_max_worker: dict[str, int] = {}
            for event in scheduler.events:
                if event.get("code") == "granted":
                    event_global += 1
                    worker_id = event["worker_id"]
                    event_workers[worker_id] = event_workers.get(worker_id, 0) + 1
                    event_max_global = max(event_max_global, event_global)
                    event_max_worker[worker_id] = max(event_max_worker.get(worker_id, 0), event_workers[worker_id])
                elif event.get("code") == "released":
                    event_global = max(0, event_global - 1)
                    worker_id = event["worker_id"]
                    event_workers[worker_id] = max(0, event_workers.get(worker_id, 0) - 1)
            result["stages"]["scheduling"] = {
                "max_observed": observed,
                "event_max_global": event_max_global,
                "event_max_worker": event_max_worker,
                "global_limit": bundle.rules.global_concurrency,
                "quotas": {w.worker_id: w.quota for w in bundle.workers},
                "queue_wait_records": len(queued),
                "water": scheduler.water_snapshot()["global"]["water_level"],
            }

            replay_report = replay_ledger(ledger)
            good_snapshot = ledger.export_snapshot(runtime / "snapshots" / "good-v1.json", "snapshot-good-v1")
            result["stages"]["persistence_replay"] = {"committed_batches": len(replay_report), "snapshot_digest": good_snapshot["content_digest"]}

            alpha = next(worker for worker in bundle.workers if worker.worker_id == "w-alpha")
            alpha_v2 = replace(alpha, param_version=2)
            workers_v2 = tuple(alpha_v2 if worker.worker_id == alpha.worker_id else worker for worker in bundle.workers)
            bundle_v2 = replace(bundle, workers=workers_v2)
            plan_v2 = build_plan(bundle_v2)
            invalidation = propagate_changes(bundle, bundle_v2, plan, plan_v2)
            ledger.save_invalidation_rows("epoch-param-v2", invalidation.propagation_rows("epoch-param-v2"))
            alpha_batches = [batch for batch in plan_v2.batches if batch.worker_id == alpha.worker_id]
            frozen_before = {
                row["batch_id"]: row["frozen_output_digest"]
                for row in ledger.conn.execute("SELECT batch_id,frozen_output_digest FROM batches WHERE worker_id=?", (alpha.worker_id,))
            }
            v2_scheduler = LeaseScheduler({alpha.worker_id: alpha.quota}, 1, bundle.rules.lease_ttl_ms)
            v2_worker_map = bundle_v2.worker_map()

            def process_v2(batch: Any, lease: Any) -> dict[str, Any]:
                computed = forward_batch(v2_worker_map[batch.worker_id], batch, bundle_v2.request_map())
                computed["worker_snapshot"] = v2_worker_map[batch.worker_id].to_json()
                return computed

            v2_rows = [row for row in invalidation.propagation_rows("epoch-param-v2") if row["batch_id"] in {b.batch_id for b in alpha_batches}]
            v2_scheduler.run(
                alpha_batches,
                process_v2,
                on_grant=ledger.record_dispatch,
                on_finish=lambda batch, lease, computed: ledger.commit_batch_result(
                    batch, lease, computed, propagation_rows=v2_rows
                ),
            )
            if v2_scheduler.errors:
                raise AssertionError(f"失效重算失败: {v2_scheduler.errors}")
            frozen_after = {
                row["batch_id"]: row["frozen_output_digest"]
                for row in ledger.conn.execute("SELECT batch_id,frozen_output_digest FROM batches WHERE worker_id=?", (alpha.worker_id,))
            }
            if frozen_before != frozen_after:
                raise AssertionError("冻结旧批次摘要发生变化")
            replay_ledger(ledger)
            result["stages"]["invalidation"] = {
                "source_changes": len(invalidation.source_changes),
                "invalid_batches": invalidation.invalid_batch_ids,
                "invalid_requests": invalidation.invalid_request_ids,
                "recompute_batches": invalidation.recompute_batch_ids,
                "reattribute_requests": invalidation.reattribute_request_ids,
                "frozen_digests_unchanged": frozen_before == frozen_after,
            }

            beta = next(worker for worker in bundle.workers if worker.worker_id == "w-beta")
            beta_reshaped = replace(beta, batch_size=1)
            bundle_v3 = replace(bundle, workers=tuple(beta_reshaped if w.worker_id == beta.worker_id else w for w in bundle.workers))
            plan_v3 = build_plan(bundle_v3)
            shape_report = propagate_changes(bundle_v2, bundle_v3, plan_v2, plan_v3)
            result["stages"]["batch_size_propagation"] = {
                "new_batch_count": len([b for b in plan_v3.batches if b.worker_id == beta.worker_id]),
                "recompute_batches": [b for b in shape_report.recompute_batch_ids if b in {x.batch_id for x in plan_v3.batches if x.worker_id == beta.worker_id}],
                "reattribute_requests": shape_report.reattribute_request_ids,
            }

            gamma_batch = next(batch for batch in plan.batches if batch.worker_id == "w-gamma")
            crash_now = monotonic_ms()
            crash_lease = Lease(
                lease_id="lease-crash-gamma-0001",
                batch_id=gamma_batch.batch_id,
                worker_id=gamma_batch.worker_id,
                version=1,
                granted_monotonic_ms=crash_now - 20,
                expires_monotonic_ms=crash_now + bundle.rules.lease_ttl_ms,
                ttl_ms=bundle.rules.lease_ttl_ms,
            )
            ledger.record_dispatch(gamma_batch, crash_lease)
            pool.kill_worker("w-gamma")
            deadline = time.monotonic() + 2
            gamma_port = service_worker_map["w-gamma"].port
            while port_is_open("127.0.0.1", gamma_port) and time.monotonic() < deadline:
                time.sleep(0.02)
            port_closed = not port_is_open("127.0.0.1", gamma_port)
            unknown = ledger.mark_interrupted_batches_unknown("verification_force_kill")
            gamma_unknown = gamma_batch.batch_id in unknown
            pool.restart_worker("w-gamma")

            recovery_scheduler = LeaseScheduler({"w-gamma": 1}, 1, bundle.rules.lease_ttl_ms)
            recovery_scheduler.run(
                [gamma_batch],
                process_batch,
                on_grant=ledger.record_dispatch,
                on_finish=finish_batch,
            )
            if recovery_scheduler.errors:
                raise AssertionError(f"恢复重跑失败: {recovery_scheduler.errors}")
            replay_after = replay_ledger(ledger)
            final_snapshot = ledger.export_snapshot(runtime / "snapshots" / "recovered.json", "snapshot-recovered")
            restored = Ledger.restore_snapshot_to_path(runtime / "snapshots" / "recovered.json", runtime / "restored.sqlite3")
            try:
                restored_replay = replay_ledger(restored)
                restored.verify_digest()
            finally:
                restored.close()
            result["stages"]["kill_recovery"] = {
                "forced_kill": True,
                "loopback_port_closed_after_kill": port_closed,
                "unknown_batch_marked": gamma_unknown,
                "unknown_batch_ids": unknown,
                "replayed_batches_after_recovery": len(replay_after),
                "restored_replay_batches": len(restored_replay),
                "snapshot_format": final_snapshot["snapshot_format"],
            }
            ledger.close()
        finally:
            pool.stop_all()

    result["stages"]["injection_spec"] = _expect_code("E_SPEC", inject_spec_corruption, runtime)
    result["stages"]["injection_ref"] = _expect_code("E_REF", inject_dangling_reference, sample_path)
    result["stages"]["injection_snapshot"] = _expect_code("E_SNAPSHOT", inject_snapshot_digest_mismatch, runtime)
    result["ok"] = True
    result["finished_monotonic_ms"] = monotonic_ms()
    result["elapsed_monotonic_ms"] = result["finished_monotonic_ms"] - result["started_monotonic_ms"]
    return result
