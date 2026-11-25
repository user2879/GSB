"""并行批调度：全局 + 每工作进程双重活跃租约上限（需求 6）。

调度真实投射到 :class:`concurrent.futures.ThreadPoolExecutor`：
  * 全局上限用信号量强制，在途批次数 == 活跃租约数；
  * 每工作进程上限由 :class:`~batch_receipt_ledger.leases.LeaseManager`
    的配额等待强制；
  * 每次排队都记录单调时钟毫秒的等待开始/结束，并在运行期断言不超限。
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from .errors import LeaseError
from .leases import STATE_RELEASED_CANCELLED, STATE_RELEASED_DONE


class Scheduler:
    def __init__(self, spec, worker_pool, lease_manager, clock,
                 on_event=None, audit=True):
        self.spec = spec
        self.pool = worker_pool
        self.lease_manager = lease_manager
        self.clock = clock
        self.on_event = on_event
        self.audit = audit
        self._global_slots = threading.Semaphore(spec.global_max_active)
        self._inflight = 0
        self._inflight_lock = threading.Lock()
        self._max_observed_active = 0
        self._max_observed_inflight = 0
        self.wait_metrics = []

    def run_plan(self, plans, delivery_index):
        results = []
        max_workers = min(
            max(1, self.spec.global_max_active),
            max(1, sum(self.spec.effective_quota(w.worker_id) for w in self.spec.workers)),
            max(1, len(plans)),
        )
        with ThreadPoolExecutor(max_workers=max_workers,
                                thread_name_prefix="batch-dispatch") as executor:
            futures = [
                executor.submit(self._dispatch, plan, delivery_index)
                for plan in plans
            ]
            for future in as_completed(futures):
                results.append(future.result())
        results.sort(key=lambda item: item["batch_seq"])
        return results

    def _dispatch(self, plan, delivery_index):
        worker_id = plan.worker_id
        wait_start = self.clock.now_ms()
        self._emit({"event": "queued", "batch_id": plan.batch_id,
                    "worker_id": worker_id, "at": wait_start})

        self._global_slots.acquire()
        lease, lease_wait_ms = self.lease_manager.acquire(plan.batch_id, worker_id)
        wait_end = self.clock.now_ms()
        metric = {
            "batch_id": plan.batch_id,
            "worker_id": worker_id,
            "wait_start_ms": wait_start,
            "wait_end_ms": wait_end,
            "wait_ms": wait_end - wait_start,
            "lease_wait_ms": lease_wait_ms,
        }
        self.wait_metrics.append(metric)

        with self._inflight_lock:
            self._inflight += 1
            self._max_observed_inflight = max(self._max_observed_inflight, self._inflight)
            active = self.lease_manager.active_count()
            self._max_observed_active = max(self._max_observed_active, active)
            if self.audit:
                assert self._inflight <= self.spec.global_max_active
                assert active <= self.spec.global_max_active
                per_worker = sum(
                    1 for item in self.lease_manager.held_leases()
                    if item.worker_id == worker_id)
                assert per_worker <= self.spec.effective_quota(worker_id)

        self._emit({"event": "dispatched", **metric})
        try:
            return self._run_batch(plan, lease, delivery_index)
        finally:
            with self._inflight_lock:
                self._inflight -= 1
            self._global_slots.release()

    def _run_batch(self, plan, lease, delivery_index):
        worker = self.pool.get(plan.worker_id)
        with worker.client(self.spec.connect_timeout_ms,
                           self.spec.read_timeout_ms) as client:
            self.lease_manager.grant_on_worker(lease, client)
            items = [
                {"request_id": request_id,
                 "value": delivery_index[request_id].value}
                for request_id in plan.request_ids
            ]
            reply = client.request({
                "type": "batch",
                "batch_id": plan.batch_id,
                "lease_id": lease.lease_id,
                "param_version": self.spec.worker(plan.worker_id).param_version,
                "items": items,
            })
        if reply.get("type") == "error":
            self.lease_manager.cancel(lease.lease_id)
            raise LeaseError(f"工作进程批处理失败: {reply}", field="batch")

        now_ms = self.clock.now_ms()
        if now_ms >= lease.expires_at:
            self.lease_manager.release(lease.lease_id, STATE_RELEASED_CANCELLED)
            raise LeaseError(
                f"批 {plan.batch_id} 完成时租约已过期", field="expires_at")
        self.lease_manager.release(lease.lease_id, STATE_RELEASED_DONE)

        result = {
            "batch_id": plan.batch_id,
            "batch_seq": plan.batch_seq,
            "worker_id": plan.worker_id,
            "lease_id": lease.lease_id,
            "lease_state_final": STATE_RELEASED_DONE,
            "param_version": reply["param_version"],
            "elapsed_ms": reply["elapsed_ms"],
            "results": reply["results"],
            "request_ids": list(plan.request_ids),
            "arrival_seqs": list(plan.arrival_seqs),
        }
        self._emit({"event": "completed", "batch_id": plan.batch_id,
                    "worker_id": plan.worker_id})
        return result

    def _emit(self, payload):
        if self.on_event is not None:
            self.on_event(payload)

    def audit_report(self):
        return {
            "global_max_active_leases": self.spec.global_max_active,
            "max_observed_active_leases": self._max_observed_active,
            "max_observed_inflight_batches": self._max_observed_inflight,
            "wait_metrics": list(self.wait_metrics),
        }
