"""依赖链失效传播：工作进程参数版本 -> 批次 -> 请求归属。

失效根因：
  - 工作进程种子 / 维度 / 定点口径变化（体现为该 worker 的规格摘要变化）；
  - 批大小或请求集变化导致的批成员变化；
  - 派工关系变化。
传播分三步稳定输出：step1 工作进程级根因，step2 批级失效，step3 请求级失效。
未失效批的回执与输出摘要保持逐字节不变（调用方不得触碰其账本记录）。
"""

from dataclasses import dataclass, field


@dataclass
class InvalidationReport:
    invalidated_batches: tuple = ()
    retired_batches: tuple = ()
    invalidated_requests: tuple = ()
    unaffected_batches: tuple = ()
    steps: tuple = ()
    batch_cause: dict = field(default_factory=dict)

    def rows_for_batch(self, batch_id, request_ids):
        """生成某失效批的 (batch_id, request_id, step, reason) 账本行。"""
        cause = self.batch_cause[batch_id]
        rows = []
        for rid in sorted(request_ids):
            rows.append((batch_id, rid, 1, cause))
            rows.append((batch_id, rid, 2, f"batch-invalidated:{batch_id}"))
            rows.append((batch_id, rid, 3, f"request-invalidated:{rid}<-{batch_id}<-{cause}"))
        return rows


def compute_invalidation(old_batches, old_worker_digests, plan, new_worker_digests):
    """对比账本旧状态与新计划，输出稳定的失效报告。

    old_batches: {batch_id: {"worker_id":..., "param_version":..., "request_ids":[...]}}
    old_worker_digests / new_worker_digests: {worker_id: spec_digest}
    plan: [PlannedBatch]
    """
    steps = []
    changed_workers = []
    for wid in sorted(new_worker_digests):
        old = old_worker_digests.get(wid)
        new = new_worker_digests[wid]
        if old is not None and old != new:
            changed_workers.append(wid)
            steps.append((1, "worker", wid, f"worker-spec-changed:{wid}"))

    invalidated = []
    batch_cause = {}
    plan_ids = set()
    for pb in sorted(plan, key=lambda b: b.batch_id):
        plan_ids.add(pb.batch_id)
        old = old_batches.get(pb.batch_id)
        if old is None:
            continue  # 新批：按未知处理，不属于失效传播
        causes = []
        if pb.worker_id in changed_workers:
            causes.append(f"worker-spec-changed:{pb.worker_id}")
        if old["worker_id"] != pb.worker_id:
            causes.append(f"worker-reassigned:{old['worker_id']}->{pb.worker_id}")
        if sorted(old["request_ids"]) != sorted(pb.request_ids):
            causes.append("membership-changed")
        if causes:
            invalidated.append(pb.batch_id)
            batch_cause[pb.batch_id] = ";".join(causes)
            steps.append((2, "batch", pb.batch_id, batch_cause[pb.batch_id]))

    retired = sorted(set(old_batches) - plan_ids)
    for bid in retired:
        batch_cause[bid] = "batch-retired"
        steps.append((2, "batch", bid, "batch-retired"))

    inv_requests = []
    plan_by_id = {pb.batch_id: pb for pb in plan}
    for bid in sorted(batch_cause):
        old = old_batches[bid]
        pb = plan_by_id.get(bid)
        new_ids = list(pb.request_ids) if pb else []
        for rid in sorted(set(old["request_ids"]) | set(new_ids)):
            inv_requests.append(rid)
            steps.append((3, "request", rid,
                          f"request-invalidated:{rid}<-{bid}<-{batch_cause[bid]}"))

    unaffected = tuple(sorted(
        bid for bid in plan_ids & set(old_batches) if bid not in batch_cause
    ))
    return InvalidationReport(
        invalidated_batches=tuple(sorted(invalidated)),
        retired_batches=tuple(retired),
        invalidated_requests=tuple(sorted(set(inv_requests))),
        unaffected_batches=unaffected,
        steps=tuple(steps),
        batch_cause=batch_cause,
    )
