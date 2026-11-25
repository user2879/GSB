"""参数版本/种子/批大小变更的失效传播与子集重算（需求 7）。

失效沿 ``工作进程参数版本 -> 批次 -> 请求归属`` 传播，输出顺序按
工作进程标识序、批序号稳定排序；未失效批的输出摘要保持逐字节不变。
"""

from dataclasses import dataclass

from .plan import build_plan, plan_fingerprint


@dataclass
class InvalidationStep:
    worker_id: str
    batch_seq: int
    batch_id: str
    reason: str
    request_ids: tuple

    def to_dict(self):
        return {
            "worker_id": self.worker_id,
            "batch_seq": self.batch_seq,
            "batch_id": self.batch_id,
            "reason": self.reason,
            "request_ids": list(self.request_ids),
        }


def diff_specs(old_spec, new_spec):
    changes = []
    if old_spec.seed != new_spec.seed:
        changes.append(("seed", old_spec.seed, new_spec.seed))
    if old_spec.batch_size != new_spec.batch_size:
        changes.append(("batch_size", old_spec.batch_size, new_spec.batch_size))
    old_params = {w.worker_id: w.param_version for w in old_spec.workers}
    new_params = {w.worker_id: w.param_version for w in new_spec.workers}
    for worker_id in sorted(set(old_params) | set(new_params)):
        if old_params.get(worker_id) != new_params.get(worker_id):
            changes.append(
                ("param_version", f"{worker_id}:{old_params.get(worker_id)}",
                 f"{worker_id}:{new_params.get(worker_id)}"))
    return changes


def compute_invalidation(old_spec, new_spec, old_plans, new_plans):
    """返回 (失效步骤列表, 新计划中按 batch_id 索引的未失效批集合)。"""
    changes = dict((item[0], item) for item in diff_specs(old_spec, new_spec))
    old_by_id = {plan.batch_id: plan for plan in old_plans}
    new_by_id = {plan.batch_id: plan for plan in new_plans}

    old_worker_versions = {w.worker_id: w.param_version for w in old_spec.workers}
    new_worker_versions = {w.worker_id: w.param_version for w in new_spec.workers}

    steps = []
    invalidated_new_ids = set()

    if "seed" in changes:
        for plan in sorted(new_plans, key=lambda p: (p.worker_id, p.batch_seq)):
            steps.append(InvalidationStep(
                plan.worker_id, plan.batch_seq, plan.batch_id,
                "seed_changed", plan.request_ids))
            invalidated_new_ids.add(plan.batch_id)
    if "batch_size" in changes:
        for plan in sorted(new_plans, key=lambda p: (p.worker_id, p.batch_seq)):
            steps.append(InvalidationStep(
                plan.worker_id, plan.batch_seq, plan.batch_id,
                "batch_size_changed_repartition", plan.request_ids))
            invalidated_new_ids.add(plan.batch_id)
    if not {"seed", "batch_size"} & set(changes):
        for plan in sorted(new_plans, key=lambda p: (p.worker_id, p.batch_seq)):
            old_plan = old_by_id.get(plan.batch_id)
            if old_plan is None:
                steps.append(InvalidationStep(
                    plan.worker_id, plan.batch_seq, plan.batch_id,
                    "new_batch_no_prior_receipt", plan.request_ids))
                invalidated_new_ids.add(plan.batch_id)
                continue
            old_v = old_worker_versions.get(plan.worker_id)
            new_v = new_worker_versions.get(plan.worker_id)
            if old_v != new_v:
                steps.append(InvalidationStep(
                    plan.worker_id, plan.batch_seq, plan.batch_id,
                    f"param_version_changed:{old_v}->{new_v}", plan.request_ids))
                invalidated_new_ids.add(plan.batch_id)

    for old_plan in old_plans:
        if old_plan.batch_id not in new_by_id:
            reason = ("batch_size_changed_repartition"
                      if "batch_size" in changes else "batch_removed_by_repartition")
            steps.append(InvalidationStep(
                old_plan.worker_id, old_plan.batch_seq, old_plan.batch_id,
                reason, old_plan.request_ids))

    steps.sort(key=lambda step: (step.worker_id, step.batch_seq, step.reason))
    untouched = {
        plan.batch_id: plan for plan in new_plans
        if plan.batch_id not in invalidated_new_ids
    }
    return steps, untouched


def compare_with_full_recompute(partial_results, full_results):
    """逐项比较子集重算与从空白全量重算的输出摘要。"""
    full_index = {item["batch_id"]: item for item in full_results}
    mismatches = []
    for item in partial_results:
        full_item = full_index.get(item["batch_id"])
        if full_item is None:
            mismatches.append({"batch_id": item["batch_id"], "reason": "missing_in_full"})
            continue
        if item["results"] != full_item["results"]:
            mismatches.append({
                "batch_id": item["batch_id"],
                "reason": "output_digest_mismatch",
                "partial": item["results"],
                "full": full_item["results"],
            })
    return {
        "matched": len(partial_results) - len(mismatches),
        "mismatches": mismatches,
        "all_equal": not mismatches,
    }


def replayed_untouched_digests(untouched_plans, prior_results):
    """未失效批直接复用历史回执摘要（逐字节），不重新推理。"""
    prior_index = {item["batch_id"]: item for item in prior_results}
    reused = []
    for plan in untouched_plans:
        prior = prior_index[plan.batch_id]
        reused.append({
            "batch_id": plan.batch_id,
            "batch_seq": plan.batch_seq,
            "worker_id": plan.worker_id,
            "results": prior["results"],
            "reused": True,
        })
    return reused


def plan_identity_report(old_spec, new_spec):
    old_plans = build_plan(old_spec)
    new_plans = build_plan(new_spec)
    return {
        "old_fingerprint": plan_fingerprint(old_plans),
        "new_fingerprint": plan_fingerprint(new_plans),
        "changes": diff_specs(old_spec, new_spec),
        "old_plans": old_plans,
        "new_plans": new_plans,
    }
