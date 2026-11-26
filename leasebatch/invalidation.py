"""Stable invalidation propagation along the dependency chain.

Dependency direction::

    worker parameter version -> batch -> request ownership

A change to a worker parameter version, the global seed or the batch size
invalidates batches; invalidated batches in turn invalidate the requests they
own.  Output is sorted by worker id then batch index, each step carrying an
explicit propagation basis, and unaffected batches keep byte-identical
digests.
"""

from .plan import build_plan


def baseline_versions(spec):
    """Every worker starts at parameter version 0 unless overridden."""
    return {worker.worker_id: 0 for worker in spec.workers}


def _requests_of(batch):
    return sorted(batch.request_ids)


def diff_plans(old_plan, new_plan):
    """Structural diff keyed by stable batch id."""
    old_by_id = {b.batch_id: b for b in old_plan.batches}
    new_by_id = {b.batch_id: b for b in new_plan.batches}
    removed = sorted(set(old_by_id) - set(new_by_id))
    added = sorted(set(new_by_id) - set(old_by_id))
    changed = sorted(
        batch_id
        for batch_id in set(old_by_id) & set(new_by_id)
        if old_by_id[batch_id].content_digest
        != new_by_id[batch_id].content_digest
    )
    return removed, added, changed


def propagate(spec, old_plan, new_plan, old_versions, new_versions,
              *, seed_changed=False, batch_size_changed=False):
    """Return a stable invalidation report and the recompute subset.

    Direct causes:
      * parameter version bump for a worker -> all of that worker's batches
      * seed change                          -> every batch
      * batch size change / plan structural diff -> affected batch ids
    Indirect cause:
      * invalidated batch -> requests owned by that batch
    """
    direct = {}

    def mark(batch_id, basis):
        direct.setdefault(batch_id, basis)

    removed, added, changed = diff_plans(old_plan, new_plan)
    old_by_id = {b.batch_id: b for b in old_plan.batches}

    if seed_changed:
        for batch in old_plan.batches:
            mark(batch.batch_id, "seed_change:fixed_point_inputs_and_params")
    if batch_size_changed:
        for batch_id in removed + changed:
            mark(batch_id, "batch_size_change:partition_rebuilt")
    else:
        for batch_id in changed:
            mark(batch_id, "membership_change:request_reassignment")

    bumped_workers = sorted(
        worker_id
        for worker_id in sorted(set(old_versions) | set(new_versions))
        if old_versions.get(worker_id, 0) != new_versions.get(worker_id, 0)
    )
    for worker_id in bumped_workers:
        for batch in old_plan.batches:
            if batch.worker_id == worker_id:
                mark(
                    batch.batch_id,
                    f"param_version_change:{worker_id} "
                    f"v{old_versions.get(worker_id, 0)}->"
                    f"v{new_versions.get(worker_id, 0)}",
                )

    invalid_batch_ids = sorted(direct)
    invalid_requests = {}
    steps = []
    for batch_id in invalid_batch_ids:
        batch = old_by_id[batch_id]
        steps.append(
            {
                "level": "batch",
                "batch_id": batch_id,
                "worker_id": batch.worker_id,
                "basis": direct[batch_id],
            }
        )
        for request_id in _requests_of(batch):
            invalid_requests[request_id] = batch_id
            steps.append(
                {
                    "level": "request",
                    "request_id": request_id,
                    "batch_id": batch_id,
                    "basis": "request_ownership:member_of_invalidated_batch",
                }
            )

    new_by_id = {b.batch_id: b for b in new_plan.batches}
    untouched = [
        batch.batch_id
        for batch in old_plan.batches
        if batch.batch_id not in direct
        and batch.batch_id in new_by_id
        and batch.content_digest
        == new_by_id[batch.batch_id].content_digest
    ]

    recompute_plan = build_plan(
        spec, new_plan.batch_size, new_plan.seed
    )
    recompute_batches = [
        batch.batch_id
        for batch in recompute_plan.batches
        if batch.batch_id in set(added) | set(invalid_batch_ids)
    ]

    return {
        "invalid_batch_ids": invalid_batch_ids,
        "invalid_request_ids": sorted(invalid_requests),
        "request_to_batch": dict(sorted(invalid_requests.items())),
        "steps": steps,
        "untouched_batch_ids": untouched,
        "added_batch_ids": added,
        "removed_batch_ids": removed,
        "recompute_batch_ids": sorted(set(recompute_batches)),
        "bumped_workers": bumped_workers,
    }
