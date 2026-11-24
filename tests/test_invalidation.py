from __future__ import annotations

from dataclasses import replace

from batch_diagnostic.invalidation import propagate
from batch_diagnostic.planning import build_plan


def test_seed_change_invalidates_owner_chain_only(system_spec):
    worker_a = system_spec.workers[0]
    new_workers = (
        replace(worker_a, seed=worker_a.seed + 500, param_version=worker_a.param_version + 1),
        system_spec.workers[1],
    )
    new_spec = replace(system_spec, workers=new_workers)
    report = propagate(system_spec, new_spec, build_plan(new_spec))
    assert report.invalid_workers == ("w-a",)
    assert all(batch_id.startswith("batch-w-a") for batch_id in report.invalid_batch_ids)
    levels = {step.level for step in report.steps}
    assert levels == {"worker_param_version", "batch", "request_attribution"}
    ordered_subjects = [(step.level, step.subject) for step in report.steps]
    batch_positions = [
        index for index, (level, _) in enumerate(ordered_subjects) if level == "batch"
    ]
    request_positions = [
        index for index, (level, _) in enumerate(ordered_subjects) if level == "request_attribution"
    ]
    assert min(batch_positions) < min(request_positions)
    assert any("seed" in reason for reason in report.worker_reasons["w-a"])


def test_batch_size_change_invalidates_worker_batches(system_spec):
    worker_b = system_spec.workers[1]
    new_workers = (
        system_spec.workers[0],
        replace(worker_b, batch_size=worker_b.batch_size + 1),
    )
    new_spec = replace(system_spec, workers=new_workers)
    report = propagate(system_spec, new_spec, build_plan(new_spec))
    assert report.invalid_workers == ("w-b",)
    assert any("batch_size" in reason for reason in report.worker_reasons["w-b"])


def test_worker_addition_and_removal(system_spec):
    added = replace(system_spec.workers[0], worker_id="w-c", port=6199)
    new_spec = replace(system_spec, workers=system_spec.workers + (added,))
    report = propagate(system_spec, new_spec, build_plan(new_spec))
    assert "w-c" in report.invalid_workers

    removed_spec = replace(system_spec, workers=(system_spec.workers[0],))
    report = propagate(system_spec, removed_spec, build_plan(removed_spec))
    assert "w-b" in report.invalid_workers


def test_no_change_means_no_invalidation(system_spec):
    report = propagate(system_spec, system_spec, build_plan(system_spec))
    assert report.invalid_batch_ids == ()
    assert report.steps == ()
