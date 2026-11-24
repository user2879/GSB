from __future__ import annotations

from dataclasses import replace

import numpy as np

from batch_diagnostic.inference import build_parameters, forward_batch
from batch_diagnostic.planning import build_plan
from batch_diagnostic.spec import WorkerSpec


def test_plan_deterministic_and_remainder(system_spec):
    first = build_plan(system_spec)
    second = build_plan(system_spec)
    assert [b.to_dict() for b in first.batches] == [b.to_dict() for b in second.batches]
    assert any(batch.remainder for batch in first.batches)
    covered = [rid for batch in first.batches for rid in batch.request_ids]
    assert sorted(covered) == sorted(r.request_id for r in system_spec.requests)


def test_batch_ids_embed_worker_index_and_seqs(system_spec):
    plan = build_plan(system_spec)
    for batch in plan.batches:
        assert batch.worker_id in batch.batch_id
        assert f"{batch.batch_index:04d}" in batch.batch_id


def test_forward_digest_changes_with_seed(system_spec):
    plan = build_plan(system_spec)
    batch = plan.batches[0]
    worker = system_spec.worker_by_id()[batch.worker_id]
    result_a = forward_batch(system_spec, worker, batch)
    changed_worker = replace(worker, seed=worker.seed + 1, param_version=worker.param_version + 1)
    result_b = forward_batch(system_spec, changed_worker, batch)
    assert result_a.output_digest != result_b.output_digest
    assert result_a.intermediate_dim == worker.hidden_dim


def test_forward_order_independent(system_spec):
    plan = build_plan(system_spec)
    batch = plan.batches[0]
    worker = system_spec.worker_by_id()[batch.worker_id]
    baseline = forward_batch(system_spec, worker, batch).output_digest
    shuffled = replace(batch, request_ids=tuple(reversed(batch.request_ids)))
    reordered = forward_batch(system_spec, worker, shuffled).output_digest
    assert baseline == reordered


def test_parameter_digest_deterministic(system_spec):
    worker: WorkerSpec = system_spec.workers[0]
    params_a = build_parameters(worker)
    params_b = build_parameters(worker)
    assert params_a.param_digest == params_b.param_digest
    assert np.array_equal(params_a.weight, params_b.weight)
