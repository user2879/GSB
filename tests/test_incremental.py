from __future__ import annotations

import numpy as np
import pytest

from spmm_diag.config import MODE_FIXED, ORDER_B_ABS_DESC, SpMMConfig
from spmm_diag.csr import DistributionSpec, generate_csr
from spmm_diag.engine import IncrementalEngine, compare_incremental_to_full, full_rebuild


def test_replace_row_only_recomputes_referencing_block(sample_bundle):
    csr, b, c, config = sample_bundle
    engine = IncrementalEngine(csr, b, c, config)
    initial = engine.run()
    assert initial.complete
    before = {block_id: block.output_fingerprint for block_id, block in engine.output_blocks.items()}
    targeted_row = 7
    expected_block = targeted_row // config.row_block_size
    engine.replace_row(targeted_row, [0, 1, 8], [1.25, -0.5, 2.0])
    report = engine.run()
    assert report.recomputed == (expected_block,)
    assert expected_block not in report.reused
    assert set(report.reused) == set(engine.current_block_ids) - {expected_block}
    for block_id in report.reused:
        assert engine.output_blocks[block_id].output_fingerprint == before[block_id]
        assert engine.compute_invocations[block_id] == 1
    assert engine.hot.to_dict()["entries"][0]["block_id"] == expected_block
    assert engine.hot.to_dict()["entries"][0]["invalidations"] == 1


def test_delete_row_invalidates_shifted_suffix(sample_bundle):
    csr, b, c, config = sample_bundle
    engine = IncrementalEngine(csr, b, c, config)
    engine.run()
    deleted_row = 5
    first_affected = deleted_row // config.row_block_size
    engine.delete_row(deleted_row)
    report = engine.run()
    assert report.recomputed == tuple(range(first_affected, len(engine.block_ranges)))
    assert all(block_id < first_affected for block_id in report.reused)


def test_pending_budget_is_stable_and_drains(sample_bundle):
    csr, b, c, config = sample_bundle
    engine = IncrementalEngine(csr, b, c, config)
    first = engine.run(budget=0)
    pending_codes = [item.code for item in first.failures if item.code == "E_RECOMPUTE_PENDING"]
    assert pending_codes
    assert not first.recomputed
    assert first.pending == tuple(range(len(engine.block_ranges)))

    costs = [engine._block_cost(block_id) for block_id in range(len(engine.block_ranges))]
    budget = costs[0] + costs[1]
    second = engine.run(budget=budget)
    assert second.recomputed == (0, 1)
    assert 0 not in second.pending and 1 not in second.pending
    assert second.budget_deductions[0]["block_id"] == 0
    assert second.budget_deductions[0]["budget_before"] == budget
    reports = [second, *engine.run_until_complete()]
    assert reports[-1].complete


def test_any_edit_sequence_matches_full_rebuild(sample_bundle):
    csr, b, c, config = sample_bundle
    engine = IncrementalEngine(csr, b, c, config)
    engine.run()

    engine.replace_row(4, [0, 4, 8], [1.2, -0.4, 0.8])
    engine.run()
    engine.replace_row(4, [1, 4], [-0.7, 1.1])
    engine.run()
    engine.delete_row(2)
    engine.run()
    engine.append_row([0, 3, 7], [0.3, -1.4, 0.9], c_init_row=[0.1] * b.shape[1])
    engine.run()
    engine.set_block_size(2, 3)
    engine.run()
    engine.set_alpha_beta(1.15, 0.2)
    engine.run()
    engine.set_precision(MODE_FIXED, 4, 10)
    engine.run(budget=0)
    assert engine.run(budget=0).pending
    engine.set_order(ORDER_B_ABS_DESC)
    engine.run_until_complete()

    reference = full_rebuild(engine.csr, engine.b_matrix, engine.c_matrix, engine.config)
    comparison = compare_incremental_to_full(engine, reference, compare_history_dependent=False)
    assert comparison["all_exact"]
    assert comparison["max_abs_difference"] == 0.0
    assert engine.materialize_c().shape == reference.c_matrix.shape


def test_concurrent_and_single_thread_reports_match(sample_bundle):
    csr, b, c, config = sample_bundle

    def exercise(cfg: SpMMConfig):
        engine = IncrementalEngine(csr, b, c, cfg)
        reports = [engine.run(budget=0), engine.run()]
        engine.replace_row(7, [1, 5], [0.7, -0.4])
        reports.append(engine.run(budget=0))
        reports.append(engine.run())
        engine.set_config(cfg.with_changes(alpha=0.9, beta=0.1))
        reports.append(engine.run())
        return engine, reports

    single, single_reports = exercise(config.with_changes(max_workers=1))
    concurrent, concurrent_reports = exercise(config.with_changes(max_workers=4))
    assert np.array_equal(single.materialize_c(), concurrent.materialize_c())
    assert single.hot.to_dict() == concurrent.hot.to_dict()
    assert single.budget_ledger == concurrent.budget_ledger
    for left, right in zip(single_reports, concurrent_reports):
        left_value = left.to_dict()
        right_value = right.to_dict()
        left_value.pop("max_workers_used")
        right_value.pop("max_workers_used")
        assert left_value == right_value


@pytest.mark.parametrize("seed", range(5))
def test_random_edit_sequence_always_matches_full_rebuild(seed):
    kind = ("uniform", "powerlaw", "extreme", "powerlaw", "uniform")[seed]
    csr = generate_csr(15, 8, DistributionSpec(kind, 0, 7, power=2.2), seed=100 + seed)
    rng = np.random.default_rng(200 + seed)
    b = rng.normal(size=(8, 4))
    c = rng.normal(size=(15, 4))
    config = SpMMConfig(
        alpha=0.8,
        beta=0.25,
        row_block_size=3,
        nnz_chunk_size=2,
        order="col_asc",
        precision_mode="fp32",
        max_workers=3,
        recompute_budget=10_000,
        hot_capacity=5,
    )
    engine = IncrementalEngine(csr, b, c, config)
    engine.run_until_complete()
    orders = ["col_asc", "col_desc", "b_abs_desc"]
    for step in range(24):
        operation = str(rng.choice(["replace", "delete", "append", "config"]))
        if operation == "replace" and engine.csr.rows:
            row = int(rng.integers(0, engine.csr.rows))
            count = int(rng.integers(0, 6))
            columns = sorted(int(value) for value in rng.choice(8, size=count, replace=False))
            values = rng.uniform(-2.0, 2.0, size=count)
            if count:
                values[0] = np.copysign(max(abs(values[0]), 0.05), values[0])
            engine.replace_row(row, columns, values)
        elif operation == "delete" and engine.csr.rows > 3:
            engine.delete_row(int(rng.integers(0, engine.csr.rows)))
        elif operation == "append":
            count = int(rng.integers(0, 6))
            columns = sorted(int(value) for value in rng.choice(8, size=count, replace=False))
            values = rng.uniform(-2.0, 2.0, size=count)
            engine.append_row(columns, values, c_init_row=rng.normal(size=4))
        else:
            choice = step % 5
            if choice == 0:
                engine.set_alpha_beta(float(rng.uniform(0.5, 1.5)), float(rng.uniform(0.0, 0.4)))
            elif choice == 1:
                engine.set_block_size(int(rng.integers(2, 5)), int(rng.integers(1, 4)))
            elif choice == 2:
                engine.set_precision(MODE_FIXED, int(rng.integers(4, 8)), int(rng.integers(12, 25)))
            elif choice == 3:
                engine.set_precision("fp32", 8, 24)
            else:
                engine.set_order(str(rng.choice(orders)), int(rng.integers(1, 5)))
        engine.run()
        engine.run_until_complete()
    reference = full_rebuild(engine.csr, engine.b_matrix, engine.c_matrix, engine.config)
    comparison = compare_incremental_to_full(engine, reference)
    assert comparison["all_exact"], comparison
