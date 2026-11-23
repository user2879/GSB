from __future__ import annotations

from spmm_diag.audit import replay_audit
from spmm_diag.engine import IncrementalEngine


def test_history_audit_replays_hot_table_and_budget_invariants(sample_bundle):
    csr, b, c, config = sample_bundle
    engine = IncrementalEngine(csr, b, c, config)
    engine.run()
    engine.replace_row(4, [0, 4], [1.0, -1.0])
    engine.run(budget=0)
    engine.run_until_complete()
    engine.set_alpha_beta(0.9, 0.1)
    engine.run(budget=engine._block_cost(0))
    engine.run_until_complete()
    audit = replay_audit(engine.audit_manifest(), config.hot_capacity)
    assert audit["hot_table"] == engine.hot.to_dict()
    assert audit["budget_sequence_valid"]
    assert audit["budget_arithmetic_valid"]
    assert audit["budget_block_order_stable"]

