from kvdiag.converge import assert_converged, judge_convergence
from kvdiag.errors import DivergeError
from kvdiag.log import MemoryLogger
from kvdiag.spec import load_spec

import pytest


def _states(spec, lo_by_shard, expiry_by_shard):
    replayed = {sid: {"index_pos_lo": lo_by_shard[sid]} for sid in spec.shard_ids()}
    return replayed, expiry_by_shard


def test_converged_within_tolerance(tree):
    spec = load_spec(tree["spec"])
    los = {sid: 5 for sid in spec.shard_ids()}
    expiry = {sid: 5 for sid in spec.shard_ids()}
    replayed, expiry = _states(spec, los, expiry)
    conclusions = judge_convergence(spec, replayed, expiry)
    assert all(c["status"] == "converged" for c in conclusions.values())
    assert_converged(conclusions)
    assert "tolerance" in conclusions["shard-0"]["basis"]


def test_divergence_reports_positions_and_diff(tree):
    spec = load_spec(tree["spec"])  # tolerance 0
    los = {sid: 5 for sid in spec.shard_ids()}
    expiry = {sid: 5 for sid in spec.shard_ids()}
    expiry["shard-2"] = 2
    logger = MemoryLogger()
    replayed, expiry = _states(spec, los, expiry)
    conclusions = judge_convergence(spec, replayed, expiry, logger=logger)
    assert conclusions["shard-2"]["status"] == "diverged"
    assert conclusions["shard-2"]["diff"] == 3
    assert "E_DIVERGE" in conclusions["shard-2"]["basis"]
    with pytest.raises(DivergeError) as exc:
        assert_converged(conclusions)
    assert exc.value.shard_id == "shard-2"
    assert exc.value.details["index_pos"] == 5
    assert exc.value.details["lease_expiry_pos"] == 2
    assert exc.value.details["diff"] == 3
    codes = [r["code"] for r in logger.records]
    assert "E_DIVERGE" in codes


def test_conclusion_order_independent(tree):
    spec = load_spec(tree["spec"])
    los = {sid: i for i, sid in enumerate(spec.shard_ids())}
    expiry = {sid: i for i, sid in enumerate(spec.shard_ids())}
    replayed, expiry = _states(spec, los, expiry)
    forward = judge_convergence(spec, replayed, expiry)
    reversed_conclusions = judge_convergence(
        spec, replayed, expiry, shard_ids=list(reversed(spec.shard_ids())))
    assert forward == reversed_conclusions
