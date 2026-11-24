from __future__ import annotations

from faultisolation.planner import choose_path, estimate_work, tie_threshold, PATHS


def test_work_estimates_are_ints(small_bundle):
    work = estimate_work(small_bundle)
    assert set(work) == set(PATHS)
    assert all(isinstance(v, int) for v in work.values())


def test_choose_path_returns_reason_and_params(small_bundle):
    decision = choose_path(small_bundle)
    assert decision["selected"] in PATHS
    assert "threshold" in decision and "scale_parameters" in decision
    assert isinstance(decision["reason"], str) and decision["reason"]


def test_tie_threshold_positive(small_bundle):
    assert tie_threshold(small_bundle) >= 1
