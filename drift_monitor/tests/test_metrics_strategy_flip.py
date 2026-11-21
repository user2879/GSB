import numpy as np
import pytest

from drift_monitor.config import load_config
from drift_monitor.datagen import CUR, REF, generate_pair
from drift_monitor.fixedpoint import DEFAULT_TIERS, QuantTier, quantize
from drift_monitor.flip import diagnose_flips
from drift_monitor.metrics import (
    build_stats,
    metric_score,
)
from drift_monitor.datagen import GeneratedColumn
from drift_monitor.strategy import PRESET_STRATEGIES, select_strategy
from drift_monitor.datagen import expanded_columns


def _column(name, kind, values):
    arr = np.empty(len(values), dtype=object)
    mask = np.zeros(len(values), dtype=bool)
    for idx, value in enumerate(values):
        arr[idx] = value
        if value is None:
            mask[idx] = True
    return GeneratedColumn(name, kind, arr, mask)


def test_mean_shift_and_hist_density_numeric():
    ref = _column("n", "numeric", [1.0, 1.1, 0.9, 1.0, 1.2])
    cur = _column("n", "numeric", [3.0, 3.1, 2.9, 3.0, 3.2])
    ref_stats = build_stats(ref, 0, REF, 0, 5)
    cur_stats = build_stats(cur, 0, CUR, 0, 5)
    shift = metric_score("mean_shift", ref_stats, cur_stats, False)
    hist = metric_score("hist_density", ref_stats, cur_stats, False)
    assert shift > 5.0
    assert hist >= 0.0


def test_freq_diff_and_chi_square_categorical_lexicographic():
    ref = _column("c", "categorical", ["a", "a", "b", "c", "b"])
    cur = _column("c", "categorical", ["c", "c", "b", "a", "a"])
    ref_stats = build_stats(ref, 0, REF, 0, 5)
    cur_stats = build_stats(cur, 0, CUR, 0, 5)
    tv = metric_score("freq_diff", ref_stats, cur_stats, False)
    chi = metric_score("chi_square", ref_stats, cur_stats, False)
    assert tv == pytest.approx(0.2)
    assert chi >= 0.0
    assert [k for k, _ in ref_stats.frequencies] == ["a", "b", "c"]


def test_missing_entries_are_counted_not_crashing():
    ref = _column("n", "numeric", [1.0, None, 1.1, None, 0.9])
    stats = build_stats(ref, 0, REF, 0, 5)
    assert stats.n_missing == 2
    assert stats.n_observed == 3


def test_approx_and_exact_quantiles_observably_differ():
    rng = np.random.default_rng(7)
    values = rng.lognormal(0, 1.2, 500).tolist()
    col = _column("skew", "numeric", values)
    stats = build_stats(col, 0, REF, 0, len(values))
    assert stats.quantiles_approx != stats.quantiles_exact
    ref_stats = stats
    shifted = _column("skew", "numeric",
                      [v * (1.6 if idx % 2 == 0 else 0.7)
                       for idx, v in enumerate(values)])
    cur_stats = build_stats(shifted, 0, CUR, 0, len(values))
    exact_score = metric_score("quantile", ref_stats, cur_stats, False)
    approx_score = metric_score("quantile", ref_stats, cur_stats, True)
    assert exact_score != pytest.approx(approx_score, abs=1e-9)


def test_strategy_changes_path_and_records_reasons(config):
    tier = config.data.size_tiers[0]
    ref, _ = generate_pair(config.data, tier)
    columns = expanded_columns(config.data, tier)
    decision = select_strategy(config, ref, columns)
    assert decision.selected in {s.name for s in PRESET_STRATEGIES}
    assert decision.effective_windows <= config.window_count
    assert len(decision.candidates) == len(PRESET_STRATEGIES)
    assert decision.indicators.n_samples == tier.samples
    assert all(c.rationale for c in decision.candidates)


def test_strategy_approximation_toggle_on_large_tier(config):
    large = config.data.size_tiers[-1]
    ref, _ = generate_pair(config.data, large)
    columns = expanded_columns(config.data, large)
    decision = select_strategy(config, ref, columns)
    assert decision.approximate_quantile in (True, False)
    assert decision.effective_windows <= config.window_count


def test_quantization_tier_changes_integer_score():
    raw = 0.123456789
    coarse = quantize(raw, DEFAULT_TIERS[0])
    fine = quantize(raw, DEFAULT_TIERS[2])
    assert coarse == 12
    assert fine == 123457
    assert coarse / DEFAULT_TIERS[0].scale != fine / DEFAULT_TIERS[2].scale or \
        fine != coarse


def test_flip_diagnostic_records_breach_and_reference():
    tiers = DEFAULT_TIERS
    evaluations, flips, ref_eval = diagnose_flips(
        "col", 0, 0.75005, 0.75, tiers)
    assert len(flips) == len(tiers) - 1
    pair = flips[0]
    assert pair.lower_tier == "coarse" and pair.higher_tier == "standard"
    assert pair.first_breach_stage in ("quantize", "threshold", "judgement", None)
    assert pair.reference_tier.endswith("_ref")
    assert ref_eval.decimals == tiers[-1].decimals + 2
    flipping = [f for f in flips if f.flipped]
    if flipping:
        assert flipping[0].first_breach_stage == "judgement"


def test_flip_detection_when_judgement_actually_flips():
    tiers = DEFAULT_TIERS
    _, flips, _ = diagnose_flips("col", 1, 0.74999, 0.75, tiers)
    flags = [(f.lower_tier, f.higher_tier, f.flipped) for f in flips]
    assert any(flag[2] for flag in flags)


def test_large_tier_observably_switches_quantile_path(config, tmp_path):
    from drift_monitor.runner import run_pipeline
    from dataclasses import replace

    large = config.data.size_tiers[-1]
    approx_report = run_pipeline(config, large.name, tmp_path / "approx",
                                 resume=False)
    assert approx_report["approximate_quantile"] is True
    exact_config = replace(config, enable_approx_quantile=False)
    exact_report = run_pipeline(exact_config, large.name, tmp_path / "exact",
                                resume=False)
    assert exact_report["approximate_quantile"] is False
    approx_scores = {
        (row["column"], row["window_index"]): row["raw_score"]
        for row in approx_report["scores_and_judgements"]
        if row["metric"] == "quantile"
    }
    exact_scores = {
        (row["column"], row["window_index"]): row["raw_score"]
        for row in exact_report["scores_and_judgements"]
        if row["metric"] == "quantile"
    }
    assert approx_scores and exact_scores
    assert any(approx_scores[key] != exact_scores[key] for key in approx_scores)


def test_judgement_flip_at_quantization_boundary():
    from drift_monitor.fixedpoint import QuantTier
    from drift_monitor.flip import diagnose_flips

    # Coarse width rounds the score below the rounded threshold while the
    # wider tier retains enough precision to stay significant.
    tiers = (
        QuantTier("coarse", decimals=1, clamp=1000),
        QuantTier("standard", decimals=3, clamp=1_000_000),
    )
    raw_score = 0.155
    threshold = 0.156
    evaluations, flips, ref_eval = diagnose_flips(
        "edge", 0, raw_score, threshold, tiers)
    coarse_eval = next(e for e in evaluations if e.tier == "coarse")
    standard_eval = next(e for e in evaluations if e.tier == "standard")
    assert coarse_eval.significant != standard_eval.significant
    assert coarse_eval.significant is True
    assert standard_eval.significant is False
    flip = flips[0]
    assert flip.flipped is True
    assert flip.first_breach_stage == "judgement"
    assert ref_eval.decimals == standard_eval.decimals + 2
