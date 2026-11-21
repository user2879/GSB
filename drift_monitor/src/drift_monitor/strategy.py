"""Data-scale adaptive strategy selector.

A fixed preset table is scored against concrete reference-period scale
indicators (samples, column count, high-cardinality ratio, missing rate and
window count).  Scores are built from explicit scale bands plus continuous
pressure terms, so every configured size tier can land on a distinct
strategy.  The winning strategy changes the real computation path by capping
the window count and toggling approximate quantiles.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import ColumnSpec, MonitorConfig
from .datagen import Dataset


@dataclass(frozen=True)
class Strategy:
    name: str
    max_windows: int
    approximate_quantile: bool
    high_card_metric: str
    description: str


PRESET_STRATEGIES: tuple[Strategy, ...] = (
    Strategy("tiny_exact", 16, False, "freq_diff",
             "small data: exact quantiles, full window count"),
    Strategy("standard_mixed", 8, False, "freq_diff",
             "moderate data: exact quantiles with capped windows"),
    Strategy("large_approx", 4, True, "chi_square",
             "large data: grid approximation and aggressive window cap"),
    Strategy("huge_stream", 2, True, "chi_square",
             "very large / sparse data: minimal windows, approximate path"),
)


@dataclass(frozen=True)
class ScaleIndicators:
    n_samples: int
    n_columns: int
    high_card_ratio: float
    missing_rate: float
    requested_windows: int


@dataclass(frozen=True)
class CandidateScore:
    strategy: str
    score: float
    rationale: str


@dataclass(frozen=True)
class DecisionRecord:
    indicators: ScaleIndicators
    candidates: tuple[CandidateScore, ...]
    selected: str
    reasons: tuple[str, ...]
    effective_windows: int
    approximate_quantile: bool
    metric_overrides: dict[str, str] = field(default_factory=dict)


def compute_indicators(config: MonitorConfig, reference: Dataset,
                       columns: tuple[ColumnSpec, ...]) -> ScaleIndicators:
    n_hc = sum(1 for c in columns if c.kind == "high_cardinality")
    total_missing = sum(col.missing_mask.sum() for col in reference.columns)
    total_cells = reference.n_samples * max(1, len(reference.columns))
    observed_missing_rate = float(total_missing / total_cells) if total_cells else 0.0
    return ScaleIndicators(
        n_samples=reference.n_samples,
        n_columns=len(columns),
        high_card_ratio=round(n_hc / max(1, len(columns)), 6),
        missing_rate=round(observed_missing_rate, 6),
        requested_windows=config.window_count,
    )


def _distance_penalty(ind: ScaleIndicators, sample_lo: int, sample_hi: int,
                      column_limit: int) -> float:
    midpoint = (sample_lo + sample_hi) / 2.0
    sample_pressure = abs(ind.n_samples - midpoint) / max(1.0, midpoint)
    column_pressure = max(0.0, ind.n_columns - column_limit) / max(1, column_limit)
    return round(sample_pressure + column_pressure, 6)


def _score(strategy: Strategy, ind: ScaleIndicators) -> tuple[float, str]:
    # Continuous pressure shared by the approximation-oriented strategies.
    sparsity = ind.missing_rate * 4.0 + ind.high_card_ratio * 3.0
    window_pressure = max(0.0, ind.requested_windows - strategy.max_windows) * 0.5
    if strategy.name == "tiny_exact":
        band = 10.0 if ind.n_samples <= 1_000 and ind.n_columns <= 12 else 0.0
        penalty = _distance_penalty(ind, 1, 1_000, 12) + window_pressure
        rationale = "exact path fits tiny in-memory data"
    elif strategy.name == "standard_mixed":
        band = 10.0 if 1_000 < ind.n_samples <= 5_000 and ind.n_columns <= 32 else 0.0
        penalty = _distance_penalty(ind, 1_000, 5_000, 32) + window_pressure
        rationale = "mixed exact path for moderate tabular data"
    elif strategy.name == "large_approx":
        band = 10.0 if 5_000 < ind.n_samples <= 50_000 else 0.0
        penalty = _distance_penalty(ind, 5_000, 50_000, 64) + window_pressure
        rationale = "grid approximation reduces quantile cost at large scale"
        return round(band + sparsity - penalty, 6), rationale
    else:  # huge_stream
        band = 10.0 if ind.n_samples > 50_000 else 0.0
        penalty = _distance_penalty(ind, 50_000, 200_000, 128) + window_pressure
        rationale = "minimal-window streaming path for huge sparse data"
        return round(band + sparsity * 1.5 - penalty, 6), rationale
    return round(band + sparsity - penalty, 6), rationale


def select_strategy(config: MonitorConfig, reference: Dataset,
                    columns: tuple[ColumnSpec, ...]) -> DecisionRecord:
    indicators = compute_indicators(config, reference, columns)
    scored = [
        CandidateScore(s.name, *_score(s, indicators)) for s in PRESET_STRATEGIES
    ]
    winner_name = max(scored, key=lambda c: c.score).strategy
    winner = next(s for s in PRESET_STRATEGIES if s.name == winner_name)
    effective_windows = min(config.window_count, winner.max_windows)
    overrides = {
        c.name: winner.high_card_metric
        for c in columns
        if c.kind == "high_cardinality" and c.metric != winner.high_card_metric
    }
    reasons = (
        f"samples={indicators.n_samples}, columns={indicators.n_columns}",
        f"high_card_ratio={indicators.high_card_ratio}, "
        f"missing_rate={indicators.missing_rate}",
        f"requested_windows={indicators.requested_windows} capped to {effective_windows}",
        f"approximate_quantile={winner.approximate_quantile}",
    )
    return DecisionRecord(
        indicators=indicators,
        candidates=tuple(scored),
        selected=winner.name,
        reasons=reasons,
        effective_windows=effective_windows,
        approximate_quantile=winner.approximate_quantile,
        metric_overrides=overrides,
    )
