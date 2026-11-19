from __future__ import annotations

from hls_workbench.buffer import build_delay_table, simulate_buffer
from hls_workbench.config import (
    BudgetConfig,
    ClientBufferConfig,
    DEFAULT_PROFILES,
    MediaConfig,
)
from hls_workbench.decision import DecisionEngine


def test_buffer_720p_underruns_and_suggests_downgrade() -> None:
    media = MediaConfig()
    table = build_delay_table(DEFAULT_PROFILES, media, media.seed)
    ready = {sequence: sequence * 20 for sequence in range(media.slice_count)}
    durations = [2000] * media.slice_count

    high = DEFAULT_PROFILES[1]
    high_model = simulate_buffer(
        profile=high,
        lower_profile="360p",
        ready_times_ms=ready,
        delays_ms=table["delays_ms"]["720p"],
        segment_durations_ms=durations,
        media_duration_ms=30000,
        config=ClientBufferConfig(capacity_ms=8000, threshold_ms=2600, step_ms=100),
    )
    low = DEFAULT_PROFILES[0]
    low_model = simulate_buffer(
        profile=low,
        lower_profile=None,
        ready_times_ms=ready,
        delays_ms=table["delays_ms"]["360p"],
        segment_durations_ms=durations,
        media_duration_ms=30000,
        config=ClientBufferConfig(capacity_ms=8000, threshold_ms=2600, step_ms=100),
    )

    assert high_model.underrun_count >= 1
    assert high_model.underrun_events[0]["code"] == "E_BUFFER_UNDERRUN"
    assert high_model.switch_recommendations
    assert high_model.switch_recommendations[0]["suggested_profile"] == "360p"
    assert high_model.switch_recommendations[0]["buffer_ms"] < 2600
    assert low_model.underrun_count == 0
    assert low_model.safe


def test_decision_downgrades_and_reuses_unaffected_artifacts() -> None:
    media = MediaConfig()
    table = build_delay_table(DEFAULT_PROFILES, media, media.seed)
    ready = {sequence: sequence * 20 for sequence in range(media.slice_count)}
    engine = DecisionEngine(
        profiles=DEFAULT_PROFILES,
        ready_times_ms={"360p": ready, "720p": ready},
        delay_table=table,
        segment_durations_ms=[2000] * media.slice_count,
        media_duration_ms=30000,
        buffer_config=ClientBufferConfig(capacity_ms=8000, threshold_ms=2600, step_ms=100),
        budget=BudgetConfig(max_workers=4, memory_budget_mb=300, cpu_budget_ms=200),
        bandwidth_safety_ratio=0.85,
    )
    primary = engine.decide(bandwidth_kbps=2600, threshold_ms=2600)
    assert primary.selected_profile == "360p"
    assert primary.safe

    variant = engine.decide(bandwidth_kbps=900, threshold_ms=2600)
    assert variant.selected_profile == "360p"
    assert "360p" in variant.model_cache_hits
    assert variant.model_cache_misses == ()
    assert variant.recomputed_decisions == (variant.decision_key,)
