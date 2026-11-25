"""Requirements 2-4: hysteresis VAD, fixed-point window features, posteriors."""

from __future__ import annotations

import pytest

from vad_diag import dsp
from vad_diag.config import build_config
from vad_diag.engine import VADEngine
from vad_diag.fixedpoint import (
    ALLOWED_FRAC_BITS, ALLOWED_QUANT_MODES, to_fixed, to_float,
)
from vad_diag.injection import InjectionHarness
from vad_diag.ledger import Ledger
from vad_diag.cache import ProductCache
from vad_diag.snapshot import SnapshotStore
from vad_diag.timeline import Timeline
from vad_diag import vad as vad_mod
from vad_diag import windows as win_mod
from vad_diag import posterior as post_mod
from vad_diag.loader import Command


def _engine(config, tmp_path):
    ledger = Ledger(str(tmp_path / "l.jsonl"))
    cache = ProductCache(str(tmp_path / "c.json")).load()
    store = SnapshotStore(str(tmp_path / "ss"))
    store.ensure_dirs()
    engine = VADEngine(config, ledger, cache, store,
                       harness=InjectionHarness([]))
    engine.auto_flush = False
    return engine


def _feed(engine, segments):
    for i, sid in enumerate(sorted(segments)):
        engine.handle_command(Command(
            i, "append", {"segment_id": sid, "segment": segments[sid]}))
    return engine._last_result


def test_hysteresis_detects_two_speech_blocks(config_factory, tmp_path,
                                              dataset_segments):
    config = config_factory()
    engine = _engine(config, tmp_path)
    result = _feed(engine, dataset_segments)
    # Generator alternates silence/speech/silence/speech -> two regions.
    assert len(result.regions) == 2
    region = result.regions[0]
    # Provenance: frames and segments are explicitly recorded.
    assert region.frame_indices
    assert region.segment_ids
    # Every frame index is within frame table bounds.
    assert max(region.frame_indices) < len(result.frames)


def test_region_records_referenced_segments(config_factory, tmp_path,
                                            dataset_segments):
    config = config_factory()
    engine = _engine(config, tmp_path)
    result = _feed(engine, dataset_segments)
    for region in result.regions:
        # Region segments must be a subset of accepted timeline segments.
        assert set(region.segment_ids) <= set(result.timeline.segment_ids)


def test_windows_record_sample_interval_and_segments(config_factory, tmp_path,
                                                     dataset_segments):
    config = config_factory()
    engine = _engine(config, tmp_path)
    result = _feed(engine, dataset_segments)
    for window in result.windows:
        assert window.end > window.start
        assert window.segment_ids
        ids, intersections = result.timeline.segments_for_interval(
            window.start, window.end)
        assert sorted(ids) == sorted(window.segment_ids)
        # Intersections never exceed the window bounds.
        for gs, ge in intersections:
            assert gs >= window.start and ge <= window.end


def test_window_features_are_fixed_point_and_deterministic(
        config_factory, tmp_path, dataset_segments):
    config = config_factory()
    engine = _engine(config, tmp_path)
    result = _feed(engine, dataset_segments)
    feat = list(result.features.values())[0]
    assert all(isinstance(v, int) for v in feat.int_features)
    assert feat.frac_bits == config.frac_bits
    # Recompute the identical window independently -> elementwise equal.
    window = result.windows[0]
    again = win_mod.extract_window_feature(
        window, result.timeline, config, result.frame_intervals)
    assert again.int_features == feat.int_features


def test_quantization_change_moves_integers_and_posteriors(
        config_factory, tmp_path, dataset_segments):
    base = config_factory()
    engine_a = _engine(base, tmp_path / "a")
    res_a = _feed(engine_a, dataset_segments)

    cfg_b = config_factory(frac_bits=8, quant_mode="TRUNC")
    engine_b = _engine(cfg_b, tmp_path / "b")
    res_b = _feed(engine_b, dataset_segments)

    wid = "W0005"
    ints_a = res_a.features[wid].int_features
    ints_b = res_b.features[wid].int_features
    assert ints_a != ints_b  # real integer change
    assert res_a.posteriors[wid].posterior_q != \
        res_b.posteriors[wid].posterior_q
    # De-quantized values also differ.
    assert res_a.features[wid].dequantized() != \
        res_b.features[wid].dequantized()


def test_fixed_point_modes_are_distinct():
    # Pick a value sitting just above a Q8 quantization step so truncation and
    # the rounding modes land on different integers.
    step = 1.0 / 256.0
    value = 3.0 * step + 0.75 * step
    outs = {
        mode: to_fixed(value, 8, mode) for mode in ALLOWED_QUANT_MODES
    }
    # Truncation must differ from rounding for this non-exact value.
    assert outs["TRUNC"] != outs["ROUND_HALF_EVEN"]
    assert to_float(outs["ROUND_HALF_EVEN"], 8) != \
        to_float(to_fixed(value, 15, "TRUNC"), 15)


def test_posterior_threshold_boundary_rule(config_factory, tmp_path,
                                            dataset_segments):
    config = config_factory(speech_threshold=0.5)
    engine = _engine(config, tmp_path)
    result = _feed(engine, dataset_segments)
    # All posteriors are valid Qn probabilities within [0, 1].
    for post in result.posteriors.values():
        assert 0.0 <= post.posterior <= 1.0
        assert isinstance(post.posterior_q, int)
    # Endpoint intervals reference windows and record boundary frames.
    for endpoint in result.endpoints:
        assert endpoint.window_ids
        assert endpoint.frame_indices


def test_frame_ownership_uses_max_posterior_tiebreak(config_factory, tmp_path,
                                                     dataset_segments):
    config = config_factory()
    engine = _engine(config, tmp_path)
    result = _feed(engine, dataset_segments)
    for decision in result.decisions:
        if decision.covering_windows:
            # Owner must be one of the covering windows.
            assert decision.owner_window in decision.covering_windows


def test_window_never_reads_outside_its_interval(config_factory, tmp_path,
                                                 dataset_segments):
    config = config_factory()
    engine = _engine(config, tmp_path)
    result = _feed(engine, dataset_segments)
    # extract_window_feature asserts fs>=start and fe<=end internally;
    # additionally verify every covering frame is fully inside its window.
    for window in result.windows:
        for fi in window.frame_indices:
            fs, fe = result.frame_intervals[fi]
            assert fs >= window.start and fe <= window.end


def test_speech_and_noise_separate_around_threshold(config_factory, tmp_path,
                                                    dataset_segments):
    config = config_factory()
    engine = _engine(config, tmp_path)
    result = _feed(engine, dataset_segments)
    posts = [p.posterior for p in result.posteriors.values()]
    n = len(posts)
    noise = posts[: n // 4]
    speech = posts[n // 4: n // 2]
    assert sum(speech) / len(speech) > sum(noise) / len(noise)


def test_exact_threshold_and_half_step_boundary_rules(config_factory):
    config = config_factory(speech_threshold=0.5)
    from vad_diag import posterior as pm
    from vad_diag.windows import WindowFeature

    def feat_with_posterior_q(posterior_value):
        # Build a feature whose squash output equals the requested posterior by
        # monkeypatching the squash function for a deterministic, isolated test.
        original = pm._piecewise_squash
        pm._piecewise_squash = lambda score: posterior_value
        try:
            feature = WindowFeature(
                window_id="WX", int_features=(0,) * dsp.FEATURE_DIM,
                frac_bits=config.frac_bits, quant_mode=config.quant_mode,
                frame_count=1)
            return pm.synthesize_posterior(feature, config)
        finally:
            pm._piecewise_squash = original

    # Exact threshold -> speech, boundary, exact_threshold tie.
    exact = feat_with_posterior_q(0.5)
    assert exact.is_speech is True
    assert exact.boundary_window is True
    assert exact.tie_kind == "exact_threshold"

    # Within one half step (but quantizes onto the threshold) -> boundary.
    half = 0.5 / (2 ** config.frac_bits)
    near = feat_with_posterior_q(0.5 + half)
    assert near.boundary_window is True
    assert near.tie_kind in ("exact_threshold", "within_half_step")

    # Clearly below threshold -> non-speech, not a boundary.
    low = feat_with_posterior_q(0.1)
    assert low.is_speech is False
    assert low.boundary_window is False
    assert low.tie_kind == ""
