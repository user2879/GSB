import numpy as np
import pytest

from hdr_sdr_pipeline.config import (
    GAMMA_MAX, GAMMA_MIN, ProcessingState, TONEMAP_INPUT_MAX,
    TRANSFER_ENDPOINT_TOL, TONEMAP_MONO_TOL,
)
from hdr_sdr_pipeline.errors import InjectBoundsError, TonemapDomainError
from hdr_sdr_pipeline.pipeline import process_frame
from hdr_sdr_pipeline.quantize import quantize_array, quantization_step
from hdr_sdr_pipeline.tonemap import aces_scalar, reinhard_scalar, tonemap_array
from hdr_sdr_pipeline.transfer import encode_array, gamma_scalar, srgb_scalar


def test_reinhard_monotonic_and_endpoints():
    state = ProcessingState(tonemap_white=1.0)
    xs = np.linspace(0.0, TONEMAP_INPUT_MAX, 2048)
    ys = np.array([reinhard_scalar(x, 1.0) for x in xs])
    assert np.all(np.diff(ys) >= -TONEMAP_MONO_TOL)
    assert abs(reinhard_scalar(0.0, 1.0)) <= TRANSFER_ENDPOINT_TOL
    assert 0.0 <= reinhard_scalar(1000.0, 1.0) <= 1.0  # overflow saturates


def test_aces_monotonic_endpoints_and_frozen_constants():
    state = ProcessingState(tonemap_operator="aces")
    xs = np.linspace(0.0, TONEMAP_INPUT_MAX, 2048)
    ys = np.array([aces_scalar(x, state.tonemap_a, state.tonemap_b,
                               state.tonemap_c, state.tonemap_d,
                               state.tonemap_e) for x in xs])
    assert np.all(np.diff(ys) >= -TONEMAP_MONO_TOL)
    assert abs(aces_scalar(0.0, state.tonemap_a, state.tonemap_b,
                           state.tonemap_c, state.tonemap_d,
                           state.tonemap_e)) <= TRANSFER_ENDPOINT_TOL
    assert 0.0 <= aces_scalar(1.0e6, state.tonemap_a, state.tonemap_b,
                              state.tonemap_c, state.tonemap_d,
                              state.tonemap_e) <= 1.0


def test_tonemap_non_finite_is_domain_error():
    state = ProcessingState()
    bad = np.array([[[np.nan, 0.0, 0.0]]], dtype=np.float32)
    with pytest.raises(TonemapDomainError):
        tonemap_array(bad, state, frame_index=0)


def test_negative_and_overflow_are_deterministic():
    state = ProcessingState()
    x = np.array([[[-3.0, 0.5, 80.0]]], dtype=np.float32)
    y1 = tonemap_array(x, state, 0)
    y2 = tonemap_array(x, state, 0)
    assert np.array_equal(y1, y2)
    assert y1[0, 0, 0] == 0.0            # negative -> zero
    assert y1[0, 0, 2] <= 1.0            # overflow -> saturated in [0,1]


def test_srgb_transfer_endpoints_and_monotonicity():
    xs = np.linspace(0.0, 1.0, 1024)
    ys = np.array([srgb_scalar(x) for x in xs])
    assert abs(srgb_scalar(0.0)) <= TRANSFER_ENDPOINT_TOL
    assert abs(srgb_scalar(1.0) - 1.0) <= TRANSFER_ENDPOINT_TOL
    assert np.all(np.diff(ys) >= -1e-9)


def test_gamma_bounds_enforced():
    assert gamma_scalar(0.5, 2.2) >= 0.0
    with pytest.raises(InjectBoundsError):
        gamma_scalar(0.5, GAMMA_MAX + 1.0)
    state = ProcessingState(transfer="gamma", transfer_gamma=GAMMA_MIN / 2)
    with pytest.raises(InjectBoundsError):
        encode_array(np.zeros((1, 1, 3), np.float32), state)


def test_quantize_step_and_determinism():
    assert quantization_step() == 1.0 / 255.0
    state = ProcessingState(dither=False)
    x = np.linspace(0.0, 1.0, 300, dtype=np.float32).reshape(10, 10, 3)
    a = quantize_array(x, state)
    b = quantize_array(x, state)
    assert a.dtype == np.uint8 and np.array_equal(a, b)
    assert a.min() == 0 and a.max() == 255


def test_pipeline_stage_hashes_deterministic(tiny_workdir):
    state = ProcessingState()
    u1, h1, _ = process_frame(0, tiny_workdir + "/frames", state,
                              width=96, height=54)
    u2, h2, _ = process_frame(0, tiny_workdir + "/frames", state,
                              width=96, height=54)
    assert h1 == h2 and np.array_equal(u1, u2)
    assert list(h1) == ["load", "normalize", "colorspace", "exposure",
                        "tonemap", "transfer", "quantize"]


def test_bt2020_differs_from_bt709_and_is_fixed(tiny_workdir):
    s709 = ProcessingState(colorspace="bt709")
    s2020 = ProcessingState(colorspace="bt2020")
    u709, h709, _ = process_frame(1, tiny_workdir + "/frames", s709,
                                   width=96, height=54)
    u2020, h2020, _ = process_frame(1, tiny_workdir + "/frames", s2020,
                                    width=96, height=54)
    assert h709 != h2020
    u2020_b, h2020_b, _ = process_frame(1, tiny_workdir + "/frames",
                                          s2020, width=96, height=54)
    assert h2020 == h2020_b and np.array_equal(u2020, u2020_b)


def test_historical_state_recomputes_identically(tiny_workdir):
    s1 = ProcessingState(exposure_ev=0.5)
    s2 = ProcessingState(exposure_ev=1.5, tonemap_operator="aces")
    _u, h1_first, _ = process_frame(2, tiny_workdir + "/frames", s1,
                                      width=96, height=54)
    process_frame(2, tiny_workdir + "/frames", s2, width=96, height=54)
    _u, h1_again, _ = process_frame(2, tiny_workdir + "/frames", s1,
                                    width=96, height=54)
    assert h1_first == h1_again
