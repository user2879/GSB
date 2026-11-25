from __future__ import annotations

import numpy as np

from fixedtex.fixed import quantize_fixed, round_ratio_array
from fixedtex.noise import compute_layer_chunk, layer_fingerprint

from .helpers import SAMPLE_CONFIG, SAMPLE_TEXTURE
from fixedtex.config import load_render_config
from fixedtex.texture import load_texture


def test_quantization_steps_change_integers_and_residuals() -> None:
    raw = np.arange(1, 17, dtype=np.int64)
    q4, r4, d4 = quantize_fixed(raw, 4, "ties_to_even")
    q8, r8, d8 = quantize_fixed(raw, 8, "ties_to_even")
    assert not np.array_equal(q4, q8)
    assert not np.array_equal(r4, r8)
    assert q4.tolist() == [0, 0, 4, 4, 4, 8, 8, 8, 8, 8, 12, 12, 12, 16, 16, 16]
    assert q8.tolist() == [0, 0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 16, 16, 16, 16, 16]
    assert d4.dtype == np.int8 and d8.dtype == np.int8


def test_tie_rules_are_deterministic() -> None:
    values = np.asarray([2, 6, 10], dtype=np.int64)
    even, decisions_even = round_ratio_array(values, 4, "ties_to_even")
    lower, decisions_lower = round_ratio_array(values, 4, "ties_to_lower")
    upper, decisions_upper = round_ratio_array(values, 4, "ties_to_upper")
    assert even.tolist() == [0, 2, 2]
    assert lower.tolist() == [0, 1, 2]
    assert upper.tolist() == [1, 2, 3]
    assert decisions_even.tolist() != decisions_lower.tolist()
    assert decisions_upper.tolist() != decisions_lower.tolist()


def test_layer_noise_is_deterministic_and_chunk_invariant() -> None:
    config, _ = load_render_config(SAMPLE_CONFIG)
    texture, _ = load_texture(config.texture_path, expected_resolution=config.resolution)
    layer = config.layers[1]
    fingerprint = layer_fingerprint(layer, config, texture)
    full = compute_layer_chunk(layer, config, texture, 0, 64 * 64, fingerprint=fingerprint)
    pieces = np.concatenate(
        [
            compute_layer_chunk(layer, config, texture, 0, 700, fingerprint=fingerprint),
            compute_layer_chunk(layer, config, texture, 700, 2500, fingerprint=fingerprint),
            compute_layer_chunk(layer, config, texture, 2500, 4096, fingerprint=fingerprint),
        ]
    )
    again = compute_layer_chunk(layer, config, texture, 0, 64 * 64, fingerprint=fingerprint)
    assert np.array_equal(full, pieces)
    assert np.array_equal(full, again)
    assert full.min() >= 0 and full.max() <= 1 << config.fixed.fraction_bits
