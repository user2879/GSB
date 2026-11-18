"""Requirement 3: fixed-point quantization, integer projection and width sensitivity."""
from __future__ import annotations

import os

import numpy as np

from voiceprint.aggregation import aggregate
from voiceprint.audio import read_wav
from voiceprint.embedding import (
    build_projection_matrix,
    fixed_embedding,
    float_reference_embedding,
)
from voiceprint.features import extract_frames
from voiceprint.fixedpoint import (
    FixedPoint,
    integer_cosine,
    integer_l2_normalize,
    quantize_symmetric,
)


def _agg(dataset_dir, config, name):
    wave = read_wav(os.path.join(dataset_dir, name))
    return aggregate(extract_frames(wave, config), config)


def test_quantization_width_changes_integers(make_config):
    cfg = make_config()
    values = np.linspace(-1.0, 1.0, 17)
    q12 = quantize_symmetric(values, cfg.quant_levels, FixedPoint(12))
    q13 = quantize_symmetric(values, cfg.quant_levels, FixedPoint(13))
    # same codes, different fixed-point integer scale
    assert q12["codes"].tolist() == q13["codes"].tolist()
    assert not np.array_equal(q12["q_ints"], q13["q_ints"])
    assert np.all(np.abs(q13["q_ints"]) <= (1 << 13) * (cfg.quant_levels - 1))


def test_quant_levels_change_codes(make_config):
    values = np.linspace(-0.8, 0.8, 20)
    coarse = quantize_symmetric(values, 8, FixedPoint(12))
    fine = quantize_symmetric(values, 64, FixedPoint(12))
    assert not np.array_equal(coarse["codes"], fine["codes"])
    assert coarse["codes"].max() <= 7 and fine["codes"].max() <= 63


def test_projection_matrix_is_deterministic(make_config):
    cfg = make_config()
    m1 = build_projection_matrix(cfg, 50)
    m2 = build_projection_matrix(cfg, 50)
    np.testing.assert_array_equal(m1, m2)
    assert m1.shape == (cfg.embedding_dim, 50)


def test_width_changes_embedding_and_float_differs(dataset_dir, make_config):
    cfg = make_config()
    agg = _agg(dataset_dir, cfg, "spk_01_utt_01.wav")
    matrix = build_projection_matrix(cfg, agg.stats_vector.shape[0])
    emb12 = fixed_embedding(agg, cfg, 12, matrix)
    emb13 = fixed_embedding(agg, cfg, 13, matrix)
    assert not np.array_equal(emb12.embedding_int, emb13.embedding_int)
    assert emb12.raw_accumulator.dtype == np.int64
    # integer L2 normalization yields unit-length embedding on the Q scale
    fp = FixedPoint(12)
    norm = int(np.sqrt(np.dot(emb12.embedding_int, emb12.embedding_int)))
    assert abs(norm - fp.scale) <= 2
    float_ref = float_reference_embedding(agg, cfg, matrix)
    assert not np.allclose(emb12.embedding_float, float_ref["embedding_float"])


def test_integer_cosine_bounds_and_determinism():
    rng = np.random.default_rng(42)
    a = rng.integers(-2000, 2000, size=16, dtype=np.int64)
    r1 = integer_cosine(a, a, 12)
    r2 = integer_cosine(a, a, 12)
    assert r1["similarity_q30"] == r2["similarity_q30"]
    assert abs(r1["similarity"] - 1.0) < 1e-8
    b = -a
    r3 = integer_cosine(a, b, 12)
    assert abs(r3["similarity"] + 1.0) < 1e-8


def test_integer_l2_normalize_zero_safe():
    out = integer_l2_normalize(np.zeros(8, dtype=np.int64), FixedPoint(12))
    assert out["norm"] == 0
    assert np.all(out["embedding_int"] == 0)
