import numpy as np
import pytest

from pe_diag.encoding import PositionEncoder
from pe_diag.errors import DiagError, E_BASE_ZERO, E_SCALE_INVALID


def _enc(small_config, scheme, **kw):
    kw.setdefault("precision", "float64")
    return PositionEncoder(
        scheme=scheme, base=small_config.base,
        train_len=small_config.train_len, target_len=small_config.target_len,
        dim=small_config.dim, scale=small_config.scale,
        chunk_size=small_config.chunk_size, **kw)


def test_three_schemes_share_structure(small_config):
    for scheme in ("rope", "linear", "ntk"):
        res = _enc(small_config, scheme).encode_positions([0, 1, 5])
        assert res.position_vectors.shape == (3, small_config.dim)
        assert res.cos.shape == (3, small_config.dim // 2)
        assert res.angles.shape == (3, small_config.dim // 2)
        assert res.metadata["dim_range"] == [0, small_config.dim]
        assert np.all(np.isfinite(res.position_vectors))


def test_determinism(small_config):
    a = _enc(small_config, "rope").encode_positions([3, 7])
    b = _enc(small_config, "rope").encode_positions([3, 7])
    np.testing.assert_array_equal(a.position_vectors, b.position_vectors)


def test_scheme_change_changes_values(small_config):
    rope = _enc(small_config, "rope").encode_positions([10]).position_vectors
    linear = PositionEncoder(
        "linear", 10000.0, 8, 32, 16, 2.0, 8, "float64"
    ).encode_positions([10]).position_vectors
    ntk = PositionEncoder(
        "ntk", 10000.0, 8, 32, 16, 8.0, 8, "float64"
    ).encode_positions([10]).position_vectors
    assert not np.allclose(rope, linear)
    assert not np.allclose(rope, ntk)
    ntk_enc = PositionEncoder("ntk", 10000.0, 8, 32, 16, 8.0, 8, "float64")
    assert ntk_enc.effective_base != ntk_enc.base


def test_chunk_size_changes_values(small_config):
    pos = [1000]
    e8 = PositionEncoder("rope", 10000.0, 8, 1_000_000, 16, 1.0, 8,
                         "float64").encode_positions(pos).position_vectors
    e16 = PositionEncoder("rope", 10000.0, 8, 1_000_000, 16, 1.0, 16,
                          "float64").encode_positions(pos).position_vectors
    # numeric (tiny, high-precision) difference, not just a relabel
    assert not np.array_equal(e8, e16)
    np.testing.assert_allclose(e8, e16, atol=1e-12)
    f8 = PositionEncoder("rope", 10000.0, 8, 1_000_000, 16, 1.0, 8,
                         "float32").encode_positions(pos).position_vectors
    f16 = PositionEncoder("rope", 10000.0, 8, 1_000_000, 16, 1.0, 16,
                          "float32").encode_positions(pos).position_vectors
    assert np.abs(f8 - f16).max() > np.abs(e8 - e16).max()


def test_active_dim_truncation(small_config):
    full = _enc(small_config, "rope", active_dim=16).encode_positions([4])
    trunc = _enc(small_config, "rope", active_dim=8).encode_positions([4])
    assert full.metadata["dim_range"] == [0, 16]
    assert trunc.metadata["dim_range"] == [0, 8]
    np.testing.assert_allclose(full.position_vectors[:, :8],
                               trunc.position_vectors, atol=1e-12)
    assert full.position_vectors.shape[1] != trunc.position_vectors.shape[1]


def test_boundary_errors(small_config):
    with pytest.raises(DiagError) as exc:
        PositionEncoder("rope", 0.0, 8, 32, 16, 1.0, 8, "float64")
    assert exc.value.code == E_BASE_ZERO
    with pytest.raises(DiagError) as exc:
        PositionEncoder("rope", 10000.0, 8, 32, 16, -1.0, 8, "float64")
    assert exc.value.code == E_SCALE_INVALID


def test_rotate_presidents_dim_and_pairs(small_config, sample):
    enc = _enc(small_config, "rope")
    out = enc.rotate(np.asarray(sample.query_features), sample.query_pos)
    assert out.shape == (small_config.dim,)
    # rotating at position 0 is the identity (cos0=1, sin0=0)
    zero = enc.rotate(np.asarray(sample.query_features), 0)
    np.testing.assert_allclose(zero, np.asarray(sample.query_features),
                               atol=1e-12)
