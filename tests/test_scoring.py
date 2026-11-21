import numpy as np
import pytest

from pe_diag.errors import DiagError, E_KEY_POS_OOR, E_NONFINITE_INPUT
from pe_diag.scoring import label_regions, score_sample, softmax_single, softmax_stream


def _decision(small_config, scheme="rope", active_dim=None, block="single"):
    return {
        "scheme": scheme,
        "active_dim": active_dim or small_config.dim,
        "dim_range": [0, active_dim or small_config.dim],
        "block_method": block,
        "layer_precision": "float64",
        "scheme_priority": [scheme],
    }


def test_boundary_equality_goes_inside(small_config):
    q = 5
    keys = (q + small_config.train_len, q + small_config.train_len + 1, q)
    rels, regions = label_regions(q, keys, small_config.train_len,
                                  small_config.target_len)
    assert regions[0] == "inside"      # equality -> inside by fixed rule
    assert regions[1] == "extrap"
    assert regions[2] == "inside"
    assert rels == (small_config.train_len,
                    small_config.train_len + 1, 0)


def test_key_out_of_range(small_config):
    with pytest.raises(DiagError) as exc:
        label_regions(0, (small_config.target_len + 1,),
                      small_config.train_len, small_config.target_len)
    assert exc.value.code == E_KEY_POS_OOR


def test_weights_normalize_and_deterministic(small_config, sample):
    d = _decision(small_config)
    r1 = score_sample(sample, small_config, d, "float64")
    r2 = score_sample(sample, small_config, d, "float64")
    np.testing.assert_allclose(np.sum(r1.weights), 1.0, atol=1e-12)
    np.testing.assert_array_equal(r1.weights, r2.weights)
    assert r1.regions.count("extrap") >= 1


def test_stream_vs_single_paths(small_config, sample):
    single = score_sample(sample, small_config,
                          _decision(small_config, block="single"), "float64")
    stream = score_sample(sample, small_config,
                          _decision(small_config, block="stream"), "float64")
    np.testing.assert_allclose(single.weights, stream.weights, atol=1e-12)
    # float32 accumulation order must be a genuine numeric difference
    s32 = score_sample(sample, small_config,
                       _decision(small_config, block="stream"), "float32")
    b32 = score_sample(sample, small_config,
                       _decision(small_config, block="single"), "float32")
    assert not np.array_equal(s32.weights, b32.weights)


def test_nonfinite_input_rejected(small_config, sample):
    overrides = {"query": [{"component": 0, "value": float("nan")}], "key": []}
    with pytest.raises(DiagError) as exc:
        score_sample(sample, small_config, _decision(small_config), "float64",
                     feature_overrides=overrides)
    assert exc.value.code == E_NONFINITE_INPUT


def test_softmax_equivalence_on_known_vector():
    x = np.array([1.0, 2.0, 3.0, 0.5], dtype=np.float64)
    np.testing.assert_allclose(softmax_single(x), softmax_stream(x), atol=1e-14)
