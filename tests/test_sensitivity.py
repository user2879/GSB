import numpy as np

from pe_diag.sensitivity import run_sensitivity


def _decision(small_config):
    return {"scheme": "rope", "active_dim": small_config.dim,
            "dim_range": [0, small_config.dim], "block_method": "single",
            "layer_precision": "float64", "scheme_priority": ["rope"]}


def test_ranking_order_and_tie_break(small_config, sample):
    cfg = small_config
    out = run_sensitivity([sample], cfg, _decision(cfg), ["rope"],
                          sample_limit=None)
    rows = out["base_ranking"]
    # deviation desc, value asc on ties -> deterministic
    keys = [(-r["mean_extrap_score_deviation"], r["value"]) for r in rows]
    assert keys == sorted(keys)
    assert [r["rank"] for r in rows] == list(range(1, len(rows) + 1))


def test_invalid_grid_points_record_boundary(small_config, sample):
    cfg = small_config
    out = run_sensitivity([sample], cfg, _decision(cfg), ["rope"])
    scale_rows = out["scale_ranking"]
    invalid = [r for r in scale_rows if r["value"] == 0.0]
    assert invalid and invalid[0]["error_code"] == "E_SCALE_INVALID"
    base_rows = out["base_ranking"]
    bad_base = [r for r in base_rows if r["value"] == 0.0]
    assert bad_base and bad_base[0]["error_code"] == "E_BASE_ZERO"


def test_baseline_zero_deviation(small_config, sample):
    out = run_sensitivity([sample], small_config,
                          _decision(small_config), ["rope"])
    base = [r for r in out["base_ranking"] if r["is_baseline"]]
    assert base and np.isclose(base[0]["mean_extrap_score_deviation"], 0.0)
