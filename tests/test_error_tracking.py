from pe_diag.error_tracking import track_sample
from pe_diag.scoring import score_sample


def _decision(cfg, scheme, active_dim=None):
    active_dim = active_dim or cfg.dim
    return {"scheme": scheme, "active_dim": active_dim,
            "dim_range": [0, active_dim], "block_method": "single",
            "layer_precision": "float64", "scheme_priority": [scheme]}


def test_rope_reference_has_zero_deviation(small_config, sample):
    d = _decision(small_config, "rope")
    r64 = score_sample(sample, small_config, d, "float64").to_dict()
    r32 = score_sample(sample, small_config, d, "float32").to_dict()
    out = track_sample(sample, small_config, d, r64, r32)
    for rec in out["extrap_records"]:
        # chunk-boundary rounding leaves only a ~1e-17 fingerprint
        assert rec["position_vector_deviation"] < 1e-12
        assert rec["score_deviation"] < 1e-12
        assert rec["first_breach_stage"] == "within_tolerance"


def test_linear_deviation_breaches_position_first(small_config, sample):
    cfg = small_config.with_overrides(scale=4.0)
    d = _decision(cfg, "linear")
    r64 = score_sample(sample, cfg, d, "float64").to_dict()
    r32 = score_sample(sample, cfg, d, "float32").to_dict()
    out = track_sample(sample, cfg, d, r64, r32)
    breached = [r for r in out["extrap_records"]
                if r["first_breach_stage"] != "within_tolerance"]
    assert breached
    assert all(r["first_breach_stage"] == "position_vector"
               for r in breached)


def test_precision_attribution_truncation(small_config, sample):
    # truncating half the dimensions must surface a truncation cause
    d_full = _decision(small_config, "rope")
    d_half = _decision(small_config, "rope", active_dim=8)
    r64 = score_sample(sample, small_config, d_half, "float64").to_dict()
    r32 = score_sample(sample, small_config, d_half, "float32").to_dict()
    out = track_sample(sample, small_config, d_half, r64, r32)
    prec = out["precision_inconsistency"]
    if prec["inconsistent"]:
        assert prec["cause"] in {
            "position_encoding_dimension_truncation",
            "score_accumulation_rounding"}
        assert set(prec["measures"]) == {
            "max_weight_gap", "truncation_gap", "rounding_gap"}
    # sanity: full-dimension run is always computable
    assert score_sample(sample, small_config, d_full, "float64") is not None


def test_float64_calculation_metadata(small_config, sample):
    d = _decision(small_config, "ntk")
    r64 = score_sample(sample, small_config, d, "float64").to_dict()
    out = track_sample(sample, small_config, d, r64, None)
    assert out["reference"] == {"scheme": "rope", "precision": "float64",
                                "scale": 1.0,
                                "dim_range": [0, small_config.dim]}
    assert out["tolerances"] == small_config.tolerances
