from fablock.config import OperatorConfig
from fablock.pipeline import Pipeline


def test_fixed_point_matches_float64_within_tolerance(cfg, dataset):
    report = Pipeline(cfg, dataset).run(1)[0]
    stats = report["compare"]
    assert stats["out_of_tolerance_count"] == 0
    for stage, diff in stats["max_abs_diff_per_stage"].items():
        assert diff < cfg.tolerance_for(stage)
    final = report["compare"]["final_out_max_abs_diff"]
    assert all(d < cfg.tol_output for d in final.values())


def test_frac_bits_change_compare_stats(cfg, dataset):
    r12 = Pipeline(cfg, dataset).run(1)[0]
    r8 = Pipeline(cfg.with_overrides(fixed_point_frac_bits=8),
                  dataset).run(1)[0]
    s12 = r12["compare"]["max_abs_diff_per_stage"]
    s8 = r8["compare"]["max_abs_diff_per_stage"]
    assert s12 != s8  # 超差统计真实变化，而非仅改报告字段
    assert s8["output"] > s12["output"]


def test_quant_level_change_compare_stats(cfg, dataset):
    r16 = Pipeline(cfg, dataset).run(1)[0]
    r8 = Pipeline(cfg.with_overrides(quantization_level_bits=8,
                                     fixed_point_frac_bits=6),
                  dataset).run(1)[0]
    assert (r16["compare"]["max_abs_diff_per_stage"]
            != r8["compare"]["max_abs_diff_per_stage"])
