from voicecalib import load_inputs, run_pipeline, build_report
from voicecalib.errors import ErrorCode


def test_four_injection_types_isolated(full_samples_dir, tmp_path):
    inputs = load_inputs(full_samples_dir, injection_file="injections.json")
    # 12 轮样本下注入清单全部在界内。
    assert ErrorCode.E_ROUND_OUT_OF_RANGE not in inputs.errors.codes()
    result = run_pipeline(inputs, tmp_path / "out", resume=False)
    report = build_report(result, inputs.config, full_samples_dir, "injection")
    codes = report["failure_summary"]
    assert codes.get(ErrorCode.E_NONFINITE_SAMPLE)
    assert codes.get(ErrorCode.E_INJECTION_DIM_MISMATCH)
    assert codes.get(ErrorCode.E_ZERO_GAIN)
    assert codes.get(ErrorCode.E_ROUND_OUT_OF_RANGE)
    hit_ids = {item["injection_id"] for item in report["injections_hit"]}
    assert {"INJ-NONFINITE-01", "INJ-DIM-01", "INJ-ZEROGAIN-01",
            "INJ-REPLACE-01"} <= hit_ids
    # 单轮失败不污染其余轮次：12 轮中 4 轮故障，8 轮完成。
    assert report["rounds_completed"] == 8
    failed_rounds = {3, 5, 7, 9}
    assert failed_rounds.isdisjoint(set(result.calibrations))


def test_clean_run_has_no_injection_hits(samples_dir, tmp_path):
    inputs = load_inputs(samples_dir)
    result = run_pipeline(inputs, tmp_path / "out", resume=False)
    assert result.injections_hit == []
    # 干净输入下不存在任何注入/故障类业务码（预算待办除外）。
    fault_codes = {ErrorCode.E_NONFINITE_SAMPLE,
                   ErrorCode.E_INJECTION_DIM_MISMATCH,
                   ErrorCode.E_ZERO_GAIN, ErrorCode.E_ROUND_OUT_OF_RANGE,
                   ErrorCode.E_AUDIO_FORMAT, ErrorCode.E_AUDIO_MISSING}
    assert fault_codes.isdisjoint(set(result.errors.codes()))
