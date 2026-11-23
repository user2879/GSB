import shutil

import numpy as np

from voicecalib import load_inputs, run_pipeline, build_report
from voicecalib.audio import write_wav


def _prime(sample_dir, out):
    inputs = load_inputs(sample_dir)
    result = run_pipeline(inputs, out, resume=False)
    return inputs, result


def _reopen(prime_out, second_out):
    shutil.copytree(prime_out / "snapshots", second_out / "snapshots")
    shutil.copy(prime_out / "ledger.json", second_out / "ledger.json")


def test_replace_audio_only_downstream_invalidated(full_samples_dir, tmp_path):
    sample_copy = tmp_path / "samples"
    shutil.copytree(full_samples_dir, sample_copy)
    first_out = tmp_path / "first"
    _prime(sample_copy, first_out)

    rng = np.random.default_rng(42)
    write_wav(sample_copy / "round_006.wav",
              np.sin(2 * np.pi * 170 * np.arange(16000) / 16000) * 0.4
              + rng.normal(0, 0.004, 16000))

    second_out = tmp_path / "second"
    second_out.mkdir()
    _reopen(first_out, second_out)
    inputs = load_inputs(sample_copy)
    result = run_pipeline(inputs, second_out, resume=True)
    report = build_report(result, inputs.config, sample_copy, "incr")

    # R0..R5 上游 6 轮全部命中，R6 及下游重算。
    assert report["incremental"]["reuse_hits"] >= 6
    recompute = set(report["incremental"]["recompute_rounds"])
    assert 6 in recompute
    assert all(r not in recompute for r in range(6))
    reasons = {d["round"]: d["reason"] for d in report["incremental"]["decisions"]
               if d["stage"] == "evidence"}
    assert reasons[6] == "recompute:audio_content_changed"


def test_gain_change_invalidates_calibration_only(full_samples_dir, tmp_path):
    """账本级精确判定：增益变化只改校准参数指纹，证据指纹保持不变。

    配置整体哈希变化会使阶段快照按 E_SNAPSHOT_STALE 冷启动（见快照测试），
    因此这里直接验证账本的分阶段失效判定逻辑。
    """
    from voicecalib.ledger import Ledger
    from voicecalib.hashing import fingerprint

    sample_copy = tmp_path / "samples"
    shutil.copytree(full_samples_dir, sample_copy)
    first_out = tmp_path / "first"
    inputs, result = _prime(sample_copy, first_out)
    config = inputs.config

    current_audio = {rnd: (inputs.clips[rnd].path.name,
                           inputs.clips[rnd].content_hash)
                     for rnd in inputs.valid_rounds}
    plan = result.ledger.plan_round_status(
        current_audio,
        param_fingerprint=config.evidence_param_fingerprint,
        frac_bits=config.frac_bits,
        quant_gear=config.quant_gear,
        calib_param_fingerprint=config.param_fingerprint,
    )
    assert all(v["evidence"].startswith("reuse") for v in plan.values())

    # 仅校准参数指纹变化（模拟增益修改）。
    changed_calib_fp = fingerprint(config.param_fingerprint + "gain=0.55")
    plan2 = result.ledger.plan_round_status(
        current_audio,
        param_fingerprint=config.evidence_param_fingerprint,
        frac_bits=config.frac_bits,
        quant_gear=config.quant_gear,
        calib_param_fingerprint=changed_calib_fp,
    )
    assert all(v["evidence"].startswith("reuse") for v in plan2.values())
    assert all("param_or_fixedpoint_changed" in v["calibration"]
               for v in plan2.values())
    assert all("calibration_invalidated" in v["attribution"]
               for v in plan2.values())


def test_delete_audio_cascades(full_samples_dir, tmp_path):
    sample_copy = tmp_path / "samples"
    shutil.copytree(full_samples_dir, sample_copy)
    first_out = tmp_path / "first"
    _prime(sample_copy, first_out)
    (sample_copy / "round_004.wav").unlink()

    second_out = tmp_path / "second"
    second_out.mkdir()
    _reopen(first_out, second_out)
    inputs = load_inputs(sample_copy)
    result = run_pipeline(inputs, second_out, resume=True)
    report = build_report(result, inputs.config, sample_copy, "delete")
    assert (4 not in result.valid_rounds) and report["rounds_total"] == 11
    assert report["incremental"]["reuse_hits"] >= 4
    assert all(r < 4 for r in report["incremental"]["recompute_rounds"]) is False
