"""要求 2：联合漂移估计、亚采样延迟、残差/置信区间与 E_DRIFT_EXCEED。"""
from __future__ import annotations

import numpy as np

from syncbench.errors import E_DRIFT_EXCEED


def test_linear_drift_ppm_accuracy(engine):
    cfg = engine.cfg
    for ch in (0, 1):
        est = engine._estimations[ch]
        truth = cfg.truth(ch)
        # 分段 ppm 与注入恒定 ppm 偏差在 5ppm 内
        for seg in est.segments:
            assert abs(seg.ppm - truth.ppm) < 5.0
            # 置信区间有限且宽度为正
            assert seg.ppm_ci[0] < seg.ppm < seg.ppm_ci[1]
            assert np.isfinite(seg.rms_residual)


def test_thermal_drift_segment_variation(engine):
    # 温漂通道各段 ppm 应随热曲线明显变化，而非单一常数
    est2 = engine._estimations[2]
    ppms = [s.ppm for s in est2.segments]
    assert max(ppms) - min(ppms) > 50.0


def test_subsample_delay_estimate(engine):
    cfg = engine.cfg
    # 线性通道亚采样延迟误差 < 0.5 样本
    for ch in (0, 1):
        est = engine._estimations[ch]
        err = abs(est.delay_samples - cfg.truth(ch).delay_samples)
        assert err < 0.5


def test_residuals_and_bias_confidence(engine):
    for ch in range(4):
        est = engine._estimations[ch]
        assert est.audio_bias_ci[0] <= est.audio_bias_samples <= est.audio_bias_ci[1]
        assert all(np.isfinite(r) for s in est.segments for r in s.residuals)


def test_drift_exceed_business_code(engine):
    est1 = engine._estimations[1]
    codes = {e.code for e in est1.errors}
    assert E_DRIFT_EXCEED in codes
    detail = next(e for e in est1.errors if e.code == E_DRIFT_EXCEED)
    assert detail.channel == 1
    assert detail.segment in (0, 1, 2)


def test_audio_video_joint_estimate(engine):
    # 联合 ppm 对线性通道应贴近真值（音视频联合而非单点）
    est0 = engine._estimations[0]
    assert abs(est0.fused_ppm - engine.cfg.truth(0).ppm) < 5.0
