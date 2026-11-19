"""要求 3：空洞检测、可复现填补与 E_GAP_TOO_LONG。"""
from __future__ import annotations

from syncbench.config import GAP_LINEAR, GAP_MARK, default_config
from syncbench.engine import Engine
from syncbench.errors import E_GAP_TOO_LONG
from syncbench.gaps import fill_channel


def test_gap_detection_matches_truth(engine):
    for ch in range(4):
        truth_audio = [g for g in engine.cfg.truth(ch).gaps if g.domain == "audio"]
        detected = engine._audio_gap_runs[ch]
        assert len(detected) == len(truth_audio)
        for g, t in zip(detected, truth_audio):
            assert g.start == t.start and g.end == t.end


def test_three_policies_distinct_and_reproducible(tmp_path):
    eng = Engine(str(tmp_path), default_config())
    eng.prepare_media()
    ch = 0
    eng._compute_drift(ch)
    rate = eng._estimations[ch].segments[-1].slope
    common = (eng._audio[ch], eng._video[ch],
              eng._audio_gap_runs[ch], eng._video_gap_runs[ch])
    r1 = fill_channel(ch, *common, GAP_LINEAR, "v1", rate, eng.cfg)
    r2 = fill_channel(ch, *common, GAP_LINEAR, "v1", rate, eng.cfg)
    assert r1.audio_hash == r2.audio_hash  # 同参数逐字节可复现
    r_mark = fill_channel(ch, *common, GAP_MARK, "v1", rate, eng.cfg)
    assert r_mark.audio_hash != r1.audio_hash  # 不同策略结果不同
    assert all(not rec.filled for rec in r_mark.records)


def test_gap_records_drift_version_and_interval(engine):
    g = engine._gaps[0]
    assert g.drift_version  # 绑定漂移估计版本
    for rec in g.records:
        assert rec.interval_ref is not None and len(rec.interval_ref) == 2


def test_gap_too_long_business_code(engine):
    # ch1 含 300 样本音频空洞，超过 240 阈值
    g1 = engine._gaps[1]
    codes = {e.code for e in g1.errors}
    assert E_GAP_TOO_LONG in codes
    detail = next(e for e in g1.errors if e.code == E_GAP_TOO_LONG)
    assert detail.detail["length"] == 300
    assert detail.detail["first_start"] == 115050
