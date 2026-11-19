"""要求 5：音视频双域同映射、一致性约束与参考切换重算。"""
from __future__ import annotations

from syncbench.config import PARAM_REFERENCE, default_config
from syncbench.engine import Engine
from syncbench.errors import E_SYNC_MISMATCH


def test_dual_domain_same_map_and_within_tolerance(engine):
    for ch in range(4):
        dual = engine._duals[ch]
        a_versions = {s.map_version for s in dual.audio_segments}
        v_versions = {s.map_version for s in dual.video_segments}
        assert a_versions == v_versions == {dual.timeline.version}
        assert dual.max_ref_diff_sec <= engine.cfg.av_sync_tol_sec
        assert dual.mismatch is None


def test_reference_switch_recomputes_maps(tmp_path):
    eng = Engine(str(tmp_path), default_config())
    eng.initial_build()
    old_refs = {ch: eng._duals[ch].timeline.reference_channel for ch in range(4)}
    assert set(old_refs.values()) == {0}
    eng.apply_change(PARAM_REFERENCE, None, 2)
    new_refs = {ch: eng._duals[ch].timeline.reference_channel for ch in range(4)}
    assert set(new_refs.values()) == {2}


def test_mismatch_locator_identifies_frame_and_sample(tmp_path):
    # 构造一个故意超差的容差，验证失配定位含帧号与样本位置
    eng = Engine(str(tmp_path), default_config())
    eng.initial_build()
    eng.cfg.av_sync_tol_sec = -1.0  # 强制任何非零差都失配
    from syncbench.timeline import check_dual_domain
    for ch in range(4):
        dual = eng._duals[ch]
        _, mismatch, errors = check_dual_domain(
            dual.timeline, dual.audio_segments, dual.video_segments, eng.cfg)
        assert errors and errors[0].code == E_SYNC_MISMATCH
        assert "video_frame" in mismatch and "audio_sample" in mismatch
