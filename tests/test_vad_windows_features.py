from voicecmd.audio import load_fragments
from voicecmd.classifier import build_projection, classify_window
from voicecmd.features import (
    extract_window_feature,
    fragment_feature_key,
    mel_filterbank,
)
from voicecmd.vad import detect_fragment
from voicecmd.windowing import build_windows


def _build(cfg, workspace):
    fragments = load_fragments(
        workspace / cfg.paths.manifest_file,
        workspace / cfg.paths.data_dir,
        set(cfg.vocab.values()),
    )
    intervals = [
        interval
        for fragment in fragments
        for interval in detect_fragment(fragment.samples, fragment.seq, cfg.vad)
    ]
    windows = build_windows(intervals, fragments, cfg.window)
    return fragments, intervals, windows


def test_vad_intervals_deterministic(workspace, cfg):
    fragments, first, _ = _build(cfg, workspace)
    second = [
        interval
        for fragment in fragments
        for interval in detect_fragment(fragment.samples, fragment.seq, cfg.vad)
    ]
    assert [(i.fragment_seq, i.start_sample, i.end_sample, i.start_frame, i.end_frame)
            for i in first] == [
        (i.fragment_seq, i.start_sample, i.end_sample, i.start_frame, i.end_frame)
        for i in second
    ]
    assert first, "样本必须检出语音区间"


def test_windows_record_span_and_cross_boundary_refs(workspace, cfg):
    fragments, _, windows = _build(cfg, workspace)
    assert windows
    cross = [w for w in windows if len(w.fragment_seqs) > 1]
    assert cross, "样本必须包含跨片段边界分析窗"
    offsets = {seq: seq * cfg.sample.samples_per_fragment for seq in range(len(fragments))}
    for window in windows:
        assert window.end_sample > window.start_sample
        assert window.end_sample - window.start_sample <= cfg.window.window_length
        expected = tuple(
            seq
            for seq in range(len(fragments))
            if (seq + 1) * cfg.sample.samples_per_fragment > window.start_sample
            and seq * cfg.sample.samples_per_fragment < window.end_sample
        )
        assert window.fragment_seqs == expected
    indices = [w.window_index for w in windows]
    assert indices == sorted(indices)


def test_window_set_byte_identical_across_runs(workspace, cfg):
    _, _, first = _build(cfg, workspace)
    _, _, second = _build(cfg, workspace)
    a = [(w.window_index, w.start_sample, w.end_sample, w.fragment_seqs) for w in first]
    b = [(w.window_index, w.start_sample, w.end_sample, w.fragment_seqs) for w in second]
    assert a == b


def test_fixed_point_features_and_accumulation(workspace, cfg):
    fragments, _, windows = _build(cfg, workspace)
    bank = mel_filterbank(cfg.mel, cfg.sample.sample_rate)
    keys = {f.seq: fragment_feature_key(f, cfg) for f in fragments}
    audio = __import__("numpy").concatenate([f.samples for f in sorted(fragments, key=lambda x: x.seq)])
    feature = extract_window_feature(windows[0], audio, cfg, bank, keys)
    assert len(feature.fixed_point) == cfg.feature_dim
    assert all(isinstance(v, int) for v in feature.fixed_point)
    assert feature.frame_count >= 1
    # 帧序号升序、特征维升序由实现固定；这里校验缓存键随内容变化
    feature2 = extract_window_feature(windows[1], audio, cfg, bank, keys)
    assert feature.cache_key != feature2.cache_key


def test_quant_level_changes_integers_and_scores(workspace, cfg):
    fragments, _, windows = _build(cfg, workspace)
    bank = mel_filterbank(cfg.mel, cfg.sample.sample_rate)
    keys = {f.seq: fragment_feature_key(f, cfg) for f in fragments}
    audio = __import__("numpy").concatenate([f.samples for f in sorted(fragments, key=lambda x: x.seq)])
    projection = build_projection(cfg)
    results = {}
    for level in cfg.quant.allowed_levels:
        cfg.quant.quant_level = level
        feature = extract_window_feature(windows[0], audio, cfg, bank, keys)
        posterior = classify_window(feature, cfg, projection)
        results[level] = (feature.fixed_point, posterior.scores, feature.cache_key)
    fixed_points = {value[0] for value in results.values()}
    scores = {value[1] for value in results.values()}
    keys_set = {value[2] for value in results.values()}
    assert len(fixed_points) == len(cfg.quant.allowed_levels)
    assert len(scores) == len(cfg.quant.allowed_levels)
    assert len(keys_set) == len(cfg.quant.allowed_levels)


def test_classifier_tie_breaks_by_code_and_marks_boundary(workspace, cfg):
    fragments, _, windows = _build(cfg, workspace)
    bank = mel_filterbank(cfg.mel, cfg.sample.sample_rate)
    keys = {f.seq: fragment_feature_key(f, cfg) for f in fragments}
    audio = __import__("numpy").concatenate([f.samples for f in sorted(fragments, key=lambda x: x.seq)])
    projection = build_projection(cfg)
    feature = extract_window_feature(windows[0], audio, cfg, bank, keys)
    posterior = classify_window(feature, cfg, projection)
    assert len(posterior.scores) == len(cfg.commands)
    assert posterior.predicted_code in cfg.vocab
    assert isinstance(posterior.boundary, bool)
    # 构造完全相等后验：全零输入 -> 全零得分 -> 平局取最小编码
    import numpy as np
    from voicecmd.features import WindowFeature
    zero = WindowFeature(
        window_index=0, fragment_seqs=(0,), span=(0, 1), frame_count=1,
        frame_features=(), fixed_point=tuple([0] * cfg.feature_dim),
        cache_key="zero", fragment_key="zero",
    )
    tied = classify_window(zero, cfg, projection)
    assert tied.predicted_code == min(cfg.vocab)
    assert tied.boundary is True
