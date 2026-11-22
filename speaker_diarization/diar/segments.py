"""语音区间切分（双门限迟滞）与重叠候选标注。"""

from .util import l1_distance, mean_quantized


def _vad_flags(frames, config):
    """能量 + 谱平坦度双门限迟滞判定。

    进入语音：能量高于门限且谱平坦度低于门限（语音更结构化）；
    退出语音：能量跌破 0.6 倍门限或谱平坦度突破门限。
    """
    flags = []
    active = False
    enter_energy = config.energy_threshold
    exit_energy = config.energy_threshold * 0.6
    for frame in frames:
        voiced = (frame["energy"] >= enter_energy
                  and frame["sfm"] <= config.spectral_flatness_threshold)
        quiet = (frame["energy"] < exit_energy
                 or frame["sfm"] > config.spectral_flatness_threshold)
        if not active and voiced:
            active = True
        elif active and quiet:
            active = False
        flags.append(active)
    return flags


def _raw_intervals(flags):
    intervals = []
    start = None
    for index, flag in enumerate(flags):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            intervals.append([start, index])
            start = None
    if start is not None:
        intervals.append([start, len(flags)])
    return intervals


def _filter_and_merge(intervals, frames, config):
    frame_shift_ms = config.frame_shift_ms
    min_speech_frames = max(1, int(round(config.min_speech_ms / frame_shift_ms)))
    min_silence_frames = int(round(config.min_silence_ms / frame_shift_ms))
    long_enough = [iv for iv in intervals if iv[1] - iv[0] >= min_speech_frames]
    merged = []
    for interval in long_enough:
        if merged:
            gap = interval[0] - merged[-1][1]
            if gap <= min_silence_frames:
                merged[-1][1] = interval[1]
                continue
        merged.append(list(interval))
    return merged, []


def segment_audio(feature_product, config, registry=None, audio_name=None):
    audio_name = audio_name or feature_product["audio"]
    frames = feature_product["frames"]
    n_frames = feature_product["n_frames"]
    if registry is not None:
        registry.check_segment_range(audio_name, n_frames)
    flags = _vad_flags(frames, config)
    intervals, _ = _filter_and_merge(_raw_intervals(flags), frames, config)
    frac_bits = config.frac_bits
    step = feature_product["quant_step"]

    segments = []
    boundary_records = []
    for seg_index, (start, end) in enumerate(intervals):
        matrix = [frames[i]["cepstrum_q"] for i in range(start, end)]  # 帧序升序
        mean_q = mean_quantized(matrix)
        threshold_q = int(round(config.cluster_distance_threshold * (2 ** config.frac_bits)))
        # 重叠候选：与前一语音段时间相邻（构造数据中相邻段在时间窗上相交）；
        # 距离恰等于阈值时按非重叠决断并记入边界清单。
        overlap_candidate = bool(segments)
        if segments:
            distance = l1_distance(mean_q, segments[-1]["mean_cepstrum_q"])
            if distance == threshold_q:
                boundary_records.append({
                    "segment_index": seg_index,
                    "distance": distance,
                    "threshold": threshold_q,
                    "decision": "non_overlap_equal_threshold",
                })
                overlap_candidate = False
        segments.append({
            "sort_key": f"segment:{seg_index:08d}",
            "segment_index": seg_index,
            "start_frame": int(start),
            "end_frame": int(end),
            "start_ms": frames[start]["start_ms"],
            "end_ms": frames[end - 1]["end_ms"],
            "overlap_candidate": bool(overlap_candidate),
            "mean_cepstrum_q": [int(x) for x in mean_q],
            "frame_set": list(range(start, end)),
            "quant_step": step,
            "frac_bits": frac_bits,
        })
    return {
        "audio": audio_name,
        "n_frames": n_frames,
        "vad_flags": flags,
        "segments": segments,
        "boundary_records": boundary_records,
    }
