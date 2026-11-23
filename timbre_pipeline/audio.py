"""音频读取、RIFF 校验、帧能量双门限迟滞切分。"""

import os
import wave

import numpy as np

from .errors import PipelineError, E_AUDIO_FORMAT
from .utils import canonical_json, sha256_text


def read_wav_strict(path):
    """严格读取 WAV。

    校验 RIFF/WAVE 容器头、fmt 参数（16kHz/单声道/16bit）以及 data 块
    长度与采样数一致；任何不合规抛出带 ``E_AUDIO_FORMAT`` 的业务异常。
    """
    audio_id = os.path.splitext(os.path.basename(path))[0]
    if not os.path.isfile(path):
        raise PipelineError(E_AUDIO_FORMAT, "WAV 文件不存在: {}".format(path),
                            "segment", audio_id)
    with open(path, "rb") as handle:
        raw = handle.read()
    if len(raw) < 44 or raw[0:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise PipelineError(E_AUDIO_FORMAT, "RIFF/WAVE 容器头非法", "segment", audio_id)
    riff_size = int.from_bytes(raw[4:8], "little")
    if riff_size + 8 != len(raw):
        raise PipelineError(E_AUDIO_FORMAT, "RIFF 块长度与文件大小不一致", "segment", audio_id)
    offset = 12
    fmt = None
    data_bytes = None
    while offset + 8 <= len(raw):
        chunk_id = raw[offset:offset + 4]
        chunk_size = int.from_bytes(raw[offset + 4:offset + 8], "little")
        body_start = offset + 8
        body_end = body_start + chunk_size
        if body_end > len(raw):
            raise PipelineError(E_AUDIO_FORMAT, "{} 块越界".format(chunk_id), "segment", audio_id)
        if chunk_id == b"fmt ":
            fmt = raw[body_start:body_end]
        elif chunk_id == b"data":
            data_bytes = raw[body_start:body_end]
        offset = body_end + (chunk_size & 1)
    if fmt is None or data_bytes is None:
        raise PipelineError(E_AUDIO_FORMAT, "缺少 fmt 或 data 块", "segment", audio_id)
    audio_format = int.from_bytes(fmt[0:2], "little")
    channels = int.from_bytes(fmt[2:4], "little")
    sample_rate = int.from_bytes(fmt[4:8], "little")
    bits_per_sample = int.from_bytes(fmt[14:16], "little")
    if audio_format != 1 or channels != 1 or sample_rate != 16000 or bits_per_sample != 16:
        raise PipelineError(
            E_AUDIO_FORMAT,
            "fmt 参数不合规: format={} ch={} sr={} bits={}".format(
                audio_format, channels, sample_rate, bits_per_sample),
            "segment", audio_id,
            extra={"channels": channels, "sample_rate": sample_rate,
                   "bits_per_sample": bits_per_sample},
        )
    if len(data_bytes) % 2 != 0:
        raise PipelineError(E_AUDIO_FORMAT, "data 块长度不是 16bit 采样的整数倍",
                            "segment", audio_id)
    samples = np.frombuffer(data_bytes, dtype="<i2").astype(np.int64)
    expected = len(data_bytes) // 2
    if samples.size != expected:
        raise PipelineError(E_AUDIO_FORMAT, "data 块长度与采样数不一致", "segment", audio_id)
    return {
        "audio_id": audio_id,
        "path": path,
        "samples": samples,
        "sample_rate": sample_rate,
        "data_bytes": len(data_bytes),
    }


def frame_energies(samples, frame_length, hop_length):
    """固定起点的无重排分帧 RMS 能量，并归一化到 [0,1]。"""
    if samples.size < frame_length:
        return np.zeros(0, dtype=np.float64)
    frame_count = 1 + (samples.size - frame_length) // hop_length
    starts = np.arange(frame_count) * hop_length
    energies = np.empty(frame_count, dtype=np.float64)
    for index, start in enumerate(starts):
        frame = samples[start:start + frame_length].astype(np.float64)
        energies[index] = np.sqrt(np.mean(frame * frame))
    peak = energies.max() if energies.size else 0.0
    if peak > 0:
        energies = energies / peak
    return energies


def _voiced_runs(energies, high_gate, low_gate):
    """双门限迟滞：高门限进入语音态，低门限才允许退回静音态。"""
    voiced = np.zeros(energies.size, dtype=bool)
    state = False
    for index, energy in enumerate(energies):
        if not state and energy >= high_gate:
            state = True
        elif state and energy <= low_gate:
            state = False
        voiced[index] = state
    runs = []
    start = None
    for index, flag in enumerate(voiced):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, energies.size - 1))
    return runs


def _merge_short_gaps(runs, gap_frames, min_gap_frames):
    """短于最短间隙的静音两侧区间合并。"""
    if not runs:
        return runs
    merged = [runs[0]]
    for run in runs[1:]:
        prev_start, prev_end = merged[-1]
        gap = run[0] - prev_end - 1
        if gap < min_gap_frames:
            merged[-1] = (prev_start, run[1])
        else:
            merged.append(run)
    return merged


def _drop_short_segments(runs, min_segment_frames):
    return [run for run in runs if (run[1] - run[0] + 1) >= min_segment_frames]


def segment_content_fingerprint(audio_id, index, start_sample, end_sample, samples):
    payload = {
        "audio_id": audio_id,
        "segment_index": index,
        "start_sample": int(start_sample),
        "end_sample": int(end_sample),
        "pcm_sha256": sha256_text(samples[start_sample:end_sample].astype("<i2").tobytes()),
    }
    return sha256_text(canonical_json(payload))


def segment_audio(config, audio):
    """输出段记录：起止采样、帧范围、段内容指纹与切分轨迹。"""
    frame_length = config.frame_length()
    hop_length = config.hop_length()
    energies = frame_energies(audio["samples"], frame_length, hop_length)
    raw_runs = _voiced_runs(
        energies,
        config.get("energy_gate_high"),
        config.get("energy_gate_low"),
    )
    min_gap_frames = max(1, int(round(config.get("min_gap_ms") / config.get("hop_ms"))))
    min_segment_frames = max(1, int(round(config.get("min_segment_ms") / config.get("hop_ms"))))
    merged_runs = _merge_short_gaps(raw_runs, min_gap_frames, min_gap_frames)
    final_runs = _drop_short_segments(merged_runs, min_segment_frames)
    segments = []
    trace = {
        "frame_count": int(energies.size),
        "raw_voiced_runs": [list(run) for run in raw_runs],
        "gap_merged_runs": [list(run) for run in merged_runs],
        "final_runs": [list(run) for run in final_runs],
        "dropped_short_runs": [list(run) for run in merged_runs if run not in final_runs],
    }
    for index, (frame_start, frame_end) in enumerate(final_runs):
        start_sample = frame_start * hop_length
        end_sample = min(audio["samples"].size, frame_end * hop_length + frame_length)
        fingerprint = segment_content_fingerprint(
            audio["audio_id"], index, start_sample, end_sample, audio["samples"])
        segments.append({
            "segment_id": "{}#s{:06d}".format(audio["audio_id"], index),
            "segment_index": index,
            "start_sample": int(start_sample),
            "end_sample": int(end_sample),
            "frame_start": int(frame_start),
            "frame_end": int(frame_end),
            "frame_count": int(frame_end - frame_start + 1),
            "content_fingerprint": fingerprint,
            "origin": "segmentation",
            "edit_history": [],
        })
    return {
        "audio_id": audio["audio_id"],
        "segments": segments,
        "energies": [float(v) for v in energies],
        "trace": trace,
    }


def scan_wav_dir(wav_dir):
    return sorted(
        os.path.join(wav_dir, name)
        for name in os.listdir(wav_dir)
        if name.lower().endswith(".wav")
    )
