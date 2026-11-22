"""固定种子的本地确定性合成数据生成。

生成内容：
- 16kHz / 单声道 / 16bit PCM WAV 混音样本（基频+共振峰控制的多说话人、
  交替/重叠片段、平稳底噪、段间静音）；
- 分离真值 truth.json；
- 分离配置 config.json；
- 异常注入清单 injections.json。

不依赖任何外部音频文件与网络。
"""

import json
import os
import wave

import numpy as np

SAMPLE_RATE = 16000

# 固定说话人画像：基频(Hz) + 三个共振峰(中心频率Hz, 带宽Hz)
SPEAKERS = {
    "S1": {"f0": 120.0, "formants": [(730.0, 120.0), (1090.0, 100.0), (2440.0, 140.0)]},
    "S2": {"f0": 180.0, "formants": [(270.0, 90.0), (2290.0, 130.0), (3010.0, 120.0)]},
    "S3": {"f0": 235.0, "formants": [(500.0, 110.0), (1500.0, 120.0), (2500.0, 130.0)]},
}
SPEAKER_IDS = list(SPEAKERS.keys())


def _formant_envelope(freq, formants):
    amp = 1.0
    for center, bandwidth in formants:
        amp *= 1.0 / np.sqrt(1.0 + ((freq - center) / bandwidth) ** 2)
    return amp


def speaker_waveform(speaker_id, n_samples, start_sample=0):
    """用基频谐波+共振峰包络合成确定性说话人波形（相位由全局采样位置决定）。"""
    profile = SPEAKERS[speaker_id]
    f0 = profile["f0"]
    idx = np.arange(start_sample, start_sample + n_samples, dtype=np.float64)
    t = idx / SAMPLE_RATE
    signal = np.zeros(n_samples, dtype=np.float64)
    for harmonic in range(1, 13):
        freq = f0 * harmonic
        if freq >= SAMPLE_RATE / 2:
            break
        amp = _formant_envelope(freq, profile["formants"]) / (harmonic ** 0.8)
        signal += amp * np.sin(2.0 * np.pi * freq * t)
    peak = np.max(np.abs(signal)) or 1.0
    return 0.5 * signal / peak


def build_timeline(rng, n_speakers, total_ms):
    """构造确定性时间线：(start_ms, end_ms, speaker_id, gain) 与重叠窗口。

    独奏段之间保留 150/200ms 静音（大于最短静音门限 120ms），保证 VAD
    切成独立段；重叠区通过“下一说话人在当前独奏段尾部提前淡入”构造，
    使相邻段在时间上相交并被标记为重叠候选。
    """
    speakers = SPEAKER_IDS[:n_speakers]
    segments = []
    cursor = 200
    order = 0
    solo_lengths = [500, 600, 700, 550]
    while cursor + 400 <= total_ms:
        # 第 4 段起让首个说话人重现，保证同说话人跨段可被聚为同一簇
        spk = speakers[0] if order >= len(speakers) else speakers[order]
        length = solo_lengths[order % len(solo_lengths)]
        segments.append((cursor, cursor + length, spk, 0.6))
        # 每隔一个独奏段，让一个不同的说话人在其尾部 200ms 提前淡入形成重叠
        if order % 2 == 1 and order + 1 < 6 and len(speakers) > 1:
            current = speakers[0] if order >= len(speakers) else speakers[order]
            other = next(spk for spk in speakers if spk != current)
            segments.append((cursor + length - 200, cursor + length, other, 0.55))
        cursor += length
        cursor += 150 if order % 2 == 0 else 200  # 段间静音（> 最短静音门限）
        order += 1
    return segments


def build_truth(segments, n_speakers):
    speaker_regions = {spk: [] for spk in SPEAKER_IDS[:n_speakers]}
    overlap_regions = []
    # 重叠窗口：被多个段同时覆盖的时间点
    boundaries = sorted({s for s, _, _, _ in segments} | {e for _, e, _, _ in segments})
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        active = [spk for start, end, spk, _ in segments if start <= left and end >= right]
        if len(active) > 1:
            overlap_regions.append([left, right])
    for start, end, spk, _ in segments:
        speaker_regions[spk].append([start, end])
    for spk, regions in speaker_regions.items():
        merged = []
        for region in sorted(regions):
            if merged and region[0] <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], region[1])
            else:
                merged.append(list(region))
        speaker_regions[spk] = merged
    return {
        "expected_speakers": n_speakers,
        "speaker_regions": speaker_regions,
        "overlap_regions": overlap_regions,
    }


def mix_timeline(segments, total_ms, rng):
    total_samples = int(total_ms * SAMPLE_RATE / 1000)
    mix = np.zeros(total_samples, dtype=np.float64)
    for start_ms, end_ms, spk, gain in segments:
        start = int(start_ms * SAMPLE_RATE / 1000)
        end = int(end_ms * SAMPLE_RATE / 1000)
        wave_part = speaker_waveform(spk, end - start, start_sample=start)
        mix[start:end] += gain * wave_part
    # 平稳底噪（固定种子高斯噪声，幅度恒定）
    noise = rng.normal(0.0, 0.008, size=total_samples)
    mix += noise
    peak = np.max(np.abs(mix))
    if peak > 0.98:
        mix *= 0.98 / peak
    return np.clip(mix, -1.0, 1.0)


def write_wav(path, samples_float):
    pcm16 = np.clip(np.round(np.asarray(samples_float) * 32767.0), -32768, 32767).astype("<i2")
    with wave.open(path, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm16.tobytes())
    return pcm16


def default_config():
    return {
        "frame_length_ms": 25.0,
        "frame_shift_ms": 10.0,
        "feature_dim": 12,
        "energy_threshold": 0.02,
        "spectral_flatness_threshold": 0.6,
        "cluster_distance_threshold": 0.5,
        "overlap_rule": "nearest",
        "min_speech_ms": 60.0,
        "min_silence_ms": 120.0,
        "frac_bits": 10,
        "max_workers": 4,
        "snapshot_interval": 2,
        "boundary_tolerance_ms": 40.0,
        "confidence_tolerance": 0.1,
    }


def default_injections(file_names):
    """对前五段音频分别注入不同的真实输入故障。"""
    specs = [
        ("wav_truncated", "audio", "截断 WAV 字节，数据块长度与采样数不一致"),
        ("nonfinite_samples", "features", "向上游特征阶段送入非有限采样值"),
        ("threshold_zero", "clusters", "聚类距离阈值被置零"),
        ("range_violation", "segments", "替换区间超出音频帧范围"),
        ("truth_mismatch", "verification", "真值说话人数量与音轨实际不符"),
    ]
    injections = []
    for index, (fault_type, stage, desc) in enumerate(specs):
        if index >= len(file_names) - 1:
            break
        item = {
            "injection_id": f"INJ-{index + 1:03d}",
            "audio": file_names[index],
            "stage": stage,
            "fault_type": fault_type,
            "description": desc,
        }
        if fault_type == "range_violation":
            item["replacement_interval_frames"] = [100000, 100090]
        if fault_type == "truth_mismatch":
            item["forced_expected_speakers"] = 99
        injections.append(item)
    return injections


def generate_dataset(input_dir, seed=20261123, n_files=6, total_ms=4000):
    """生成完整本地数据集，返回 (config_dict, truth_dict, injections)。"""
    os.makedirs(input_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    truth = {}
    file_names = []
    for index in range(n_files):
        name = f"mix_{index + 1:03d}.wav"
        file_names.append(name)
        n_speakers = 2 + (index % 2)  # 2 或 3 个说话人
        segments = build_timeline(rng, n_speakers, total_ms)
        samples = mix_timeline(segments, total_ms, rng)
        write_wav(os.path.join(input_dir, name), samples)
        truth[name] = build_truth(segments, n_speakers)
    config = default_config()
    injections = default_injections(file_names)
    with open(os.path.join(input_dir, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(config, fh, ensure_ascii=False, indent=2, sort_keys=True)
    with open(os.path.join(input_dir, "truth.json"), "w", encoding="utf-8") as fh:
        json.dump(truth, fh, ensure_ascii=False, indent=2, sort_keys=True)
    with open(os.path.join(input_dir, "injections.json"), "w", encoding="utf-8") as fh:
        json.dump(injections, fh, ensure_ascii=False, indent=2, sort_keys=True)
    return config, truth, injections
