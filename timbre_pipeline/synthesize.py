"""固定种子确定性样本生成器。

合成 16kHz/单声道/16bit PCM WAV：
- 说话人差异体现在基频统计与共振峰中心；
- 发音风格差异体现在基频抖动、语速（音节率）与节奏波动；
- 段间插入静音，全程叠加平稳高斯底噪；
- 全部随机量来自固定种子的 numpy.random.Generator，逐次运行位级一致。
"""

import json
import os
import wave

import numpy as np

from .config import DEFAULTS

SAMPLE_RATE = 16000
MAX_ABS_PCM = 32767
SPEAKERS = {
    "spk_alpha": {"f0": 118.0, "formants": (720.0, 1180.0, 2650.0)},
    "spk_beta": {"f0": 165.0, "formants": (560.0, 1680.0, 2480.0)},
    "spk_gamma": {"f0": 228.0, "formants": (440.0, 1980.0, 2900.0)},
}
STYLES = {
    "read": {"rate": 3.0, "jitter": 0.008, "rhythm": 0.05},
    "animated": {"rate": 4.6, "jitter": 0.028, "rhythm": 0.16},
}
SECONDS_PER_SYLLABLE = 0.22
SEGMENT_SYLLABLES = (3, 4, 5, 4, 3)
SEGMENT_GAP_S = 0.35
NOISE_SIGMA = 220.0
MASTER_SEED = 20240517


def _all_pole_den(formants, sr):
    """共振峰串联二阶谐振器的 IIR 分母系数。"""
    den = np.array([1.0], dtype=np.float64)
    for center_hz, bandwidth_hz in ((f, 110.0) for f in formants):
        radius = float(np.exp(-np.pi * bandwidth_hz / sr))
        angle = 2.0 * np.pi * center_hz / sr
        section = np.array([1.0, -2.0 * radius * np.cos(angle), radius * radius])
        den = np.convolve(den, section)
    return den


def _glottal_excitation(length, f0, sr, rng, unvoiced_mix):
    """以 F0 周期脉冲串叠加少量噪声模拟声门激励。"""
    excitation = np.zeros(length, dtype=np.float64)
    period = max(int(round(sr / f0)), 2)
    positions = np.arange(0, length, period)
    if positions.size:
        jitter = rng.integers(-max(1, period // 20), max(2, period // 20 + 1), positions.size)
        positions = np.clip(positions + jitter, 0, length - 1)
        for position in positions:
            for delta, amplitude in ((-1, 0.5), (0, 1.0), (1, 0.5)):
                if 0 <= position + delta < length:
                    excitation[position + delta] += amplitude
    excitation += unvoiced_mix * rng.standard_normal(length)
    return excitation


def _syllable_envelope(length, rate, style_rhythm, sr, rng):
    """音节幅度调制：峰值节奏即可被特征层的音节峰统计还原。"""
    time_axis = np.arange(length, dtype=np.float64) / sr
    phase = 2.0 * np.pi * rate * time_axis
    wobble = style_rhythm * rng.standard_normal(length)
    envelope = 0.55 + 0.45 * np.cos(phase + wobble)
    return np.clip(envelope, 0.08, 1.0)


def synthesize_segment(f0, formants, rate, jitter, rhythm, syllable_count, seed):
    """合成单个发音段，返回 float64 采样与真值标签。"""
    rng = np.random.default_rng(seed)
    duration = syllable_count * SECONDS_PER_SYLLABLE
    length = int(round(SAMPLE_RATE * duration))
    local_f0 = f0 * (1.0 + jitter * rng.standard_normal(length))
    excitation = _glottal_excitation(length, f0, SAMPLE_RATE, rng, unvoiced_mix=0.04)
    den = _all_pole_den(formants, SAMPLE_RATE)
    # 显式按 IIR 差分方程递推，保证跨 numpy 版本结果一致。
    filtered = np.zeros(length, dtype=np.float64)
    for n in range(length):
        acc = excitation[n]
        for k in range(1, len(den)):
            if n >= k:
                acc -= den[k] * filtered[n - k]
        filtered[n] = acc
    envelope = _syllable_envelope(length, rate, rhythm, SAMPLE_RATE, rng)
    clean = filtered * envelope
    # 平滑淡入淡出，避免边界产生伪音节峰。
    fade = min(256, length // 8)
    window = np.ones(length)
    window[:fade] = np.linspace(0.0, 1.0, fade)
    window[-fade:] = np.linspace(1.0, 0.0, fade)
    clean *= window
    peak = np.max(np.abs(clean)) or 1.0
    scale = 0.72 * MAX_ABS_PCM / peak
    pcm = np.rint(clean * scale).astype(np.int64)
    truth = {
        "f0_hz": float(f0),
        "formants_hz": [float(v) for v in formants],
        "syllable_rate_hz": float(rate),
        "duration_s": float(duration),
        "syllables": int(syllable_count),
    }
    local = {"f0_track_mean": float(np.mean(local_f0))}
    truth.update(local)
    return pcm, truth


def _append_silence(buffer, samples, rng):
    gap = np.zeros(samples, dtype=np.int64)
    gap += np.rint(rng.standard_normal(samples) * NOISE_SIGMA).astype(np.int64)
    return np.concatenate([buffer, np.clip(gap, -MAX_ABS_PCM, MAX_ABS_PCM)])


def write_wav(path, pcm16):
    with wave.open(path, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(np.asarray(pcm16, dtype="<i2").tobytes())


def synthesize_clip(audio_id, speaker, style, segment_seed_offset, rng):
    """合成一条包含多个发音段与段间静音的长音频。"""
    spk = SPEAKERS[speaker]
    sty = STYLES[style]
    pieces = []
    labels = []
    cursor = 0
    gap_samples = int(round(SEGMENT_GAP_S * SAMPLE_RATE))
    for idx, syllables in enumerate(SEGMENT_SYLLABLES):
        if idx > 0:
            pieces.append(np.rint(rng.standard_normal(gap_samples) * NOISE_SIGMA).astype(np.int64))
            cursor += gap_samples
        seed = MASTER_SEED + segment_seed_offset + idx * 101
        pcm, truth = synthesize_segment(
            spk["f0"], spk["formants"], sty["rate"],
            sty["jitter"], sty["rhythm"], syllables, seed,
        )
        labels.append({
            "segment_index": idx,
            "start_sample": cursor,
            "end_sample": cursor + pcm.size,
            "speaker": speaker,
            "style": style,
            "truth": truth,
        })
        pieces.append(pcm)
        cursor += pcm.size
    signal = np.concatenate(pieces)
    noise = np.rint(rng.standard_normal(signal.size) * NOISE_SIGMA).astype(np.int64)
    pcm16 = np.clip(signal + noise, -MAX_ABS_PCM, MAX_ABS_PCM).astype("<i2")
    return pcm16, labels


def write_malformed_wav(path):
    """声明的数据块长度与实际采样数不一致，用于 E_AUDIO_FORMAT 验收。"""
    good, _ = synthesize_segment(140.0, (600.0, 1500.0, 2500.0), 3.0, 0.01, 0.05, 3, MASTER_SEED + 999)
    pcm16 = np.clip(good, -MAX_ABS_PCM, MAX_ABS_PCM).astype("<i2")
    raw = pcm16.tobytes()
    declared = len(raw) + 8  # 故意多报 4 个采样
    with open(path, "wb") as handle:
        handle.write(b"RIFF")
        handle.write((36 + declared).to_bytes(4, "little"))
        handle.write(b"WAVE")
        handle.write(b"fmt ")
        handle.write((16).to_bytes(4, "little"))
        handle.write((1).to_bytes(2, "little"))
        handle.write((1).to_bytes(2, "little"))
        handle.write(SAMPLE_RATE.to_bytes(4, "little"))
        handle.write((SAMPLE_RATE * 2).to_bytes(4, "little"))
        handle.write((2).to_bytes(2, "little"))
        handle.write((16).to_bytes(2, "little"))
        handle.write(b"data")
        handle.write(declared.to_bytes(4, "little"))
        handle.write(raw)


def default_injections(dim_names):
    """生成覆盖四类真实输入故障的独立注入清单（编号 inj-01..）。"""
    return [
        {"id": "inj-01", "type": "non_finite", "audio": "clip_01", "segment": 1, "stage": "feature",
         "target_frame": 2, "enabled": True},
        {"id": "inj-02", "type": "dim_mismatch", "audio": "clip_02", "segment": 0, "stage": "quantize",
         "drop_dim": dim_names[-1], "enabled": True},
        {"id": "inj-03", "type": "threshold_out_of_range", "audio": "clip_03", "segment": 0,
         "stage": "attach", "threshold": 1.7, "enabled": True},
        {"id": "inj-04", "type": "segment_out_of_range", "audio": "clip_01", "segment": 999,
         "stage": "feature", "enabled": True},
    ]


def generate_sample_workspace(root, config_overrides=None):
    """生成完整样本目录：WAV、配置 JSON、注入清单、真值标签。"""
    wav_dir = os.path.join(root, "wav")
    os.makedirs(wav_dir, exist_ok=True)
    rng = np.random.default_rng(MASTER_SEED)
    plan = [
        ("clip_01", "spk_alpha", "read", 1000),
        ("clip_02", "spk_beta", "animated", 2000),
        ("clip_03", "spk_gamma", "read", 3000),
        ("clip_04", "spk_gamma", "animated", 4000),
        ("clip_05", "spk_beta", "read", 5000),
    ]
    labels = {}
    for audio_id, speaker, style, offset in plan:
        pcm16, segment_labels = synthesize_clip(audio_id, speaker, style, offset, rng)
        write_wav(os.path.join(wav_dir, audio_id + ".wav"), pcm16)
        labels[audio_id] = {"speaker": speaker, "style": style, "segments": segment_labels}
    write_malformed_wav(os.path.join(wav_dir, "clip_bad.wav"))
    config = dict(DEFAULTS)
    config["frame_ms"] = 40.0
    config["hop_ms"] = 10.0
    config["energy_gate_high"] = 0.28
    config["energy_gate_low"] = 0.06
    config["similarity_threshold"] = 0.78
    config["first_response_budget_ms"] = 3000.0
    if config_overrides:
        config.update(config_overrides)
    config_path = os.path.join(root, "pipeline_config.json")
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2, sort_keys=True)
    injection_path = os.path.join(root, "injections.json")
    injections = default_injections([d["name"] for d in config["dimensions"]])
    with open(injection_path, "w", encoding="utf-8") as handle:
        json.dump(injections, handle, ensure_ascii=False, indent=2, sort_keys=True)
    label_path = os.path.join(root, "truth_labels.json")
    with open(label_path, "w", encoding="utf-8") as handle:
        json.dump(labels, handle, ensure_ascii=False, indent=2, sort_keys=True)
    return {
        "root": root,
        "wav_dir": wav_dir,
        "config_path": config_path,
        "injection_path": injection_path,
        "label_path": label_path,
        "audios": [item[0] for item in plan] + ["clip_bad"],
    }
