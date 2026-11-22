"""逐帧声纹特征抽取：能量、谱平坦度、倒谱系数与定点量化。

累加顺序固定为帧序号升序、维度序号升序（所有向量化沿行/列轴执行，
并在需要显式求和时使用 np.sum 的默认行序）。
"""

import numpy as np

from .codes import E_NONFINITE_SAMPLE, PipelineError
from .util import quantize_vector

EPS = 1e-10


def frame_windows(n_samples, frame_length, frame_shift):
    starts = []
    start = 0
    while start + frame_length <= n_samples:
        starts.append(start)
        start += frame_shift
    return starts


def _spectral_flatness(magnitude):
    power = magnitude ** 2 + EPS
    arithmetic = float(np.mean(power))
    geometric = float(np.exp(np.mean(np.log(power))))
    if arithmetic <= EPS:
        return 0.0
    return float(min(1.0, geometric / arithmetic))


def _cepstrum(frame, feature_dim):
    spectrum = np.fft.rfft(frame)
    log_magnitude = np.log(np.abs(spectrum) + EPS)
    cepstrum = np.fft.irfft(log_magnitude)
    return cepstrum[1:feature_dim + 1]  # 跳过 c0，取声纹维度


def extract_frame(frame_index, start, frame_length, samples, config):
    """单帧计算（纯函数，供线程池按稳定键调度）。"""
    end = start + frame_length
    raw = samples[start:end]
    window = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(frame_length) / (frame_length - 1))
    windowed = raw * window
    energy = float(np.sqrt(np.mean(windowed ** 2) + EPS))
    magnitude = np.abs(np.fft.rfft(windowed))
    sfm = _spectral_flatness(magnitude)
    cepstrum = _cepstrum(windowed, config.feature_dim)
    cep_q, step, residual = quantize_vector(cepstrum, config.frac_bits)
    return {
        "sort_key": f"frame:{frame_index:08d}",
        "frame_index": frame_index,
        "start_sample": int(start),
        "end_sample": int(end),
        "start_ms": round(start * 1000.0 / 16000.0, 6),
        "end_ms": round(end * 1000.0 / 16000.0, 6),
        "energy": energy,
        "sfm": sfm,
        "cepstrum": [float(x) for x in cepstrum],
        "cepstrum_q": [int(x) for x in cep_q],
        "residual": [float(x) for x in residual],
    }, step


def extract_features(audio_record, config, registry=None, audio_name=None):
    audio_name = audio_name or audio_record["audio"]
    samples = audio_record["samples"]
    if registry is not None:
        samples, _ = registry.prepare_samples(audio_name, samples)
    if not np.all(np.isfinite(samples)):
        first_bad = int(np.argmax(~np.isfinite(samples)))
        raise PipelineError(
            E_NONFINITE_SAMPLE, "features",
            f"检测到非有限采样值，首个位置 sample_index={first_bad}",
            injection_id=(registry.get(audio_name, "features") or {}).get("injection_id")
            if registry else None,
        )
    frame_length = int(round(config.frame_length_ms * 16000.0 / 1000.0))
    frame_shift = int(round(config.frame_shift_ms * 16000.0 / 1000.0))
    starts = frame_windows(len(samples), frame_length, frame_shift)
    frames = []
    quant_step = None
    for frame_index, start in enumerate(starts):  # 帧序号升序
        record, quant_step = extract_frame(
            frame_index, start, frame_length, samples, config
        )
        frames.append(record)
    return {
        "audio": audio_name,
        "n_samples": int(len(samples)),
        "n_frames": len(frames),
        "frame_length": frame_length,
        "frame_shift": frame_shift,
        "frac_bits": config.frac_bits,
        "quant_step": quant_step,
        "frames": frames,
    }
