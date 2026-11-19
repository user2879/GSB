"""四阶段确定性修复管线。

固定顺序：denoise（谱减） -> gate（残留噪声门限）
       -> normalize（增益归一） -> limit（限幅）

全部运算 float64、单线程确定性公式，同参数同输入两次运行逐样本一致。
STFT 规格取自 config 并固化进 ENGINE_VERSION。
"""
from __future__ import annotations

import numpy as np

from . import config
from .audio_io import downmix_to_mono

STAGE_ORDER = ("denoise", "gate", "normalize", "limit")
GATE_FRAME = 256
EPS = 1e-12


# ---------------- 阶段 1：谱减降噪 ----------------
def _stft(x: np.ndarray, nfft: int, hop: int, window: np.ndarray) -> np.ndarray:
    n_frames = 1 + (x.shape[0] - nfft) // hop
    frames = np.lib.stride_tricks.as_strided(
        x,
        shape=(n_frames, nfft),
        strides=(x.strides[0] * hop, x.strides[0]),
        writeable=False,
    )
    return np.fft.rfft(frames * window[None, :], n=nfft, axis=1)


def _istft(spec: np.ndarray, nfft: int, hop: int, window: np.ndarray) -> np.ndarray:
    n_frames = spec.shape[0]
    out_len = nfft + (n_frames - 1) * hop
    signal = np.zeros(out_len, dtype=np.float64)
    weight = np.zeros(out_len, dtype=np.float64)
    frames = np.fft.irfft(spec, n=nfft, axis=1) * window[None, :]
    w_sq = window**2
    for i in range(n_frames):
        start = i * hop
        signal[start : start + nfft] += frames[i]
        weight[start : start + nfft] += w_sq
    return signal / np.maximum(weight, EPS)


def denoise_stage(x: np.ndarray, params: dict, sample_rate: int) -> np.ndarray:
    nfft = config.STFT_NFFT
    hop = config.STFT_HOP
    window = np.hanning(nfft).astype(np.float64)
    orig_len = x.shape[0]
    # 末尾补零，保证所有样本都被至少一帧完整覆盖
    pad = (-orig_len) % hop
    if pad:
        x = np.concatenate([x, np.zeros(pad, dtype=x.dtype)])
    spec = _stft(x, nfft, hop, window)
    mag = np.abs(spec)
    phase = np.exp(1j * np.angle(spec))

    noise_n = int(round(params["noise_seconds"] * sample_rate))
    noise_frames = max(1, min(spec.shape[0], noise_n // hop))
    # 确定性：直接取前 noise_frames 帧的平均能量谱作为噪声估计。
    noise_profile = np.mean(mag[:noise_frames] ** 2, axis=0)
    noise_mag = np.sqrt(noise_profile)

    alpha = float(params["over_subtraction"])
    floor = float(params["floor"])
    subtracted = mag - alpha * noise_mag[None, :]
    clean_mag = np.maximum(subtracted, floor * mag)
    clean_spec = clean_mag * phase
    y = _istft(clean_spec, nfft, hop, window)
    return y[:orig_len]


# ---------------- 阶段 2：残留噪声门限 ----------------
def gate_stage(x: np.ndarray, params: dict) -> np.ndarray:
    threshold = 10.0 ** (float(params["threshold_db"]) / 20.0)
    ratio = float(params["ratio"])
    n = x.shape[0]
    n_frames = max(1, (n + GATE_FRAME - 1) // GATE_FRAME)
    gains = np.ones(n_frames, dtype=np.float64)
    for i in range(n_frames):
        start = i * GATE_FRAME
        frame = x[start : start + GATE_FRAME]
        rms = float(np.sqrt(np.mean(frame**2) + EPS))
        if rms < threshold:
            # 确定性软门限：低于门限时按 1/ratio 压低，且越安静压得越深。
            depth = np.clip(rms / max(threshold, EPS), 0.0, 1.0)
            gains[i] = (1.0 / ratio) * (0.5 + 0.5 * depth)
    # 帧增益插值到样本级，避免块边界跳变。
    frame_centers = (np.arange(n_frames) + 0.5) * GATE_FRAME
    sample_idx = np.arange(n, dtype=np.float64)
    sample_gains = np.interp(sample_idx, frame_centers, gains)
    return x * sample_gains


# ---------------- 阶段 3：增益归一 ----------------
def normalize_stage(x: np.ndarray, params: dict) -> np.ndarray:
    target_peak = float(params["target_peak"])
    max_gain = 10.0 ** (float(params["max_gain_db"]) / 20.0)
    peak = float(np.max(np.abs(x)))
    if peak <= EPS:
        return x.copy()
    gain = min(target_peak / peak, max_gain)
    return x * gain


# ---------------- 阶段 4：限幅 ----------------
def limit_stage(x: np.ndarray, params: dict) -> np.ndarray:
    ceiling = float(params["ceiling"])
    return np.tanh(x / ceiling) * ceiling


STAGE_FUNCS = {
    "denoise": denoise_stage,
    "gate": gate_stage,
    "normalize": normalize_stage,
    "limit": limit_stage,
}


def _denoise_channels(
    raw_float: np.ndarray, channels: int, params: dict, sample_rate: int
) -> np.ndarray:
    """逐声道独立谱减，返回形状 (samples, channels)，与通道处理顺序无关。"""
    out = np.empty_like(raw_float, dtype=np.float64)
    for ch in range(channels):
        out[:, ch] = denoise_stage(raw_float[:, ch], params, sample_rate)
    return out


def _gate_channels(denoised: np.ndarray, channels: int, params: dict) -> np.ndarray:
    """逐声道独立门限，返回形状 (samples, channels)，与通道处理顺序无关。"""
    out = np.empty_like(denoised, dtype=np.float64)
    for ch in range(channels):
        out[:, ch] = gate_stage(denoised[:, ch], params)
    return out


def run_stage(
    stage: str,
    raw_float: np.ndarray,
    channels: int,
    project_v2: dict,
    sample_rate: int,
    previous: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """执行单个阶段并返回该阶段中间结果。

    denoise/gate 产物为逐声道数组 (samples, channels)，其内容只取决于本阶段
    参数与素材，与 channel_order 无关；normalize/limit 在门限之后按
    channel_order 决定的下混点工作，产物为单声道 (samples,)。
    """
    previous = previous or {}
    if stage == "denoise":
        return _denoise_channels(
            raw_float, channels, project_v2["denoise"], sample_rate
        )
    if stage == "gate":
        return _gate_channels(previous["denoise"], channels, project_v2["gate"])

    order = project_v2["channel_order"]
    if stage == "normalize":
        norm_p = project_v2["normalize"]
        if order == config.ORDER_MONO_FIRST:
            # 先下混为单声道，再归一（归一看到的是混合信号峰值）
            mono_in = downmix_to_mono(previous["gate"], channels)
            return normalize_stage(mono_in, norm_p)
        if order == config.ORDER_PER_CHANNEL:
            # 每声道独立归一，再等权下混（通道顺序依赖点）
            per_ch = np.empty_like(previous["gate"], dtype=np.float64)
            for ch in range(channels):
                per_ch[:, ch] = normalize_stage(previous["gate"][:, ch], norm_p)
            return downmix_to_mono(per_ch, channels)
        raise ValueError(f"unknown channel_order: {order}")  # pragma: no cover
    if stage == "limit":
        limit_p = project_v2["limit"]
        if order == config.ORDER_MONO_FIRST:
            return limit_stage(previous["normalize"], limit_p)
        if order == config.ORDER_PER_CHANNEL:
            # normalize 阶段已下混为单声道，限幅作用于单声道结果
            return limit_stage(previous["normalize"], limit_p)
        raise ValueError(f"unknown channel_order: {order}")  # pragma: no cover
    raise KeyError(f"unknown stage: {stage}")


def run_full_pipeline(
    raw_float: np.ndarray, channels: int, project_v2: dict, sample_rate: int
) -> dict[str, np.ndarray]:
    """按固定顺序执行四阶段，返回每阶段中间结果（denoise/gate 逐声道）。"""
    results: dict[str, np.ndarray] = {}
    for stage in STAGE_ORDER:
        results[stage] = run_stage(
            stage, raw_float, channels, project_v2, sample_rate, results
        )
    return results
