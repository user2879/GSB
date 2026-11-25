"""逐窗定点声学特征：梅尔能量、过零率、谱质心，固定顺序累加。"""

from dataclasses import dataclass

import numpy as np

from .config import MelConfig, PipelineConfig, QuantConfig, VadConfig
from .errors import (
    BusinessError,
    E_QUANT_LEVEL_INVALID,
    E_SPAN_OUT_OF_RANGE,
)
from .fingerprint import (
    PREFIX_FRAGMENT,
    PREFIX_WINDOW_FEATURE,
    cache_key,
    content_fingerprint,
)
from .windowing import AnalysisWindow, fragment_offsets


@dataclass(frozen=True)
class FrameFeature:
    frame_index: int
    start_sample: int
    values: tuple[float, ...]  # 顺序：n_mels 梅尔 + zcr + centroid


@dataclass(frozen=True)
class WindowFeature:
    window_index: int
    fragment_seqs: tuple[int, ...]
    span: tuple[int, int]
    frame_count: int
    frame_features: tuple[FrameFeature, ...]
    fixed_point: tuple[int, ...]
    cache_key: str
    fragment_key: str


def _hz_to_mel(hz: float) -> float:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(cfg_mel: MelConfig, sample_rate: int) -> np.ndarray:
    n_bins = cfg_mel.n_fft // 2 + 1
    bank = np.zeros((cfg_mel.n_mels, n_bins), dtype=np.float64)
    mel_points = np.linspace(
        _hz_to_mel(cfg_mel.low_hz), _hz_to_mel(cfg_mel.high_hz), cfg_mel.n_mels + 2
    )
    hz_points = np.array([_mel_to_hz(mel) for mel in mel_points])
    bin_points = np.floor((cfg_mel.n_fft + 1) * hz_points / sample_rate).astype(int)
    for mel in range(cfg_mel.n_mels):
        left, center, right = bin_points[mel], bin_points[mel + 1], bin_points[mel + 2]
        if right <= left:
            right = left + 1
        for bin_index in range(left, center):
            if center > left:
                bank[mel, bin_index] = (bin_index - left) / max(center - left, 1)
        for bin_index in range(center, right):
            if right > center:
                bank[mel, bin_index] = (right - bin_index) / max(right - center, 1)
    return bank


def frame_feature(block: np.ndarray, frame_index: int, start_sample: int,
                  cfg: PipelineConfig, bank: np.ndarray) -> FrameFeature:
    windowed = block.astype(np.float64) * np.hanning(len(block))
    spectrum = np.abs(np.fft.rfft(windowed, n=cfg.mel.n_fft)) ** 2
    mel_energies = bank @ spectrum
    mel_log = np.log(np.maximum(mel_energies, 1e-12))
    crossings = np.count_nonzero(np.diff(np.sign(block.astype(np.float64))) != 0)
    zcr = float(crossings) / max(len(block) - 1, 1)
    freqs = np.fft.rfftfreq(cfg.mel.n_fft, d=1.0 / cfg.sample.sample_rate)
    centroid = float(np.sum(freqs * spectrum) / max(np.sum(spectrum), 1e-12))
    values = tuple(float(v) for v in mel_log) + (zcr, centroid)
    return FrameFeature(frame_index, start_sample, values)


def validate_span(window: AnalysisWindow, total_samples: int) -> None:
    if window.start_sample < 0 or window.end_sample > total_samples or \
            window.start_sample >= window.end_sample:
        raise BusinessError(
            E_SPAN_OUT_OF_RANGE,
            f"窗 {window.window_index} 采样区间 "
            f"[{window.start_sample},{window.end_sample}) 越界，总采样 {total_samples}",
            stage="window_feature",
            window_index=window.window_index,
        )


def quantize_value(value: float, full_scale: float, quant: QuantConfig) -> int:
    if quant.quant_level not in quant.allowed_levels:
        raise BusinessError(
            E_QUANT_LEVEL_INVALID,
            f"量化档位 {quant.quant_level} 不在允许枚举 {quant.allowed_levels} 内",
            stage="quantize",
        )
    # 定点小数位宽随档位变化：LSB = 2^(level-1)/(float_scale)，
    # 因此切换位宽/档位会真实改变量化后的中间整数。
    fractional_bits = max(quant.quant_level - 8, 1)
    step_scale = quant.float_scale * (1 << (fractional_bits - 1))
    scaled = value * step_scale / full_scale
    max_magnitude = 1 << (quant.quant_level - 1)
    clipped = min(max(scaled, -max_magnitude), max_magnitude - 1)
    return int(round(clipped))


def full_scales(cfg: PipelineConfig) -> list[float]:
    return (
        [cfg.quant.mel_full_scale] * cfg.mel.n_mels
        + [cfg.quant.zcr_full_scale]
        + [cfg.quant.centroid_full_scale]
    )


def extract_window_feature(
    window: AnalysisWindow,
    global_audio: np.ndarray,
    cfg: PipelineConfig,
    bank: np.ndarray,
    fragment_key_by_seq: dict[int, str],
) -> WindowFeature:
    validate_span(window, len(global_audio))
    vad_cfg: VadConfig = cfg.vad
    frames: list[FrameFeature] = []
    start = window.start_sample
    frame_index = 0
    while start + vad_cfg.frame_length <= window.end_sample:
        block = global_audio[start:start + vad_cfg.frame_length]
        frames.append(frame_feature(block, frame_index, start, cfg, bank))
        start += vad_cfg.frame_shift
        frame_index += 1
    if not frames:
        raise BusinessError(
            E_SPAN_OUT_OF_RANGE,
            f"窗 {window.window_index} 内无有效帧",
            stage="window_feature",
            window_index=window.window_index,
        )
    scales = full_scales(cfg)
    accum = [0] * cfg.feature_dim
    for frame in sorted(frames, key=lambda item: item.frame_index):
        for dim in range(cfg.feature_dim):
            accum[dim] += quantize_value(frame.values[dim], scales[dim], cfg.quant)
    frag_key = content_fingerprint(
        sorted((seq, fragment_key_by_seq[seq]) for seq in window.fragment_seqs)
    )
    param_fp = content_fingerprint(
        {
            "vad": cfg.vad.model_dump(),
            "mel": cfg.mel.model_dump(),
            "quant": cfg.quant.model_dump(),
            "span": [window.start_sample, window.end_sample],
        }
    )
    key = cache_key(PREFIX_WINDOW_FEATURE, frag_key, param_fp)
    return WindowFeature(
        window_index=window.window_index,
        fragment_seqs=window.fragment_seqs,
        span=(window.start_sample, window.end_sample),
        frame_count=len(frames),
        frame_features=tuple(frames),
        fixed_point=tuple(accum),
        cache_key=key,
        fragment_key=frag_key,
    )


def fragment_feature_key(fragment, cfg: PipelineConfig) -> str:
    samples_fp = content_fingerprint([int(v) for v in fragment.samples.tolist()])
    param_fp = content_fingerprint(
        {"sample_rate": cfg.sample.sample_rate, "frame": [cfg.vad.frame_length,
                                                           cfg.vad.frame_shift]}
    )
    return cache_key(PREFIX_FRAGMENT, samples_fp, param_fp)


def assemble_global_audio(fragments) -> np.ndarray:
    pieces = []
    for fragment in sorted(fragments, key=lambda item: item.seq):
        pieces.append(fragment.samples)
    return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.int16)


def fragment_offsets_total(fragments) -> tuple[list[tuple[int, int, int]], int]:
    offsets = fragment_offsets(fragments)
    total = offsets[-1][2] if offsets else 0
    return offsets, total
