"""端点检测：帧能量 + 谱平坦度，双门限迟滞与最短语音/静音后处理。"""

from dataclasses import dataclass

import numpy as np

from .config import VadConfig


@dataclass(frozen=True)
class Frame:
    index: int
    start: int
    end: int
    energy: float
    flatness: float


@dataclass(frozen=True)
class SpeechInterval:
    fragment_seq: int
    start_sample: int  # 片段内采样序号
    end_sample: int    # 左闭右开
    start_frame: int
    end_frame: int


def frame_signal(samples: np.ndarray, cfg: VadConfig) -> list[Frame]:
    frames = []
    index = 0
    start = 0
    while start + cfg.frame_length <= len(samples):
        end = start + cfg.frame_length
        block = samples[start:end].astype(np.float64)
        energy = float(np.mean(block * block))
        spectrum = np.abs(np.fft.rfft(block * np.hanning(cfg.frame_length))) ** 2
        spectrum = np.maximum(spectrum, 1e-12)
        flatness = float(
            np.exp(np.mean(np.log(spectrum))) / max(np.mean(spectrum), 1e-12)
        )
        frames.append(Frame(index, start, end, energy, flatness))
        start += cfg.frame_shift
        index += 1
    return frames


def _hysteresis(frames: list[Frame], cfg: VadConfig) -> list[tuple[int, int]]:
    speech_ranges: list[tuple[int, int]] = []
    active = False
    run_start = 0
    for frame in frames:
        strong = (
            frame.energy >= cfg.energy_high
            and frame.flatness <= cfg.flatness_high
        )
        weak = (
            frame.energy >= cfg.energy_low
            and frame.flatness <= cfg.flatness_low
        )
        if not active and strong:
            active = True
            run_start = frame.index
        elif active and not weak:
            speech_ranges.append((run_start, frame.index))
            active = False
    if active:
        speech_ranges.append((run_start, frames[-1].index + 1))
    return speech_ranges


def _post_process(
    ranges: list[tuple[int, int]], frames: list[Frame], cfg: VadConfig
) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in ranges:
        if merged and start - merged[-1][1] <= cfg.min_silence_frames:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    kept = [
        (start, end)
        for start, end in merged
        if end - start >= cfg.min_speech_frames
    ]
    intervals = []
    for start, end in kept:
        intervals.append(
            SpeechInterval(
                fragment_seq=-1,
                start_sample=frames[start].start,
                end_sample=frames[end - 1].end,
                start_frame=start,
                end_frame=end,
            )
        )
    return intervals


def detect_fragment(
    samples: np.ndarray, fragment_seq: int, cfg: VadConfig
) -> list[SpeechInterval]:
    frames = frame_signal(samples, cfg)
    if not frames:
        return []
    raw_ranges = _hysteresis(frames, cfg)
    intervals = _post_process(raw_ranges, frames, cfg)
    return [
        SpeechInterval(
            fragment_seq=fragment_seq,
            start_sample=item.start_sample,
            end_sample=item.end_sample,
            start_frame=item.start_frame,
            end_frame=item.end_frame,
        )
        for item in intervals
    ]
