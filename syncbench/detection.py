"""同步脉冲、同步帧与空洞检测。

- 音频脉冲：与已知三角窗做归一化互相关，在每个参考脉冲的期望窗口内取峰值，
  峰值位置再用抛物线插值得到亚样本中心；
- 视频同步帧：左上角 8x32 亮斑掩膜匹配，输出本地帧号；
- 空洞：音频近零游程 / 视频全黑帧游程，过滤短噪声后输出 [start,end)。
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from .config import FRAME_H, FRAME_W, PULSE_HALF_WIDTH, WorkspaceConfig
from .errors import E_CH_FORMAT, E_PULSE_MISSING, make_record
from .media import SYNC_PIXEL, read_wav

MIN_GAP_AUDIO = 8
MIN_GAP_VIDEO = 1
ZERO_EPS = 1e-6


@dataclass(frozen=True)
class PulseHit:
    ref_sample: int          # 参考通道脉冲样本位置（真值配对键）
    local_peak: int          # 检测到的整数峰值样本
    local_center: float      # 抛物线插值后的亚样本中心
    strength: float          # 归一化相关强度


@dataclass(frozen=True)
class SyncFrameHit:
    ref_frame: int
    local_frame: int
    local_float: float


@dataclass(frozen=True)
class Gap:
    start: int
    end: int
    domain: str

    @property
    def length(self) -> int:
        return self.end - self.start


def _pulse_template() -> np.ndarray:
    r = np.arange(-PULSE_HALF_WIDTH, PULSE_HALF_WIDTH + 1)
    t = 1.0 - np.abs(r) / (PULSE_HALF_WIDTH + 1)
    return t / np.sqrt(np.sum(t * t))


def _parabolic_peak(y: np.ndarray, center: int) -> float:
    """三点抛物线插值，返回亚样本偏移（相对 center）。"""
    if center <= 0 or center >= len(y) - 1:
        return float(center)
    y0, y1, y2 = y[center - 1], y[center], y[center + 1]
    denom = (y0 - 2.0 * y1 + y2)
    if denom == 0.0:
        return float(center)
    offset = 0.5 * (y0 - y2) / denom
    return center + float(np.clip(offset, -1.0, 1.0))


def detect_pulses(audio: np.ndarray, ref_centers: np.ndarray,
                  cfg: WorkspaceConfig) -> tuple[list[PulseHit], list]:
    template = _pulse_template()
    # 脉冲注入电平（~0.9）远高于背景信号（<0.5），使用绝对互相关幅值，
    # 避免低能量背景段因归一化产生虚高峰值
    score = np.correlate(audio, template, mode="same")
    hits: list[PulseHit] = []
    errors = []
    # 全序列找候选峰，再按参考脉冲顺序做单调贪婪配对；
    # 非极大值抑制半径取半周期，天然杜绝跨周期误峰，且容忍高漂移累积偏移
    min_dist = int(round(cfg.pulse_period_sec * cfg.audio_rate * 0.5))
    candidates: list[int] = []
    order = np.argsort(score)[::-1]
    for idx in order:
        idx = int(idx)
        if score[idx] < 0.5:
            break
        if all(abs(idx - c) > min_dist for c in candidates):
            candidates.append(idx)
        if len(candidates) >= len(ref_centers):
            break
    candidates.sort()
    if len(candidates) < len(ref_centers):
        errors.append(make_record(
            E_PULSE_MISSING,
            f"候选脉冲数不足 expected={len(ref_centers)} "
            f"detected={len(candidates)}",
            detail={"expected": int(len(ref_centers)),
                    "detected": len(candidates)}))
    for i, ref in enumerate(ref_centers):
        if i >= len(candidates):
            errors.append(make_record(E_PULSE_MISSING,
                                      f"参考脉冲 {int(ref)} 无配对候选",
                                      segment=int(ref)))
            continue
        peak = candidates[i]
        sub = _parabolic_peak(score, peak)
        hits.append(PulseHit(int(ref), int(peak), float(sub), float(score[peak])))
    return hits, errors


def detect_sync_frames(video: np.ndarray, ref_sync_frames: np.ndarray
                       ) -> tuple[list[SyncFrameHit], list]:
    mask = np.zeros((FRAME_H, FRAME_W), dtype=bool)
    mask[0:8, 0:32] = True
    bright = np.mean(video[:, mask], axis=(1,))
    is_sync = bright > (SYNC_PIXEL - 12)
    local_frames = np.nonzero(is_sync)[0]
    hits: list[SyncFrameHit] = []
    errors = []
    if len(local_frames) != len(ref_sync_frames):
        errors.append(make_record(
            E_PULSE_MISSING,
            f"同步帧数量不符 ref={len(ref_sync_frames)} detected={len(local_frames)}",
            detail={"expected": int(len(ref_sync_frames)),
                    "detected": int(len(local_frames))}))
    # 亚帧相位：第 10 行、第 40 列起每 2 列一个 bit（小端 16 位）
    for i, lf in enumerate(local_frames):
        ref = int(ref_sync_frames[i]) if i < len(ref_sync_frames) else -1
        phase16 = 0
        for bit in range(16):
            if video[lf, 10, 40 + bit * 2] > (SYNC_PIXEL - 12):
                phase16 |= (1 << bit)
        sub = phase16 / 65535.0
        hits.append(SyncFrameHit(ref, int(lf), float(lf) + sub))
    return hits, errors


def detect_gaps_audio(audio: np.ndarray) -> list[Gap]:
    zero = np.abs(audio) < ZERO_EPS
    return _runs(zero, MIN_GAP_AUDIO, "audio")


def detect_gaps_video(video: np.ndarray) -> list[Gap]:
    black = np.all(video.reshape(video.shape[0], -1) == 0, axis=1)
    return _runs(black, MIN_GAP_VIDEO, "video")


def _runs(mask: np.ndarray, min_len: int, domain: str) -> list[Gap]:
    gaps: list[Gap] = []
    n = len(mask)
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if j - i >= min_len:
                gaps.append(Gap(int(i), int(j), domain))
            i = j
        else:
            i += 1
    return gaps


def validate_manifest_media(manifest: dict, ws_dir: str) -> list:
    """素材格式校验：通道数、采样率、位深、帧尺寸。"""
    errors = []
    if len(manifest["channels"]) != 4:
        errors.append(make_record(E_CH_FORMAT,
                                  f"通道数应为 4，实际 {len(manifest['channels'])}"))
    for ch in manifest["channels"]:
        wav_path = os.path.join(ws_dir, ch["audio"]["path"])
        try:
            import wave
            with wave.open(wav_path, "rb") as wf:
                if wf.getframerate() != manifest["audio_rate"]:
                    errors.append(make_record(E_CH_FORMAT, "采样率不符",
                                              channel=ch["channel"]))
                if wf.getsampwidth() != 3 or wf.getnchannels() != 1:
                    errors.append(make_record(E_CH_FORMAT,
                                              "非 24bit 单声道 PCM",
                                              channel=ch["channel"]))
        except (OSError, wave.Error) as exc:  # pragma: no cover - 防御
            errors.append(make_record(E_CH_FORMAT, f"WAV 无法读取: {exc}",
                                      channel=ch["channel"]))
        raw_path = os.path.join(ws_dir, ch["video"]["path"])
        if not os.path.exists(raw_path):
            errors.append(make_record(E_CH_FORMAT, "视频原始文件缺失",
                                      channel=ch["channel"]))
            continue
        frames = np.fromfile(raw_path, dtype=np.uint8)
        expected = ch["video"]["frames"] * FRAME_H * FRAME_W
        if frames.size != expected:
            errors.append(make_record(E_CH_FORMAT,
                                      f"视频字节数不符 {frames.size}!={expected}",
                                      channel=ch["channel"]))
    return errors


def load_video(ch_entry: dict, ws_dir: str) -> np.ndarray:
    path = os.path.join(ws_dir, ch_entry["video"]["path"])
    return np.fromfile(path, dtype=np.uint8).reshape(
        ch_entry["video"]["frames"], FRAME_H, FRAME_W)


def load_audio(ch_entry: dict, ws_dir: str) -> np.ndarray:
    audio, _ = read_wav(os.path.join(ws_dir, ch_entry["audio"]["path"]))
    return audio
