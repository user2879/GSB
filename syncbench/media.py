"""确定性多通道合成采集素材生成器。

时钟模型（每通道独立晶振，相对参考时钟）：
    rate(u) = 1 + (ppm + thermal_ppm * cos(2*pi*u/period)) / 1e6
    dev(u)  = integral_0^u rate(s) ds            # 参考时间 u -> 设备时间
    content(u) = dev(u) + delay / fs             # 非整数采样固定延迟
音频样本 n 对应参考时间 u_n（对 content(u)=n/fs 做牛顿反演），
视频帧 k 取参考时间 u=k/fps 经同一 content() 映射到本地时间后换算帧号。

输出：
- chN_audio.wav   48kHz/24bit/单声道 PCM
- chN_video.raw   320x180 uint8 灰度帧序列
- manifest.json   全局配置 + 逐通道真值 + 同步脉冲/同步帧真值清单
同种子两次生成逐字节一致（不使用系统时间/全局随机状态）。
"""
from __future__ import annotations

import math
import os
import wave
from dataclasses import asdict
from typing import Any

import numpy as np

from .config import (
    FRAME_H,
    FRAME_W,
    PULSE_HALF_WIDTH,
    WorkspaceConfig,
)
from .hashing import canonical_bytes, hash_array, sha256_hex
from .rng import channel_rng

SYNC_PIXEL = 255
BASE_FLOOR = 40
BASE_CEIL = 200
PULSE_LEVEL = 235
QUANT_SCALE = (1 << 23) - 1


# --- 时钟物理模型 ---------------------------------------------------------------
def thermal_curve(u: np.ndarray | float, ppm: float, thermal_ppm: float,
                  period: float) -> np.ndarray | float:
    if thermal_ppm == 0.0:
        return np.zeros_like(np.asarray(u, dtype=np.float64)) if np.ndim(u) else 0.0
    return thermal_ppm * np.cos(2.0 * math.pi * np.asarray(u, dtype=np.float64) / period)


def device_time(u: np.ndarray, ppm: float, thermal_ppm: float,
                period: float) -> np.ndarray:
    """参考时间 u（秒）-> 设备时间（秒）。"""
    u = np.asarray(u, dtype=np.float64)
    linear = (1.0 + ppm * 1e-6) * u
    if thermal_ppm == 0.0:
        return linear
    thermal = (thermal_ppm * 1e-6 * period / (2.0 * math.pi)
               * np.sin(2.0 * math.pi * u / period))
    return linear + thermal


def rate_at(u: float, ppm: float, thermal_ppm: float, period: float) -> float:
    return 1.0 + (ppm + float(thermal_curve(u, ppm, thermal_ppm, period))) * 1e-6


def invert_content_time(target_local: np.ndarray, ppm: float, thermal_ppm: float,
                        period: float, delay: float, fs: int) -> np.ndarray:
    """对 content(u) = dev(u) + delay/fs 做牛顿反演，返回参考时间 u。"""
    target_time = target_local / fs - delay / fs
    u = np.asarray(target_time, dtype=np.float64).copy()
    for _ in range(6):  # 6 次牛顿迭代，远超双精度收敛需求
        f = device_time(u, ppm, thermal_ppm, period) - target_time
        d = 1.0 + (ppm + thermal_curve(u, ppm, thermal_ppm, period)) * 1e-6
        u -= f / d
    return u


# --- 24bit PCM 读写 -------------------------------------------------------------
def float_to_pcm24(samples: np.ndarray) -> bytes:
    clipped = np.clip(samples, -1.0, 1.0)
    ints = np.round(clipped * QUANT_SCALE).astype(np.int32)
    ints = ints.reshape(-1, 1)
    b0 = ints & 0xFF
    b1 = (ints >> 8) & 0xFF
    b2 = (ints >> 16) & 0xFF
    out = np.empty((ints.shape[0], 3), dtype=np.uint8)
    out[:, 0] = b0[:, 0]
    out[:, 1] = b1[:, 0]
    out[:, 2] = b2[:, 0]
    return out.tobytes()


def pcm24_to_float(raw: bytes) -> np.ndarray:
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
    ints = arr[:, 0] | (arr[:, 1] << 8) | (arr[:, 2] << 16)
    ints = np.where(ints & 0x800000, ints - (1 << 24), ints)
    return ints.astype(np.float64) / QUANT_SCALE


def write_wav(path: str, samples: np.ndarray, rate: int) -> None:
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(3)
        wf.setframerate(rate)
        wf.writeframes(float_to_pcm24(samples))


def read_wav(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, "rb") as wf:
        if wf.getnchannels() != 1 or wf.getsampwidth() != 3:
            raise ValueError("bad wav format")
        rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    return pcm24_to_float(raw), rate


# --- 源信号（参考时间域，确定性） -----------------------------------------------
def audio_source(u: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    sig = (
        0.32 * np.sin(2 * math.pi * 220.0 * u)
        + 0.22 * np.sin(2 * math.pi * 660.0 * u + 0.7)
        + 0.12 * np.sin(2 * math.pi * 1500.0 * u + 1.9)
    )
    sig += rng.normal(0.0, 0.01, size=u.shape)
    return sig * 0.8


def pulse_train_samples(n: int, fs: int, period_sec: float) -> np.ndarray:
    """参考域脉冲中心样本位置（含半宽窗）。"""
    first = int(round(0.4 * fs))
    period = int(round(period_sec * fs))
    centers = np.arange(first, n, period, dtype=np.int64)
    return centers


def render_frame(k: int, is_sync: bool, rng: np.random.Generator,
                 sub_phase: float = 0.0) -> np.ndarray:
    frame = np.empty((FRAME_H, FRAME_W), dtype=np.uint8)
    base = BASE_FLOOR + int(30 * (0.5 + 0.5 * math.sin(k * 0.37)))
    frame.fill(base)
    # 确定性条纹（内容随帧号变化，便于区分）
    for y in range(FRAME_H):
        frame[y, :] = np.uint8(
            np.clip(base + 40 * np.sin((y + k) * 0.25) + 20 * np.sin(y * 0.07),
                    0, BASE_CEIL))
    if is_sync:
        # 左上 8 行常量亮块：同步存在标记
        frame[0:8, 0:32] = SYNC_PIXEL
        # 亚帧相位：在第 10 行用 16 个像素列编码 16 位无符号整数
        # phase16 = round(phase * 65535)，量化误差 < 1.6e-5 帧
        phase16 = int(round(float(np.clip(sub_phase, 0.0, 1.0)) * 65535))
        for bit in range(16):
            if (phase16 >> bit) & 1:
                frame[10, 40 + bit * 2] = SYNC_PIXEL
        frame[FRAME_H - 4:FRAME_H, FRAME_W - 32:FRAME_W] = SYNC_PIXEL
    return frame


def _apply_gaps(values: np.ndarray, gaps: tuple) -> list[dict[str, int]]:
    applied = []
    for g in gaps:
        start = max(0, g.start)
        end = min(values.shape[0], g.end)
        if end > start:
            values[start:end] = 0
            applied.append({"start": int(start), "end": int(end),
                            "length": int(end - start)})
    return applied


# --- 主生成入口 -----------------------------------------------------------------
def generate_workspace(cfg: WorkspaceConfig, ws_dir: str) -> dict[str, Any]:
    media_dir = os.path.join(ws_dir, "media")
    os.makedirs(media_dir, exist_ok=True)
    fs = cfg.audio_rate
    n_audio = int(round(cfg.duration_sec * fs))
    n_video = int(round(cfg.duration_sec * cfg.video_fps))
    ref_centers = pulse_train_samples(n_audio, fs, cfg.pulse_period_sec)
    ref_sync_frames = np.arange(
        int(round(0.4 * cfg.video_fps)), n_video,
        int(round(cfg.pulse_period_sec * cfg.video_fps)), dtype=np.int64)

    manifest: dict[str, Any] = {
        "schema": "syncbench-manifest/1",
        "seed": cfg.seed,
        "audio_rate": fs,
        "video_fps": cfg.video_fps,
        "frame_w": cfg.frame_w,
        "frame_h": cfg.frame_h,
        "duration_sec": cfg.duration_sec,
        "pulse_period_sec": cfg.pulse_period_sec,
        "drift_tol_ppm": cfg.drift_tol_ppm,
        "audio_gap_max": cfg.audio_gap_max,
        "video_gap_max": cfg.video_gap_max,
        "av_sync_tol_sec": cfg.av_sync_tol_sec,
        "reference_pulse_samples": [int(x) for x in ref_centers],
        "reference_sync_frames": [int(x) for x in ref_sync_frames],
        "channels": [],
    }

    for truth in cfg.truths:
        ch = truth.channel
        rng = channel_rng(ch, cfg.seed)
        idx = np.arange(n_audio, dtype=np.float64)
        u_audio = invert_content_time(
            idx, truth.ppm, truth.thermal_ppm, truth.thermal_period_sec,
            truth.delay_samples, fs)
        audio = audio_source(u_audio, rng)

        # 脉冲窗：在参考域脉冲中心附近放置高电平（三角窗）
        pulse_truth_local: list[dict[str, Any]] = []
        for c in ref_centers:
            uc = c / fs
            local_center_time = device_time(
                uc, truth.ppm, truth.thermal_ppm, truth.thermal_period_sec
            ) + truth.delay_samples / fs
            local_center = local_center_time * fs
            lo = int(math.floor(local_center)) - PULSE_HALF_WIDTH
            hi = int(math.floor(local_center)) + PULSE_HALF_WIDTH + 1
            for n in range(max(0, lo), min(n_audio, hi)):
                w = max(0.0, 1.0 - abs(n - local_center) / (PULSE_HALF_WIDTH + 1))
                audio[n] = max(audio[n], 0.9 * (0.4 + 0.6 * w))
            pulse_truth_local.append({
                "ref_sample": int(c),
                "local_center": float(local_center),
                "local_sample_rounded": int(round(local_center)),
            })

        audio_gaps = [g for g in truth.gaps if g.domain == "audio"]
        video_gaps = [g for g in truth.gaps if g.domain == "video"]
        applied_audio_gaps = _apply_gaps(audio, tuple(audio_gaps))

        wav_path = os.path.join(media_dir, f"ch{ch}_audio.wav")
        write_wav(wav_path, audio, fs)
        with open(wav_path, "rb") as fh:
            wav_hash = sha256_hex(fh.read())

        # 视频（正向映射，与音频同方向）：参考帧 k 经 content(u) 得到
        # 本地帧位置 local_frame_float；整数部分决定落到哪个本地帧槽，
        # 小数部分作为亚帧相位编码进同步像素，30fps 取整不再抹掉漂移。
        video = np.zeros((n_video, FRAME_H, FRAME_W), dtype=np.uint8)
        frame_truth: list[dict[str, Any]] = []
        sync_set = set(int(x) for x in ref_sync_frames)
        ref_slots = np.arange(n_video, dtype=np.float64)
        u_ref = ref_slots / cfg.video_fps
        local_time = device_time(
            u_ref, truth.ppm, truth.thermal_ppm, truth.thermal_period_sec
        ) + truth.delay_samples / fs
        local_frame_float = local_time * cfg.video_fps
        local_frame_idx = np.clip(np.floor(local_frame_float).astype(np.int64),
                                  0, n_video - 1)
        dropped_local: set[int] = set()
        for g in video_gaps:
            for lf in range(max(0, g.start), min(n_video, g.end)):
                dropped_local.add(lf)
        applied_video_gaps: list[dict[str, int]] = []
        if video_gaps:
            for g in video_gaps:
                start = max(0, g.start)
                end = min(n_video, g.end)
                if end > start:
                    applied_video_gaps.append(
                        {"start": start, "end": end, "length": end - start})
        frng = channel_rng(100 + ch, cfg.seed)
        local_sync_frames: list[int] = []
        for k in range(n_video):
            lf = int(local_frame_idx[k])
            is_sync = k in sync_set
            sub_phase = float(local_frame_float[k]
                              - math.floor(local_frame_float[k]))
            if is_sync:
                local_sync_frames.append(lf)
            frame_truth.append({
                "ref_frame": int(k),
                "local_frame_float": float(local_frame_float[k]),
                "local_frame": lf,
                "is_sync": is_sync,
                "sub_phase": sub_phase,
            })
            if lf not in dropped_local:
                video[lf] = render_frame(k, is_sync, frng, sub_phase)
        raw_path = os.path.join(media_dir, f"ch{ch}_video.raw")
        video.tofile(raw_path)
        with open(raw_path, "rb") as fh:
            raw_hash = sha256_hex(fh.read())

        manifest["channels"].append({
            "channel": ch,
            "truth": asdict(truth),
            "audio": {"path": os.path.relpath(wav_path, ws_dir).replace(os.sep, "/"),
                      "samples": n_audio, "sha256": wav_hash,
                      "float_hash": hash_array(audio)},
            "video": {"path": os.path.relpath(raw_path, ws_dir).replace(os.sep, "/"),
                      "frames": n_video, "sha256": raw_hash,
                      "array_hash": hash_array(video)},
            "pulses": pulse_truth_local,
            "frames": frame_truth,
            "local_sync_frames": local_sync_frames,
            "applied_audio_gaps": applied_audio_gaps,
            "applied_video_gaps": applied_video_gaps,
        })

    manifest_path = os.path.join(ws_dir, "manifest.json")
    tmp = manifest_path + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(canonical_bytes(manifest))
    os.replace(tmp, manifest_path)
    return manifest
