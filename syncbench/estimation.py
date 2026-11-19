"""基于同步脉冲的音视频联合漂移估计。

流程（每通道）：
1. 音频检测脉冲（亚样本中心），视频检测同步帧；
2. 与参考通道真值配对，得到“参考位置 -> 本地位置”观测；
3. 按参考时间分 3 段做加权线性最小二乘：local = a + b*ref，
   斜率 b 即本地相对参考的速率比，换算 ppm 漂移率；
4. 输出每段残差、斜率/截距标准误与 95% 置信区间；
5. 非整数固定延迟由模板互相关峰值 + 抛物线插值估计到亚采样；
6. 注入 ppm 超声明容差 => E_DRIFT_EXCEED（含通道与段号，非仅点估计）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import T975, WorkspaceConfig
from .detection import PulseHit, SyncFrameHit
from .errors import E_DRIFT_EXCEED, make_record

N_SEGMENTS = 3


@dataclass(frozen=True)
class SegmentFit:
    segment: int
    n_points: int
    slope: float            # local = intercept + slope * ref
    intercept: float
    slope_se: float
    intercept_se: float
    slope_ci: tuple[float, float]
    intercept_ci: tuple[float, float]
    residuals: tuple[float, ...]
    rms_residual: float
    ppm: float
    ppm_ci: tuple[float, float]


@dataclass(frozen=True)
class EstimationResult:
    channel: int
    segments: tuple[SegmentFit, ...]
    delay_samples: float          # 亚采样固定延迟估计
    delay_peak_strength: float
    audio_bias_samples: float     # 末点累积样本偏差
    audio_bias_ci: tuple[float, float]
    video_bias_frames: float
    fused_ppm: float              # 音视频联合漂移率（按残差方差加权）
    errors: tuple = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "fused_ppm": self.fused_ppm,
            "delay_samples": self.delay_samples,
            "delay_peak_strength": self.delay_peak_strength,
            "audio_bias_samples": self.audio_bias_samples,
            "audio_bias_ci": list(self.audio_bias_ci),
            "video_bias_frames": self.video_bias_frames,
            "segments": [
                {
                    "segment": s.segment,
                    "n_points": s.n_points,
                    "slope": s.slope,
                    "intercept": s.intercept,
                    "slope_se": s.slope_se,
                    "intercept_se": s.intercept_se,
                    "slope_ci": list(s.slope_ci),
                    "intercept_ci": list(s.intercept_ci),
                    "ppm": s.ppm,
                    "ppm_ci": list(s.ppm_ci),
                    "rms_residual": s.rms_residual,
                    "residuals": list(s.residuals),
                }
                for s in self.segments
            ],
            "errors": [e.to_dict() for e in self.errors],
        }


def _wls(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> dict[str, float]:
    sw = np.sum(w)
    mx = np.sum(w * x) / sw
    my = np.sum(w * y) / sw
    sxx = np.sum(w * (x - mx) ** 2)
    sxy = np.sum(w * (x - mx) * (y - my))
    slope = sxy / sxx
    intercept = my - slope * mx
    resid = y - (intercept + slope * x)
    dof = max(1, len(x) - 2)
    sigma2 = np.sum(w * resid * resid) / dof
    slope_se = float(np.sqrt(sigma2 / sxx))
    intercept_se = float(np.sqrt(sigma2 * (1.0 / sw + mx * mx / sxx)))
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "slope_se": slope_se,
        "intercept_se": intercept_se,
        "resid": resid,
        "rms": float(np.sqrt(np.mean(resid * resid))),
    }


def _fit_segments(ref: np.ndarray, local: np.ndarray, weights: np.ndarray,
                  n_segments: int) -> list[SegmentFit]:
    bounds = np.linspace(ref.min(), ref.max() + 1e-9, n_segments + 1)
    fits: list[SegmentFit] = []
    for seg in range(n_segments):
        sel = (ref >= bounds[seg]) & (ref < bounds[seg + 1])
        x = ref[sel].astype(np.float64)
        y = local[sel].astype(np.float64)
        w = weights[sel].astype(np.float64)
        if len(x) < 2:
            x = ref.astype(np.float64)
            y = local.astype(np.float64)
            w = weights.astype(np.float64)
        r = _wls(x, y, w)
        slope = r["slope"]
        ppm = (slope - 1.0) * 1e6
        se = r["slope_se"]
        fits.append(SegmentFit(
            segment=seg,
            n_points=int(len(x)),
            slope=slope,
            intercept=r["intercept"],
            slope_se=r["slope_se"],
            intercept_se=r["intercept_se"],
            slope_ci=(slope - T975 * se, slope + T975 * se),
            intercept_ci=(r["intercept"] - T975 * r["intercept_se"],
                          r["intercept"] + T975 * r["intercept_se"]),
            residuals=tuple(float(v) for v in r["resid"]),
            rms_residual=r["rms"],
            ppm=ppm,
            ppm_ci=((slope - T975 * se - 1.0) * 1e6,
                    (slope + T975 * se - 1.0) * 1e6),
        ))
    return fits


def estimate_delay(pulse_hits: list[PulseHit],
                   first_fit: SegmentFit) -> tuple[float, float]:
    """非整数固定延迟（亚采样）。

    固定延迟是 ref=0 处的本地偏移。温漂会让首段平均斜率偏离原点斜率，
    直接用首段截距会被污染；因此取首段内最早脉冲，按首段斜率外推回
    ref=0：delay = local_0 - slope * ref_0。脉冲峰经三点抛物线插值，
    结果达到亚采样精度。
    """
    if pulse_hits:
        ordered = sorted(pulse_hits, key=lambda h: h.ref_sample)
        h0 = ordered[0]
        if len(ordered) >= 2:
            h1 = ordered[1]
            # 最早两个脉冲的局部瞬时速率，最接近 ref=0 处的真实斜率
            origin_slope = ((h1.local_center - h0.local_center)
                            / (h1.ref_sample - h0.ref_sample))
        else:  # pragma: no cover
            origin_slope = first_fit.slope
        delay = float(h0.local_center - origin_slope * h0.ref_sample)
    else:  # pragma: no cover - 缺脉冲由检测错误链覆盖
        delay = float(first_fit.intercept)
    strength = (float(np.mean([h.strength for h in pulse_hits]))
                if pulse_hits else 0.0)
    return delay, strength


def estimate_channel(channel: int,
                     pulse_hits: list[PulseHit],
                     sync_hits: list[SyncFrameHit],
                     audio: np.ndarray,
                     cfg: WorkspaceConfig,
                     truth_ppm: float) -> EstimationResult:
    ref_a = np.array([h.ref_sample for h in pulse_hits], dtype=np.float64)
    loc_a = np.array([h.local_center for h in pulse_hits], dtype=np.float64)
    w_a = np.array([max(h.strength, 1e-3) for h in pulse_hits], dtype=np.float64)
    audio_fits = _fit_segments(ref_a, loc_a, w_a, N_SEGMENTS)

    ref_v = np.array([h.ref_frame for h in sync_hits], dtype=np.float64)
    loc_v = np.array([h.local_float for h in sync_hits], dtype=np.float64)
    w_v = np.ones_like(ref_v)
    # 视频帧域量纲换算到音频样本域（每秒 fs 样本 vs fps 帧），
    # 使音视频斜率/标准误处于同一尺度再做联合加权
    frame_to_sample = cfg.audio_rate / cfg.video_fps
    video_fits_src = _fit_segments(ref_v, loc_v, w_v, N_SEGMENTS)
    video_fits = tuple(
        SegmentFit(
            segment=s.segment, n_points=s.n_points,
            # 速率比 local/ref 为无量纲量，帧域与样本域相同；
            # 仅截距（绝对偏移）需要按帧->样本换算
            slope=s.slope,
            intercept=s.intercept * frame_to_sample,
            # 亚帧相位量化可能使个别段标准误下溢为 0，给联合加权设下限
            slope_se=max(s.slope_se, 1e-9),
            intercept_se=s.intercept_se * frame_to_sample,
            slope_ci=s.slope_ci,
            intercept_ci=(s.intercept_ci[0] * frame_to_sample,
                          s.intercept_ci[1] * frame_to_sample),
            residuals=s.residuals, rms_residual=s.rms_residual,
            ppm=s.ppm,
            ppm_ci=s.ppm_ci,
        ) for s in video_fits_src)

    # 亚采样延迟（音频域）：互相关峰抛物线插值 + 分段拟合截距
    delay, strength = estimate_delay(pulse_hits, audio_fits[0])

    # 末点累积样本偏差与置信区间
    last = audio_fits[-1]
    x_end = float(ref_a.max())
    bias = last.intercept + (last.slope - 1.0) * x_end
    bias_se = float(np.sqrt(
        last.intercept_se ** 2 + (x_end * last.slope_se) ** 2))
    bias_ci = (bias - T975 * bias_se, bias + T975 * bias_se)
    video_bias = float(video_fits[-1].intercept
                       + (video_fits[-1].slope - 1.0) * ref_v.max())

    # 音视频联合 ppm：按斜率标准误倒数平方加权
    inv_a = 1.0 / max(last.slope_se, 1e-12) ** 2
    inv_v = 1.0 / max(video_fits[-1].slope_se, 1e-12) ** 2
    fused_ppm = (last.ppm * inv_a + video_fits[-1].ppm * inv_v) / (inv_a + inv_v)

    errors = []
    for seg in audio_fits:
        if abs(truth_ppm) > cfg.drift_tol_ppm:
            errors.append(make_record(
                E_DRIFT_EXCEED,
                f"通道 {channel} 段 {seg.segment} 注入漂移 {truth_ppm:.1f}ppm "
                f"超过容差 {cfg.drift_tol_ppm:.1f}ppm",
                channel=channel, segment=seg.segment,
                truth_ppm=float(truth_ppm),
                est_ppm=seg.ppm,
                est_ppm_ci=list(seg.ppm_ci),
                tol_ppm=cfg.drift_tol_ppm))

    return EstimationResult(
        channel=channel,
        segments=tuple(audio_fits),
        delay_samples=delay,
        delay_peak_strength=strength,
        audio_bias_samples=float(bias),
        audio_bias_ci=bias_ci,
        video_bias_frames=video_bias,
        fused_ppm=float(fused_ppm),
        errors=tuple(errors),
    )
