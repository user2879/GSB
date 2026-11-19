"""采集参数与默认配置。

参数分两层：
- 全局参数：参考通道、漂移容差、空洞阈值、预算等（GLOBAL 键）；
- 逐通道参数：漂移模型开关、延迟补偿开关、空洞填补策略（CHANNEL 键）。

真值（ppm/温漂/非整数延迟/空洞区间）由本配置固定，素材生成器据此注入，
清单中同时写入真值供估计模块配对与评分。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .hashing import hash_json

# --- 依赖图边所标注的参数键 -----------------------------------------------------
PARAM_DRIFT_MODEL = "drift_model"        # 漂移模型（线性/温漂）
PARAM_DELAY_COMP = "delay_compensation"  # 非整数延迟补偿开关
PARAM_GAP_POLICY = "gap_policy"          # 空洞填补策略
PARAM_REFERENCE = "reference_channel"    # 参考通道

CHANNEL_PARAM_KEYS = (PARAM_DRIFT_MODEL, PARAM_DELAY_COMP, PARAM_GAP_POLICY)
GLOBAL_PARAM_KEYS = (PARAM_REFERENCE,)
ALL_PARAM_KEYS = CHANNEL_PARAM_KEYS + GLOBAL_PARAM_KEYS

# 空洞策略
GAP_LINEAR = "linear_interp"     # 线性插值
GAP_DRIFT = "drift_extrapolate"  # 按漂移模型外推
GAP_MARK = "mark_unfillable"     # 标记为不可填补
GAP_POLICIES = (GAP_LINEAR, GAP_DRIFT, GAP_MARK)

# 漂移模型
DRIFT_LINEAR = "linear"
DRIFT_THERMAL = "thermal"
DRIFT_MODELS = (DRIFT_LINEAR, DRIFT_THERMAL)

# 素材物理常量
AUDIO_RATE = 48000          # 48 kHz
VIDEO_FPS = 30              # 30 fps
FRAME_W = 320
FRAME_H = 180
DURATION_SEC = 8.0
PULSE_PERIOD_SEC = 0.8      # 同步脉冲周期
PULSE_HALF_WIDTH = 2        # 音频脉冲半宽（样本）

# 默认容差/阈值（全部为声明值，超限即返回业务码）
DEFAULT_DRIFT_TOL_PPM = 1000.0
DEFAULT_DELAY_TOL_SAMPLES = 2.0
DEFAULT_AUDIO_GAP_MAX = 240     # 5 ms @48k
DEFAULT_VIDEO_GAP_MAX = 5       # 帧
DEFAULT_AV_SYNC_TOL_SEC = 0.0025
DEFAULT_TILE_BUDGET_MS = 5.0
T975 = 3.186  # 小样本 95% t 临界值（保守常量，分段样本数固定）


@dataclass(frozen=True)
class GapRange:
    """丢样本/丢帧真值区间：[start, end)，单位为本地样本或本地帧号。"""

    start: int
    end: int
    domain: str  # "audio" | "video"

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class ChannelTruth:
    """逐通道注入真值。"""

    channel: int
    ppm: float                 # 恒定晶振漂移 ppm
    thermal_ppm: float         # 温漂峰峰幅度 ppm
    thermal_period_sec: float  # 温漂周期
    delay_samples: float       # 非整数采样固定延迟（音频样本单位）
    drift_model: str
    delay_compensation: bool
    gap_policy: str
    gaps: tuple[GapRange, ...]


@dataclass(frozen=True)
class ChannelParams:
    """现场可增量修改的逐通道参数（真值不可改，模型开关/策略可改）。"""

    channel: int
    drift_model: str
    delay_compensation: bool
    gap_policy: str


@dataclass
class WorkspaceConfig:
    """完整工作台配置：素材真值 + 可调参数 + 声明容差。"""

    seed: int = 20261120
    audio_rate: int = AUDIO_RATE
    video_fps: int = VIDEO_FPS
    frame_w: int = FRAME_W
    frame_h: int = FRAME_H
    duration_sec: float = DURATION_SEC
    pulse_period_sec: float = PULSE_PERIOD_SEC
    drift_tol_ppm: float = DEFAULT_DRIFT_TOL_PPM
    delay_tol_samples: float = DEFAULT_DELAY_TOL_SAMPLES
    audio_gap_max: int = DEFAULT_AUDIO_GAP_MAX
    video_gap_max: int = DEFAULT_VIDEO_GAP_MAX
    av_sync_tol_sec: float = DEFAULT_AV_SYNC_TOL_SEC
    tile_budget_ms: float = DEFAULT_TILE_BUDGET_MS
    reference_channel: int = 0
    channel_params: tuple[ChannelParams, ...] = field(default_factory=tuple)
    truths: tuple[ChannelTruth, ...] = field(default_factory=tuple)

    def param_version(self) -> str:
        """参数版本戳：仅对可调参数（含参考通道）求哈希。"""
        material = {
            "reference_channel": self.reference_channel,
            "channels": [asdict(p) for p in self.channel_params],
        }
        return hash_json(material)[:16]

    def channel_param(self, channel: int) -> ChannelParams:
        return self.channel_params[channel]

    def truth(self, channel: int) -> ChannelTruth:
        return self.truths[channel]

    def with_channel_param(self, channel: int, key: str, value: Any) -> "WorkspaceConfig":
        """返回修改单个参数后的新配置（不可变风格）。"""
        params = list(self.channel_params)
        old = params[channel]
        data = asdict(old)
        data[key] = value
        params[channel] = ChannelParams(**data)
        return WorkspaceConfig(
            seed=self.seed,
            audio_rate=self.audio_rate,
            video_fps=self.video_fps,
            frame_w=self.frame_w,
            frame_h=self.frame_h,
            duration_sec=self.duration_sec,
            pulse_period_sec=self.pulse_period_sec,
            drift_tol_ppm=self.drift_tol_ppm,
            audio_gap_max=self.audio_gap_max,
            video_gap_max=self.video_gap_max,
            av_sync_tol_sec=self.av_sync_tol_sec,
            tile_budget_ms=self.tile_budget_ms,
            reference_channel=self.reference_channel,
            channel_params=tuple(params),
            truths=self.truths,
        )

    def with_reference(self, channel: int) -> "WorkspaceConfig":
        return WorkspaceConfig(
            seed=self.seed,
            audio_rate=self.audio_rate,
            video_fps=self.video_fps,
            frame_w=self.frame_w,
            frame_h=self.frame_h,
            duration_sec=self.duration_sec,
            pulse_period_sec=self.pulse_period_sec,
            drift_tol_ppm=self.drift_tol_ppm,
            audio_gap_max=self.audio_gap_max,
            video_gap_max=self.video_gap_max,
            av_sync_tol_sec=self.av_sync_tol_sec,
            tile_budget_ms=self.tile_budget_ms,
            reference_channel=channel,
            channel_params=self.channel_params,
            truths=self.truths,
        )

    def snapshot(self) -> dict[str, Any]:
        """可调参数快照（真值不参与，素材不变）。"""
        return {
            "reference_channel": self.reference_channel,
            "channel_params": [asdict(p) for p in self.channel_params],
            "drift_tol_ppm": self.drift_tol_ppm,
            "audio_gap_max": self.audio_gap_max,
            "video_gap_max": self.video_gap_max,
            "av_sync_tol_sec": self.av_sync_tol_sec,
            "tile_budget_ms": self.tile_budget_ms,
        }

    def with_snapshot(self, snap: dict[str, Any]) -> "WorkspaceConfig":
        params = tuple(ChannelParams(**p) for p in snap["channel_params"])
        return WorkspaceConfig(
            seed=self.seed,
            audio_rate=self.audio_rate,
            video_fps=self.video_fps,
            frame_w=self.frame_w,
            frame_h=self.frame_h,
            duration_sec=self.duration_sec,
            pulse_period_sec=self.pulse_period_sec,
            drift_tol_ppm=snap["drift_tol_ppm"],
            audio_gap_max=snap["audio_gap_max"],
            video_gap_max=snap["video_gap_max"],
            av_sync_tol_sec=snap["av_sync_tol_sec"],
            tile_budget_ms=snap["tile_budget_ms"],
            reference_channel=snap["reference_channel"],
            channel_params=params,
            truths=self.truths,
        )


def _default_truths() -> tuple[ChannelTruth, ...]:
    # 4 路：低漂移线性、高漂移线性（超 1000ppm 容差）、温漂变体、温漂+大延迟
    return (
        ChannelTruth(
            channel=0, ppm=120.0, thermal_ppm=0.0, thermal_period_sec=6.0,
            delay_samples=7.35, drift_model=DRIFT_LINEAR,
            delay_compensation=True, gap_policy=GAP_LINEAR,
            gaps=(GapRange(38500, 38560, "audio"), GapRange(52, 54, "video")),
        ),
        ChannelTruth(
            channel=1, ppm=1350.0, thermal_ppm=0.0, thermal_period_sec=6.0,
            delay_samples=-11.72, drift_model=DRIFT_LINEAR,
            delay_compensation=True, gap_policy=GAP_DRIFT,
            # 含一个超出 240 样本阈值的音频空洞（触发 E_GAP_TOO_LONG）
            gaps=(GapRange(115050, 115350, "audio"), GapRange(120, 123, "video")),
        ),
        ChannelTruth(
            channel=2, ppm=260.0, thermal_ppm=180.0, thermal_period_sec=5.0,
            delay_samples=3.88, drift_model=DRIFT_THERMAL,
            delay_compensation=False, gap_policy=GAP_MARK,
            gaps=(GapRange(153500, 153590, "audio"), GapRange(150, 152, "video")),
        ),
        ChannelTruth(
            channel=3, ppm=420.0, thermal_ppm=240.0, thermal_period_sec=4.0,
            delay_samples=15.27, drift_model=DRIFT_THERMAL,
            delay_compensation=True, gap_policy=GAP_LINEAR,
            gaps=(GapRange(268600, 268660, "audio"), GapRange(200, 202, "video")),
        ),
    )


def default_config(seed: int = 20261120) -> WorkspaceConfig:
    truths = _default_truths()
    params = tuple(
        ChannelParams(
            channel=t.channel,
            drift_model=t.drift_model,
            delay_compensation=t.delay_compensation,
            gap_policy=t.gap_policy,
        )
        for t in truths
    )
    return WorkspaceConfig(seed=seed, channel_params=params, truths=truths)
