"""确定性 HDR 线性光帧序列合成。

固定种子（默认 20240517）的 numpy ``default_rng`` PCG64 生成器，按固定
调用顺序产出 60 帧 1920x1080 的线性光 RGB 图像，落盘为 16 位半浮点
（uint16 位模式，小端 raw，每帧 ``width*height*2`` 字节）。

每帧包含三类区域：
  1. 深阴影基底：很低的线性光（约 0.001–0.03），含固定空间图样；
  2. 彩色渐变横幅：横跨画面中部的 BT.2020 基色/补色线性渐变；
  3. 高光窗口：随帧平移、亮度可达数十倍 SDR 参考白的矩形窗口。

真值清单按 BT.2020 线性亮度系数
Y = 0.2627*R + 0.6780*G + 0.0593*B 计算峰值/平均亮度。
"""

from __future__ import annotations

import os
import zlib
from dataclasses import dataclass

import numpy as np

from .errors import E_FRAME_FORMAT, WorkbenchError
from .halffloat import float32_to_half_bytes, half_bytes_to_float32

DEFAULT_SEED = 20240517
DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080
DEFAULT_FRAME_COUNT = 60

# BT.2020 线性亮度系数（float32 常量）
LUMA_BT2020 = (np.float32(0.2627), np.float32(0.6780), np.float32(0.0593))
FRAME_SUFFIX = ".half.z"


@dataclass(frozen=True)
class SequenceConfig:
    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    frame_count: int = DEFAULT_FRAME_COUNT
    seed: int = DEFAULT_SEED

    def validate(self) -> None:
        if not (isinstance(self.width, int) and isinstance(self.height, int)
                and isinstance(self.frame_count, int)):
            raise WorkbenchError(E_FRAME_FORMAT, "width/height/frame_count 必须为整数")
        if self.width <= 0 or self.height <= 0 or self.frame_count <= 0:
            raise WorkbenchError(E_FRAME_FORMAT,
                                 f"非法帧维度: {self.width}x{self.height}x{self.frame_count}")
        if self.width > 7680 or self.height > 4320:
            raise WorkbenchError(E_FRAME_FORMAT,
                                 f"帧维度超出支持上限: {self.width}x{self.height}")
        if not isinstance(self.seed, int) or not (0 <= self.seed < 2**32):
            raise WorkbenchError(E_FRAME_FORMAT, f"非法随机种子: {self.seed}")

    def to_dict(self) -> dict:
        return {"width": self.width, "height": self.height,
                "frame_count": self.frame_count, "seed": self.seed}


def frame_path(frames_dir: str, index: int) -> str:
    return os.path.join(frames_dir, f"frame_{index:04d}{FRAME_SUFFIX}")


def write_half_frame(path: str, half_bits: np.ndarray) -> None:
    """原子写入单帧 half raw（zlib 压缩；压缩是字节级无损的）。"""
    if half_bits.dtype != np.uint16 or half_bits.ndim != 3 or half_bits.shape[2] != 3:
        raise WorkbenchError(E_FRAME_FORMAT, "half 帧必须为 HxWx3 uint16 数组")
    packed = half_bits.astype("<u2").tobytes()
    blob = zlib.compress(packed, level=6)
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(blob)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def read_half_frame(path: str, width: int, height: int) -> np.ndarray:
    """读取单帧并解码为 float32 HxWx3 线性光数组。"""
    with open(path, "rb") as fh:
        try:
            blob = zlib.decompress(fh.read())
        except zlib.error as exc:
            raise WorkbenchError(
                E_FRAME_FORMAT, f"帧压缩数据损坏: {os.path.basename(path)}",
                details={"path": path}) from exc
    expected = width * height * 3 * 2
    if len(blob) != expected:
        raise WorkbenchError(
            E_FRAME_FORMAT,
            f"帧字节数不符: 期望 {expected}, 实际 {len(blob)} ({os.path.basename(path)})",
            details={"expected_bytes": expected, "actual_bytes": len(blob)},
        )
    bits = np.frombuffer(blob, dtype="<u2").reshape(height, width, 3).copy()
    return half_bytes_to_float32(bits)


def _luma_stats(linear: np.ndarray) -> tuple[float, float]:
    """固定运算顺序的 BT.2020 亮度统计，返回 (峰值, 平均) Python float。"""
    r = linear[:, :, 0].astype(np.float32)
    g = linear[:, :, 1].astype(np.float32)
    b = linear[:, :, 2].astype(np.float32)
    y = r * LUMA_BT2020[0] + g * LUMA_BT2020[1] + b * LUMA_BT2020[2]
    peak = float(y.max())
    # numpy sum 使用成对求和，固定版本下确定性；为跨平台稳定转 float64 累加口径
    mean = float(y.astype(np.float64).sum() / np.float64(y.size))
    return peak, mean


def _synthetic_frame(rng: np.random.Generator, cfg: SequenceConfig,
                     index: int, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    h, w = cfg.height, cfg.width
    # 1) 深阴影基底：固定低频图样 + 逐像素确定性微噪点
    base = (np.float32(0.004)
            + np.float32(0.006) * np.sin(xs * np.float32(6.2831853)
                                         / np.float32(max(w, 1))).astype(np.float32)
            + np.float32(0.003) * np.cos(ys * np.float32(6.2831853)
                                         / np.float32(max(h, 1))).astype(np.float32))
    noise = rng.random((h, w), dtype=np.float32) * np.float32(0.002)
    shade = (base + noise).astype(np.float32)
    frame = np.empty((h, w, 3), dtype=np.float32)
    frame[:, :, 0] = shade
    frame[:, :, 1] = shade * np.float32(0.9)
    frame[:, :, 2] = shade * np.float32(0.8)

    # 2) 彩色渐变横幅：纵向 34%–58%，R->G->B 线性渐变（BT.2020 基色量级）
    y0, y1 = int(h * 0.34), int(h * 0.58)
    t = (np.arange(w, dtype=np.float32) / np.float32(max(w - 1, 1)))
    ramp = np.empty((w, 3), dtype=np.float32)
    third = np.float32(1.0 / 3.0)
    two_third = np.float32(2.0 / 3.0)
    ramp[:, 0] = np.clip(np.float32(1.5) * (np.float32(1.0) - t / two_third), 0, 1).astype(np.float32)
    green = np.where(t < third, t / third,
                     np.float32(1.0) - (t - third) / third)
    ramp[:, 1] = np.clip(np.float32(1.5) * green, 0, 1).astype(np.float32)
    ramp[:, 2] = np.clip(np.float32(1.5) * np.maximum(t - third, np.float32(0.0))
                         / two_third, 0, 1).astype(np.float32)
    frame[y0:y1, :, :] = np.maximum(frame[y0:y1, :, :], ramp[np.newaxis, :, :])

    # 3) 高光窗口：随帧水平往返移动，亮度 8–40 倍参考白
    win_w = max(w // 8, 2)
    win_h = max(h // 8, 2)
    travel = w - win_w
    step = (index * 2) % max(2 * travel, 1)
    x0 = step if step <= travel else 2 * travel - step
    x0 = int(x0)
    y0w = int(h * 0.68)
    intensity = np.float32(8.0 + 32.0 * (0.5 + 0.5 * np.sin(
        np.float32(index) * np.float32(0.314159265))))
    win_color = np.array([1.0, 0.85, 0.6], dtype=np.float32) * intensity
    frame[y0w:y0w + win_h, x0:x0 + win_w, :] = np.maximum(
        frame[y0w:y0w + win_h, x0:x0 + win_w, :],
        win_color[np.newaxis, np.newaxis, :])

    # 防御性裁剪到 half 有限范围，再做 RNE half 量化往返
    frame = np.clip(frame, np.float32(0.0), np.float32(65504.0)).astype(np.float32)
    half_bits = float32_to_half_bytes(frame)
    return half_bits


def generate_sequence(work_dir: str, cfg: SequenceConfig | None = None) -> dict:
    """生成完整帧序列与真值清单，返回清单 dict。可安全重复调用（覆盖式原子写）。"""
    cfg = cfg or SequenceConfig()
    cfg.validate()
    frames_dir = os.path.join(work_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    h, w = cfg.height, cfg.width
    ys = np.repeat(np.arange(h, dtype=np.float32)[:, np.newaxis], w, axis=1)
    xs = np.repeat(np.arange(w, dtype=np.float32)[np.newaxis, :], h, axis=0)

    rng = np.random.default_rng(cfg.seed)
    truth = []
    for index in range(cfg.frame_count):
        half_bits = _synthetic_frame(rng, cfg, index, xs, ys)
        write_half_frame(frame_path(frames_dir, index), half_bits)
        linear = half_bytes_to_float32(half_bits)
        peak, mean = _luma_stats(linear)
        truth.append({
            "frame": index,
            "peak_luminance": peak,
            "mean_luminance": mean,
            "source_bytes": os.path.basename(frame_path(frames_dir, index)),
        })

    manifest = {
        "config": cfg.to_dict(),
        "half_format": "IEEE754 binary16, uint16 little-endian bit patterns, zlib(raw)",
        "rounding": "round-to-nearest-even; Inf saturates to +/-65504; NaN maps to 0",
        "luma_coefficients_bt2020": [0.2627, 0.6780, 0.0593],
        "frames": truth,
    }
    return manifest
