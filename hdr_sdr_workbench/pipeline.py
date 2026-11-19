"""参数化 HDR -> SDR 颜色管线（全部 float32、固定运算顺序）。

管线步骤（顺序固定，每步中间结果均为 float32）::

    线性 BT.2020 RGB
      -> 曝光补偿  * 2**stops
      -> 色域转换  3x3 矩阵（源 BT.2020 -> 目标 BT.709 / BT.2020）
      -> 色调映射  Reinhard 或 ACES 拟合（逐通道）
      -> 饱和处理  clip 到 [0, 1]
      -> OETF      sRGB / BT.709 / BT.2020 传递函数
      -> 8 位量化  floor(x*255 + 0.5)，再 clip 到 [0, 255]

矩阵口径
--------
源线性光为 BT.2020 原色（ITU-R BT.2020，D65）。RGB->XYZ 与 XYZ->RGB
由各原色 xy 色度与白点推导，BT.2020 -> BT.709 矩阵 M = XYZ709^-1 @ XYZ2020，
在模块导入时以 float64 推导一次再舍入为 float32 常量，运行期不再变化。

输出传递函数
------------
* ``srgb``   : 分段 sRGB，x<=0.0031308 用 12.92x，否则 1.055*x^(1/2.4)-0.055
* ``bt709``  : 分段 BT.709 OETF，x<0.018 用 4.5x，否则 1.099*x^0.45-0.099
* ``bt2020`` : 与 BT.709 同形的分段 OETF（10 位系统常用常数）
"""

from __future__ import annotations

import math

import numpy as np

from .errors import E_TONEMAP_PARAM, WorkbenchError

VALID_COLORSPACES = ("bt709", "bt2020")
VALID_TONEMAPS = ("reinhard", "aces")
VALID_OETF = ("srgb", "bt709", "bt2020")

# ---------------------------------------------------------------------------
# 色域矩阵（float32 常量，推导见模块 docstring）
# ---------------------------------------------------------------------------

def _rgb_to_xyz_matrix(primaries: tuple, white_xy: tuple) -> np.ndarray:
    """由原色 xy 与白点 xy 推导 RGB->XYZ（Bradford 之外的标准直接推导）。"""
    xy = np.array(primaries, dtype=np.float64)
    xyz = np.column_stack([xy[:, 0] / xy[:, 1],
                           np.ones(3),
                           (1.0 - xy[:, 0] - xy[:, 1]) / xy[:, 1]])
    w = np.array([white_xy[0] / white_xy[1], 1.0,
                  (1.0 - white_xy[0] - white_xy[1]) / white_xy[1]], dtype=np.float64)
    s = np.linalg.solve(xyz, w)
    return xyz * s


_PRIM_709 = ((0.640, 0.330), (0.300, 0.600), (0.150, 0.060))
_PRIM_2020 = ((0.708, 0.747), (0.170, 0.797), (0.131, 0.046))
_WHITE_D65 = (0.3127, 0.3290)

_XYZ_709 = _rgb_to_xyz_matrix(_PRIM_709, _WHITE_D65)
_XYZ_2020 = _rgb_to_xyz_matrix(_PRIM_2020, _WHITE_D65)
_M_2020_TO_709_F64 = np.linalg.inv(_XYZ_709) @ _XYZ_2020

M_2020_TO_709 = _M_2020_TO_709_F64.astype(np.float32)
M_IDENTITY = np.eye(3, dtype=np.float32)

# 各目标色域线性亮度系数（用于指标，float32）
LUMA_COEFFS = {
    "bt709": (np.float32(0.2126), np.float32(0.7152), np.float32(0.0722)),
    "bt2020": (np.float32(0.2627), np.float32(0.6780), np.float32(0.0593)),
}

# ACES 拟合曲线固定常数（Narkowicz 2015, "ACES Filmic Tone Mapping Curve"）
ACES_A = np.float32(2.51)
ACES_B = np.float32(0.03)
ACES_C = np.float32(2.43)
ACES_D = np.float32(0.59)
ACES_E = np.float32(0.14)


def validate_state_params(colorspace: str, exposure_stops: float,
                          tonemap: str, tonemap_param: float, oetf: str) -> None:
    if colorspace not in VALID_COLORSPACES:
        raise WorkbenchError(E_TONEMAP_PARAM, f"非法目标色域: {colorspace}")
    if tonemap not in VALID_TONEMAPS:
        raise WorkbenchError(E_TONEMAP_PARAM, f"非法色调映射算子: {tonemap}")
    if oetf not in VALID_OETF:
        raise WorkbenchError(E_TONEMAP_PARAM, f"非法输出传递函数: {oetf}")
    if not isinstance(exposure_stops, (int, float)) or not math.isfinite(float(exposure_stops)):
        raise WorkbenchError(E_TONEMAP_PARAM, f"非法曝光补偿: {exposure_stops}")
    p = float(tonemap_param)
    if not math.isfinite(p) or p <= 0.0:
        raise WorkbenchError(E_TONEMAP_PARAM,
                             f"算子参数必须为正有限值: {tonemap_param}",
                             details={"tonemap": tonemap})
    if abs(float(exposure_stops)) > 20.0:
        raise WorkbenchError(E_TONEMAP_PARAM,
                             f"曝光补偿超出 [-20, 20] stops: {exposure_stops}")


def _matmul_pixel(linear: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """固定运算顺序的逐像素 3x3 矩阵乘（显式乘加，避免 BLAS 非确定性）。"""
    r = linear[:, :, 0]
    g = linear[:, :, 1]
    b = linear[:, :, 2]
    out = np.empty_like(linear)
    for row in range(3):
        out[:, :, row] = (r * matrix[row, 0]
                          + g * matrix[row, 1]
                          + b * matrix[row, 2]).astype(np.float32)
    return out


def _reinhard(x: np.ndarray, white: float) -> np.ndarray:
    """扩展 Reinhard: x*(1 + x/w^2) / (1 + x)，white 为白场亮度参数。"""
    w2 = np.float32(white) * np.float32(white)
    numerator = x * (np.float32(1.0) + x / w2)
    denominator = np.float32(1.0) + x
    return (numerator / denominator).astype(np.float32)


def _aces(x: np.ndarray) -> np.ndarray:
    """ACES 拟合曲线 (x*(a*x+b))/(x*(c*x+d)+e)，常数固定为 Narkowicz 值。"""
    numerator = x * (ACES_A * x + ACES_B)
    denominator = x * (ACES_C * x + ACES_D) + ACES_E
    return (numerator / denominator).astype(np.float32)


def _apply_oetf(x: np.ndarray, oetf: str) -> np.ndarray:
    x = x.astype(np.float32)
    if oetf == "srgb":
        lo = x * np.float32(12.92)
        hi = np.float32(1.055) * np.power(x, np.float32(1.0 / 2.4)) - np.float32(0.055)
        return np.where(x <= np.float32(0.0031308), lo, hi).astype(np.float32)
    if oetf == "bt709":
        lo = x * np.float32(4.5)
        hi = np.float32(1.099) * np.power(x, np.float32(0.45)) - np.float32(0.099)
        return np.where(x < np.float32(0.018), lo, hi).astype(np.float32)
    # bt2020：与 bt709 同形分段（SDR 监看口径）
    lo = x * np.float32(4.5)
    hi = np.float32(1.099) * np.power(x, np.float32(0.45)) - np.float32(0.099)
    return np.where(x < np.float32(0.018), lo, hi).astype(np.float32)


def run_pipeline(linear: np.ndarray, colorspace: str, exposure_stops: float,
                 tonemap: str, tonemap_param: float, oetf: str
                 ) -> tuple[np.ndarray, np.ndarray]:
    """执行完整管线，返回 (8 位 uint8 HxWx3, 编码前 float32 HxWx3)。"""
    validate_state_params(colorspace, exposure_stops, tonemap, tonemap_param, oetf)
    x = linear.astype(np.float32, copy=True)

    # 1) 曝光补偿
    gain = np.float32(2.0 ** float(exposure_stops))
    x = (x * gain).astype(np.float32)

    # 2) 色域转换（源 BT.2020 -> 目标）
    matrix = M_IDENTITY if colorspace == "bt2020" else M_2020_TO_709
    x = _matmul_pixel(x, matrix)

    # 3) 色调映射（逐通道，负值先归零）
    x = np.maximum(x, np.float32(0.0)).astype(np.float32)
    if tonemap == "reinhard":
        x = _reinhard(x, float(tonemap_param))
    else:
        x = _aces(x)

    # 4) 饱和处理：固定 clip [0,1]
    x = np.clip(x, np.float32(0.0), np.float32(1.0)).astype(np.float32)

    # 5) OETF 编码
    encoded = _apply_oetf(x, oetf)
    encoded = np.clip(encoded, np.float32(0.0), np.float32(1.0)).astype(np.float32)

    # 6) 8 位量化：floor(x*255 + 0.5)（half-up），再 clip
    q = np.floor(encoded * np.float32(255.0) + np.float32(0.5))
    u8 = np.clip(q, 0, 255).astype(np.uint8)
    return u8, x


def frame_metrics(linear: np.ndarray, mapped: np.ndarray, colorspace: str) -> dict:
    """计算色彩指标。linear=曝光/色域转换后、tonemap 前；mapped=tonemap 后。"""
    coeffs = LUMA_COEFFS[colorspace]
    y_lin = (linear[:, :, 0] * coeffs[0]
             + linear[:, :, 1] * coeffs[1]
             + linear[:, :, 2] * coeffs[2]).astype(np.float32)
    y_map = (mapped[:, :, 0] * coeffs[0]
             + mapped[:, :, 1] * coeffs[1]
             + mapped[:, :, 2] * coeffs[2]).astype(np.float32)
    clipped = np.count_nonzero(mapped >= np.float32(1.0))
    total = mapped.shape[0] * mapped.shape[1] * 3
    return {
        "linear_peak_y": float(y_lin.max()),
        "linear_mean_y": float(y_lin.astype(np.float64).sum() / np.float64(y_lin.size)),
        "mapped_peak_y": float(y_map.max()),
        "mapped_mean_y": float(y_map.astype(np.float64).sum() / np.float64(y_map.size)),
        "clipped_channel_ratio": float(np.float64(clipped) / np.float64(total)),
    }
