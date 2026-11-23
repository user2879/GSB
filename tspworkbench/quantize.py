"""定点量化：把浮点预测/真值真实转为定点整数，再计算误差指标。

切换 frac_bits 或 gear 会改变 scale，从而真实改变内部整数与指标，
不是仅修改显示格式。所有计算基于 Python int（任意精度）避免溢出。
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .config import QuantSpec
from .constants import NEAR_ZERO_THRESHOLD
from .errors import WorkbenchError, E_NON_FINITE


def quantize(value: float, quant: QuantSpec) -> int:
    if not isfinite(value):
        raise WorkbenchError(E_NON_FINITE, "待量化值非有限", context={"value": value})
    return int(round(float(value) * quant.scale))


def quantize_list(values: list[float], quant: QuantSpec) -> list[int]:
    return [quantize(v, quant) for v in values]


def dequantize(value_q: int, quant: QuantSpec) -> float:
    return value_q / quant.scale


@dataclass
class MetricResult:
    mae_q: int                 # 定点整数域上的 MAE（再放大 V1_MAE_SCALE）
    rmse_q: int                # 定点整数域上的 RMSE（再放大 V1_MAE_SCALE）
    mape_bp: int               # 基点放大整数
    near_zero_count: int
    near_zero_used: bool       # 是否触发近零绝对误差分支
    count: int

    def as_dict(self) -> dict:
        return {
            "mae_q": self.mae_q,
            "rmse_q": self.rmse_q,
            "mape_bp": self.mape_bp,
            "near_zero_count": self.near_zero_count,
            "near_zero_used": self.near_zero_used,
            "count": self.count,
        }


# 指标在整数域上的固定放大倍数（与 constants 保持一致，避免循环导入在此重声明）
MAE_SCALE = 10_000
MAPE_RATIO_SCALE = 100_000
BP_SCALE = 10_000


def compute_metrics(pred_q: list[int], actual_q: list[int], quant: QuantSpec) -> MetricResult:
    """按窗内步序升序累加，全部在定点整数上完成。"""
    if len(pred_q) != len(actual_q) or not pred_q:
        raise WorkbenchError("E_PARSE", "预测与实际长度不一致或为空",
                             context={"pred": len(pred_q), "actual": len(actual_q)})

    n = len(pred_q)
    abs_sum = 0
    sq_sum = 0
    pct_num = 0                 # 百分比分子累加（已放大）
    pct_den_count = 0
    near_zero_count = 0

    for p, a in zip(pred_q, actual_q):
        abs_err = abs(p - a)
        abs_sum += abs_err
        sq_sum += abs_err * abs_err
        if abs(a) <= NEAR_ZERO_THRESHOLD:
            near_zero_count += 1
        else:
            # |p-a|/|a| * 100；先放大 MAPE_RATIO_SCALE 再除，保持纯整数
            pct_num += (abs_err * 100 * MAPE_RATIO_SCALE) // abs(a)
            pct_den_count += 1

    mae_q = (abs_sum * MAE_SCALE) // n
    rmse_q = int(round((sq_sum / n) ** 0.5 * MAE_SCALE))

    near_zero_used = near_zero_count > 0
    if pct_den_count > 0:
        mape_bp = int(round(pct_num / pct_den_count / (MAPE_RATIO_SCALE / BP_SCALE)))
    else:
        # 全部近零：MAPE 不可定义，单独用绝对误差判定，MAPE 记 -1 哨兵
        mape_bp = -1

    return MetricResult(
        mae_q=mae_q,
        rmse_q=rmse_q,
        mape_bp=mape_bp,
        near_zero_count=near_zero_count,
        near_zero_used=near_zero_used,
        count=n,
    )
