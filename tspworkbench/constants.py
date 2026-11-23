"""全局枚举、码表与 v1 定宽布局定义。

定宽设计：每个 v1 文本行以 2 字符类型标签起始，随后按固定字符宽度切片，
解析器严格校验总长度；长度不符即判定列错位 -> E_FIELD_LAYOUT。
"""
from __future__ import annotations

# ---- 模型枚举（v2 命名模型）与 v1 整数模型码互转码表 ----
MODELS = ("naive", "moving_average", "autoregressive")
MODEL_TO_CODE = {name: idx + 1 for idx, name in enumerate(MODELS)}
CODE_TO_MODEL = {idx + 1: name for idx, name in enumerate(MODELS)}

# ---- 命名缺失处理方式枚举 ----
MISSING_STRATEGIES = ("forward_fill", "mean_fill", "zero_fill")

# ---- 交换版本号（参与缓存键派生） ----
EXCHANGE_VERSIONS = ("v1", "v2")

# ---- 量化允许档位；gear 是小数位宽的离散档位 ----
QUANT_GEARS = (0, 1, 2, 3, 4)

# ---- v1 定宽字段布局：字段名 -> 字符宽度 ----
V1_LAYOUT: dict[str, dict[str, int]] = {
    "HD": {"tag": 2, "series_id": 16, "window_index": 8, "ex_version": 16},
    "MW": {"tag": 2, "series_id": 16, "window_index": 8, "model_code": 4,
           "weight_permille": 8, "lag_order": 6},
    "PR": {"tag": 2, "series_id": 16, "window_index": 8, "model_code": 4,
           "step_index": 6, "pred_q": 18, "actual_q": 18},
    "MT": {"tag": 2, "series_id": 16, "window_index": 8,
           "mae_q": 18, "rmse_q": 18, "mape_bp": 14, "near_zero": 3},
}

# v1 指标在定点整数之上的固定放大倍数
V1_MAE_SCALE = 10_000          # MAE/RMSE：定点值的 1e-4 整数
V1_MAPE_SCALE = 100_000        # MAPE：基点（万分之一）的 1e5 整数 => 原始比率的 1e-10

# 近零判定阈值（量化后实际定点整数绝对值 <= 该值即走绝对误差分支）
NEAR_ZERO_THRESHOLD = 0


def v1_slices(tag: str) -> list[tuple[str, int, int]]:
    """返回某行类型的 (字段名, 起, 止) 切片表。"""
    widths = V1_LAYOUT[tag]
    out: list[tuple[str, int, int]] = []
    cursor = 0
    for name, width in widths.items():
        out.append((name, cursor, cursor + width))
        cursor += width
    return out


def v1_line_length(tag: str) -> int:
    return sum(V1_LAYOUT[tag].values())
