"""反量化与逐元素误差归因 (业务要求 4).

误差来源与贡献定义 (绝对值域, 单位同原浮点张量):
- saturation          饱和截断: 元素超出可表示浮点区间的距离;
- scale_quantization  缩放量化: 缩放因子定点化引入的误差 |(q-zp)| * |scale - scale_q|;
- quantile_discrete   分位离散: 元素超出分位标定区间 [q_lo, q_hi] 的距离;
- dequant_rounding    反量化舍入: 量化取整引入的残差 |x/scale_q - round(x/scale_q)| * scale_q.
决断: 取贡献最大者; 并列时按固定优先级
saturation > scale_quantization > quantile_discrete > dequant_rounding.
"""

from __future__ import annotations

SOURCE_SATURATION = "saturation"
SOURCE_SCALE_QUANT = "scale_quantization"
SOURCE_QUANTILE = "quantile_discrete"
SOURCE_DEQUANT_ROUND = "dequant_rounding"

# 并列时固定优先级 (排前者优先)
SOURCE_PRIORITY = (
    SOURCE_SATURATION,
    SOURCE_SCALE_QUANT,
    SOURCE_QUANTILE,
    SOURCE_DEQUANT_ROUND,
)

BRANCH_ZERO_TRUTH = "zero_truth_abs"
BRANCH_NORMAL = "abs_and_rel"
BRANCH_MASKED = "masked_saturated"


def dequantize_value(q: int, zero_point: int, scale_q: float) -> float:
    return (q - zero_point) * scale_q


def attribute_element(
    x: float,
    q: int,
    clip_dir: int,
    zero_point: int,
    scale: float,
    scale_q: float,
    q_lo: float,
    q_hi: float,
    qmin: int,
    qmax: int,
) -> dict:
    """对单个元素计算四类来源贡献并给出决断."""
    lo_f = (qmin - zero_point) * scale_q
    hi_f = (qmax - zero_point) * scale_q
    contrib_sat = abs(x - min(max(x, lo_f), hi_f)) if clip_dir != 0 else 0.0
    contrib_scale = abs(q - zero_point) * abs(scale - scale_q)
    if x < q_lo:
        contrib_quantile = q_lo - x
    elif x > q_hi:
        contrib_quantile = x - q_hi
    else:
        contrib_quantile = 0.0
    if clip_dir != 0:
        contrib_round = 0.0
    else:
        exact = x / scale_q
        contrib_round = abs(exact - round(exact)) * scale_q
    contributions = {
        SOURCE_SATURATION: float(contrib_sat),
        SOURCE_SCALE_QUANT: float(contrib_scale),
        SOURCE_QUANTILE: float(contrib_quantile),
        SOURCE_DEQUANT_ROUND: float(contrib_round),
    }
    best = max(
        SOURCE_PRIORITY,
        key=lambda s: (contributions[s], -SOURCE_PRIORITY.index(s)),
    )
    basis = (
        f"max_contribution={best}:{contributions[best]:.6g}; "
        f"tie_priority={'>'.join(SOURCE_PRIORITY)}"
    )
    return {"source": best, "contributions": contributions, "basis": basis}


def dequant_column(
    col_values,
    col_missing,
    quant: dict,
    scale: float,
    zero_point: int,
    q_lo: float,
    q_hi: float,
    abs_tol: float,
    rel_tol: float,
) -> dict:
    """反量化 + 误差计算 + 超差判定 + 归因."""
    frac_bits = quant["frac_bits"]
    scale_q = quant["scale_fp"] / (1 << frac_bits)
    qmin, qmax = quant["qmin"], quant["qmax"]
    saturation = quant["saturation"]
    n = len(col_values)
    abs_err: list[float | None] = []
    rel_err: list[float | None] = []
    oor: list[bool | None] = []
    branch: list[str | None] = []
    attributions: list[dict | None] = []
    oor_count = 0
    for i in range(n):
        if col_missing[i]:
            abs_err.append(None)
            rel_err.append(None)
            oor.append(None)
            branch.append(None)
            attributions.append(None)
            continue
        x = float(col_values[i])
        q = quant["q"][i]
        if saturation == "mask" and quant["clip_dir"][i] != 0:
            # mask 策略: 截断元素反量化无效, 直接判超差并归因饱和截断
            abs_err.append(None)
            rel_err.append(None)
            oor.append(True)
            branch.append(BRANCH_MASKED)
            oor_count += 1
            attributions.append(
                {
                    "source": SOURCE_SATURATION,
                    "contributions": {
                        SOURCE_SATURATION: 1e308,
                        SOURCE_SCALE_QUANT: 0.0,
                        SOURCE_QUANTILE: 0.0,
                        SOURCE_DEQUANT_ROUND: 0.0,
                    },
                    "basis": "mask_strategy_saturated_element",
                }
            )
            continue
        x_hat = dequantize_value(q, zero_point, scale_q)
        ae = abs(x_hat - x)
        if x == 0.0:
            re = None
            is_oor = ae > abs_tol
            br = BRANCH_ZERO_TRUTH
        else:
            re = ae / abs(x)
            is_oor = ae > abs_tol and re > rel_tol
            br = BRANCH_NORMAL
        abs_err.append(ae)
        rel_err.append(re)
        oor.append(bool(is_oor))
        branch.append(br)
        if is_oor:
            oor_count += 1
            attributions.append(
                attribute_element(
                    x=x,
                    q=q,
                    clip_dir=quant["clip_dir"][i],
                    zero_point=zero_point,
                    scale=scale,
                    scale_q=scale_q,
                    q_lo=q_lo,
                    q_hi=q_hi,
                    qmin=qmin,
                    qmax=qmax,
                )
            )
        else:
            attributions.append(None)
    valid_rel = [r for r, m in zip(rel_err, col_missing) if not m and r is not None]
    valid_abs = [a for a, m in zip(abs_err, col_missing) if not m and a is not None]
    return {
        "abs_err": abs_err,
        "rel_err": rel_err,
        "oor": oor,
        "branch": branch,
        "attributions": attributions,
        "oor_count": oor_count,
        "max_abs_err": max(valid_abs) if valid_abs else 0.0,
        "max_rel_err": max(valid_rel) if valid_rel else 0.0,
        "scale_q": float(scale_q),
    }
