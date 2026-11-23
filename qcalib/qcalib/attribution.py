"""反量化误差归因与列组分层敏感性分析。

误差来源用“逐阶段理想重放”的有序分解，四个分量均为非负判据数值：
  quantile      分位离散：全范围理想尺度与分位标定尺度的重构差；
  scale_quant   缩放量化：分位尺度下整数网格（含零点）引入的舍入差；
  saturation    饱和截断：裁剪到可表示区间造成的重构差；
  dequant_round 反量化舍入：反量化阶段浮点表示残差（本实现精确时为 0）。
对超差元素取分量最大者为主导来源。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from .calibration import ColumnCalib
from .config import GroupConfig
from .calibration import _clip_bounds
from .enums import qint_range
from .fingerprint import cache_key
from .kernel import ColumnRescale, _round_int

SOURCE_LABELS = ("quantile", "scale_quant", "saturation", "dequant_round")


@dataclass
class ElementAttribution:
    row_id: str
    original: float
    dequantized: float
    abs_error: float
    rel_error: Optional[float]
    tolerance_branch: str        # "relative" / "absolute-zero" / "missing"
    out_of_tolerance: bool
    dominant_source: Optional[str]
    components: Dict[str, float]
    saturated: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GroupErrorReport:
    group_id: str
    column: str
    elements: List[ElementAttribution]
    max_rel_error: Optional[float]
    saturation_ratio: float
    outlier_ratio: float
    dominant_shares: Dict[str, float]
    layer_conclusion: str
    cache_key: str
    stage: str = "error"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _ideal_full_range_scale(cfg: GroupConfig, observed: List[float]) -> float:
    """用观测到的真实最小/最大构造无分位截断的理想缩放因子。"""
    _, qhi = qint_range(cfg.qbits, cfg.scheme)
    if cfg.scheme == "symmetric":
        amplitude = max(abs(min(observed)), abs(max(observed))) or 1.0
        return amplitude / float(qhi)
    span = max(observed) - min(observed)
    if span <= 0.0:
        span = 1.0
    return span / float(qhi)


def _component_errors(
    cfg: GroupConfig,
    calib: ColumnCalib,
    ideal_scale: float,
    observed: List[float],
    original: float,
    raw_int: int,
    quantized: int,
    dequantized: float,
) -> Dict[str, float]:
    qlo, qhi = _clip_bounds(cfg.qbits, cfg.fbits, cfg.scheme)
    eff_ideal = ideal_scale
    eff_cal = calib.scale

    # 全范围理想尺度重放（理想零点使真实最小值映射到 0，对称方案零点为 0）
    if cfg.scheme == "asymmetric":
        observed_min = min(observed)
        ideal_zp = min(max(int(round(-observed_min / eff_ideal)), qlo), qhi)
    else:
        ideal_zp = 0
    d_ideal = _round_int(original / eff_ideal, cfg.rounding) + ideal_zp
    d_ideal = min(max(d_ideal, qlo), qhi)
    x_ideal = (d_ideal - ideal_zp) * eff_ideal

    # 分位尺度下、未裁剪的浮点网格重构（量化网格 + 零点）
    d_full = _round_int(original / eff_cal, cfg.rounding) + calib.zero_point
    d_full_clamped = min(max(d_full, qlo), qhi)
    x_grid = (d_full_clamped - calib.zero_point) * eff_cal

    # 最终结果（含饱和裁剪）
    exact_dequant = (quantized - calib.zero_point) * eff_cal
    d_round = abs(dequantized - exact_dequant)

    comp_quantile = abs(original - x_ideal)
    comp_scale = abs(x_ideal - x_grid)
    comp_saturation = abs(x_grid - dequantized)
    return {
        "quantile": comp_quantile,
        "scale_quant": comp_scale,
        "saturation": comp_saturation,
        "dequant_round": d_round,
    }


def analyze_column(
    cfg: GroupConfig,
    calib: ColumnCalib,
    rescale: ColumnRescale,
) -> GroupErrorReport:
    observed = [
        float(e.original)
        for e in rescale.elements
        if not e.missing and e.original is not None
    ]
    ideal_scale = _ideal_full_range_scale(cfg, observed) if observed else calib.scale

    elements: List[ElementAttribution] = []
    rel_values: List[float] = []
    n_present = 0
    n_sat = 0
    n_outlier = 0
    source_counts = {label: 0 for label in SOURCE_LABELS}

    for elem in rescale.elements:
        if elem.missing or elem.quantized is None or elem.dequantized is None:
            elements.append(
                ElementAttribution(
                    row_id=elem.row_id,
                    original=float("nan"),
                    dequantized=float("nan"),
                    abs_error=float("nan"),
                    rel_error=None,
                    tolerance_branch="missing",
                    out_of_tolerance=False,
                    dominant_source=None,
                    components={},
                    saturated=False,
                )
            )
            continue
        n_present += 1
        original = float(elem.original)
        deq = float(elem.dequantized)
        abs_err = abs(original - deq)
        if elem.saturated:
            n_sat += 1

        if original == 0.0:
            rel_err: Optional[float] = None
            branch = "absolute-zero"
            outlier = abs_err > cfg.abs_tol
        else:
            rel_err = abs_err / abs(original)
            rel_values.append(rel_err)
            branch = "relative"
            outlier = (abs_err > cfg.abs_tol) or (rel_err > cfg.rel_tol)
        if outlier:
            n_outlier += 1

        components = _component_errors(
            cfg, calib, ideal_scale, observed, original,
            elem.raw_int, elem.quantized, deq,
        )
        dominant: Optional[str] = None
        if outlier:
            dominant = max(SOURCE_LABELS, key=lambda label: (components[label], label))
            source_counts[dominant] += 1

        elements.append(
            ElementAttribution(
                row_id=elem.row_id,
                original=original,
                dequantized=deq,
                abs_error=abs_err,
                rel_error=rel_err,
                tolerance_branch=branch,
                out_of_tolerance=outlier,
                dominant_source=dominant,
                components=components,
                saturated=elem.saturated,
            )
        )

    max_rel = max(rel_values) if rel_values else None
    sat_ratio = (n_sat / n_present) if n_present else 0.0
    outlier_ratio = (n_outlier / n_present) if n_present else 0.0
    dominant_shares = {
        label: (source_counts[label] / n_outlier if n_outlier else 0.0)
        for label in SOURCE_LABELS
    }

    if not n_present:
        conclusion = "无有效元素，无法评估"
    elif outlier_ratio == 0.0:
        conclusion = f"列 {calib.column} 全部在容差内，截断比例 {sat_ratio:.3f}"
    else:
        top_source = max(SOURCE_LABELS, key=lambda label: (dominant_shares[label], label))
        conclusion = (
            f"列 {calib.column} 超差比例 {outlier_ratio:.3f}，"
            f"主导来源 {top_source}，截断比例 {sat_ratio:.3f}"
        )

    key = cache_key(
        "error",
        [rescale.cache_key, calib.cache_key],
        qbits=cfg.qbits,
        fbits=cfg.fbits,
        exchange=0,
    )
    return GroupErrorReport(
        group_id=cfg.group_id,
        column=calib.column,
        elements=elements,
        max_rel_error=max_rel,
        saturation_ratio=sat_ratio,
        outlier_ratio=outlier_ratio,
        dominant_shares=dominant_shares,
        layer_conclusion=conclusion,
        cache_key=key,
    )
