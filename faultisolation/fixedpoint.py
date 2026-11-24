"""配额预算的整数定点诊断与显式误差界。

缩放因子 ``K = 10^p``（p 取 0..3）。所有比例只做整数乘除：

    q = used * K // total          （显式向零截断）
    真实比例 r 满足 q/K <= r < (q+1)/K

* 单值量化误差界: 0 <= r - q/K < 1/K（记 ``single_error_bound = 1/K``）；
* 两比例比较判定误差界: |r1-r2| 的合成界为 2/K（各带一个截断单位）；
* 当 |q1-q2|/K 落在合成界内（即 |q1-q2| <= 2 个量化单位），
  数值比较不再可靠，按稳定决胜（量化值降序，相同则标识升序）排序并标注“受误差影响”。
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import SpecError


def validate_power(scale_power: int) -> int:
    if not isinstance(scale_power, int) or isinstance(scale_power, bool) or scale_power not in (0, 1, 2, 3):
        raise SpecError("scale_power 必须是 0/1/2/3", field="scale_power", line=1)
    return scale_power


def scaled_ratio(used: int, total: int, scale_power: int) -> int:
    """返回 used/total 的定点量化值（向零截断），全程整数运算。"""
    validate_power(scale_power)
    if total <= 0 or used < 0 or used > total:
        raise SpecError(
            f"非法比例输入 used={used}, total={total}", field="used/total", line=1)
    return (used * (10 ** scale_power)) // total


def single_error_bound(scale_power: int) -> int:
    """单值量化误差界（以量化单位表示，恒为 1 个单位 = 1/K）。"""
    validate_power(scale_power)
    return 1


def comparison_error_bound(scale_power: int) -> int:
    """两比例比较的合成误差界（量化单位，恒为 2 个单位 = 2/K）。"""
    validate_power(scale_power)
    return 2


@dataclass(frozen=True)
class RatioDiagnosis:
    label: str
    used: int
    total: int
    quantized: int
    scale_factor: int
    truncation: str
    single_error_units: int
    single_error_human: str
    comparison_error_units: int
    comparison_error_human: str

    def as_dict(self) -> dict[str, object]:
        return {
            "label": self.label, "used": self.used, "total": self.total,
            "quantized": self.quantized, "scale_factor": self.scale_factor,
            "truncation": self.truncation,
            "single_error_units": self.single_error_units,
            "single_error_bound": self.single_error_human,
            "comparison_error_units": self.comparison_error_units,
            "comparison_error_bound": self.comparison_error_human,
        }


def diagnose(label: str, used: int, total: int, scale_power: int) -> RatioDiagnosis:
    k = 10 ** validate_power(scale_power)
    q = scaled_ratio(used, total, scale_power)
    return RatioDiagnosis(
        label=label, used=used, total=total, quantized=q, scale_factor=k,
        truncation="toward_zero", single_error_units=1,
        single_error_human=f"1/{k}", comparison_error_units=2,
        comparison_error_human=f"2/{k}",
    )


@dataclass(frozen=True)
class RankedRatio:
    label: str
    quantized: int
    rank: int
    error_influenced: bool
    note: str


def rank_ratios(items: list[tuple[str, int, int]], scale_power: int) -> list[RankedRatio]:
    """items 为 (label, used, total)。返回带稳定决胜与误差标注的排序结果。"""
    diagnoses = [diagnose(label, used, total, scale_power) for label, used, total in items]
    bound = comparison_error_bound(scale_power)
    ordered = sorted(diagnoses, key=lambda d: (-d.quantized, d.label))
    results: list[RankedRatio] = []
    for index, diag in enumerate(ordered):
        influenced = False
        note = "量化值严格可分"
        if index > 0 and abs(diag.quantized - ordered[index - 1].quantized) <= bound:
            influenced = True
            note = f"与相邻比例之差在合成误差界 {bound} 个量化单位内，排序受误差影响，按标识稳定决胜"
        results.append(RankedRatio(
            label=diag.label, quantized=diag.quantized, rank=index + 1,
            error_influenced=influenced, note=note,
        ))
    return results


def global_budget(used: int, total: int, scale_power: int) -> RatioDiagnosis:
    return diagnose("__global__", used, total, scale_power)
