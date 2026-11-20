"""float64 参考路径比对与逐级误差归属。

三个环节：
  accum  截断前整数累积（缓冲值 vs 精确整数和）
  trunc  截断后整数结果（截断整数 vs 精确整数和）
  deq    反量化浮点值（定点反量化 vs float64 真值）

每个超差位置只标注“首次超差环节”，同一位置不会同时归属两个环节，
并进一步区分根因：累加溢出饱和 / 边界截断 / 反量化舍入。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .kernel import PositionResult
from .models import OpDesc
from .quantize import DequantResult, TruncationResult

STAGES = ("accum", "trunc", "deq")
CAUSES = (
    "accum_overflow_saturation",
    "boundary_truncation",
    "dequant_rounding",
    "within_tolerance",
)


def _bounds(tolerance: dict, stage: str) -> tuple[float, float]:
    section = tolerance.get(stage, {})
    return float(section.get("abs", 0.0)), float(section.get("rel", 0.0))


def _rel_error(measured: float, truth: float) -> float:
    if truth == 0.0:
        return 0.0 if measured == 0.0 else float("inf")
    return abs(measured - truth) / abs(truth)


def _exceeds(abs_err: float, rel_err: float, abs_bound: float, rel_bound: float) -> bool:
    if abs_err > abs_bound:
        return True
    if rel_err == float("inf"):
        return rel_bound >= 0.0 and abs_err > 0.0
    return rel_err > rel_bound


@dataclass
class PositionAttribution:
    position: int
    stage_errors: dict
    first_fail_stage: str | None
    cause: str

    def to_dict(self) -> dict:
        return {
            "position": self.position,
            "stage_errors": self.stage_errors,
            "first_fail_stage": self.first_fail_stage,
            "cause": self.cause,
        }


@dataclass
class AttributionReport:
    positions: list[PositionAttribution] = field(default_factory=list)
    exceed_counts: dict = field(default_factory=lambda: {stage: 0 for stage in STAGES})
    max_diff: dict = field(default_factory=lambda: {stage: None for stage in STAGES})

    def add(self, item: PositionAttribution) -> None:
        self.positions.append(item)
        for stage in STAGES:
            info = item.stage_errors[stage]
            if info["exceeds"]:
                self.exceed_counts[stage] += 1
            current = self.max_diff[stage]
            if current is None or info["abs_error"] > current["abs_error"]:
                self.max_diff[stage] = {
                    "position": item.position,
                    "abs_error": info["abs_error"],
                    "rel_error": None if info["rel_error"] == float("inf") else info["rel_error"],
                    "measured": info["measured"],
                    "truth": info["truth"],
                }

    def histogram(self) -> dict:
        bins = {
            "0": 0,
            "(0,1e-6]": 0,
            "(1e-6,1e-4]": 0,
            "(1e-4,1e-2]": 0,
            "(1e-2,1]": 0,
            ">1": 0,
        }
        for item in self.positions:
            for stage in STAGES:
                value = item.stage_errors[stage]["abs_error"]
                if value == 0:
                    bins["0"] += 1
                elif value <= 1e-6:
                    bins["(0,1e-6]"] += 1
                elif value <= 1e-4:
                    bins["(1e-6,1e-4]"] += 1
                elif value <= 1e-2:
                    bins["(1e-4,1e-2]"] += 1
                elif value <= 1:
                    bins["(1e-2,1]"] += 1
                else:
                    bins[">1"] += 1
        return bins

    def to_dict(self) -> dict:
        return {
            "exceed_counts": self.exceed_counts,
            "max_diff_per_stage": self.max_diff,
            "error_histogram": self.histogram(),
            "positions": [item.to_dict() for item in self.positions],
        }


def attribute_position(
    desc: OpDesc,
    position: PositionResult,
    trunc: TruncationResult,
    deq: DequantResult,
    exact_sum: int,
    truth_deq: float,
) -> PositionAttribution:
    stage_errors: dict[str, dict] = {}

    for stage, measured, truth in (
        ("accum", float(position.buffered), float(exact_sum)),
        ("trunc", float(trunc.truncated), float(exact_sum)),
    ):
        abs_bound, rel_bound = _bounds(desc.tolerance, stage)
        abs_err = abs(measured - truth)
        rel_err = _rel_error(measured, truth)
        stage_errors[stage] = {
            "measured": measured,
            "truth": truth,
            "abs_error": abs_err,
            "rel_error": None if rel_err == float("inf") else rel_err,
            "exceeds": _exceeds(abs_err, rel_err, abs_bound, rel_bound),
        }

    abs_bound, rel_bound = _bounds(desc.tolerance, "deq")
    abs_err = abs(deq.value - truth_deq)
    rel_err = _rel_error(deq.value, truth_deq)
    stage_errors["deq"] = {
        "measured": deq.value,
        "truth": truth_deq,
        "abs_error": abs_err,
        "rel_error": None if rel_err == float("inf") else rel_err,
        "exceeds": _exceeds(abs_err, rel_err, abs_bound, rel_bound),
    }

    first_fail_stage = next((stage for stage in STAGES if stage_errors[stage]["exceeds"]), None)
    if first_fail_stage is None:
        cause = "within_tolerance"
    elif first_fail_stage == "accum":
        cause = "accum_overflow_saturation"
    elif first_fail_stage == "trunc":
        if trunc.boundary_sample:
            cause = "boundary_truncation"
        elif position.overflow is not None:
            cause = "accum_overflow_saturation"
        else:
            cause = "boundary_truncation"
    else:
        cause = "dequant_rounding"
    return PositionAttribution(
        position=position.position,
        stage_errors=stage_errors,
        first_fail_stage=first_fail_stage,
        cause=cause,
    )
