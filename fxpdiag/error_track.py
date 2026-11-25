"""逐级误差传播追踪、容差越界定位与下游结论翻转归因。"""

from __future__ import annotations

from dataclasses import dataclass

STAGES = ("row_stats", "softmax_norm", "ln_scale_bias", "cross_layer")


@dataclass
class StageError:
    stage: str
    max_abs_error: float
    max_rel_error: float | None      # 真值为零时为 None，走绝对误差分支
    zero_truth_branch: bool
    exceeded: bool
    first_exceed: bool = False

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "max_abs_error": self.max_abs_error,
            "max_rel_error": self.max_rel_error,
            "zero_truth_branch": self.zero_truth_branch,
            "exceeded": self.exceeded,
            "first_exceed": self.first_exceed,
        }


def _element_errors(approx: list[float], truth: list[float]) -> tuple[float, float | None, bool]:
    max_abs = 0.0
    max_rel: float | None = None
    zero_branch = False
    for approx_value, truth_value in zip(approx, truth):
        abs_error = abs(approx_value - truth_value)
        max_abs = max(max_abs, abs_error)
        if truth_value == 0.0:
            zero_branch = True
            continue
        rel = abs_error / abs(truth_value)
        max_rel = rel if max_rel is None else max(max_rel, rel)
    return max_abs, max_rel, zero_branch


def build_stage_error(
    stage: str,
    approx: list[float],
    truth: list[float],
    abs_tol: float,
    rel_tol: float,
) -> StageError:
    max_abs, max_rel, zero_branch = _element_errors(approx, truth)
    # 真值全为零：无法定义相对误差，改用绝对误差单独判定并标注该分支
    if all(value == 0.0 for value in truth):
        exceeded = max_abs > abs_tol
        zero_branch = True
    elif max_rel is None:
        exceeded = max_abs > abs_tol
    else:
        exceeded = (max_rel > rel_tol) or (max_abs > abs_tol)
    return StageError(stage, max_abs, max_rel, zero_branch, exceeded)


def first_exceeded_stage(stage_errors: list[StageError]) -> str | None:
    for index, item in enumerate(stage_errors):
        if item.exceeded:
            item.first_exceed = True
            return item.stage
    return None


def attribute_flip(
    degenerate: bool,
    rounding_flipped: bool,
    cross_layer_inconsistent: bool,
) -> str:
    """翻转归因，优先级固定：方差退化 > 量化舍入 > 跨层累计。"""

    if degenerate:
        return "VARIANCE_DEGENERATE"
    if rounding_flipped:
        return "QUANT_ROUNDING"
    if cross_layer_inconsistent:
        return "CROSS_LAYER_ACCUM"
    return "NONE"


@dataclass
class RowDiagnosis:
    layer: str
    row_index: int
    version: str
    stage_errors: list[StageError]
    first_exceed_stage: str | None
    flip: str
    truth_argmax: int
    fixed_argmax: int

    def to_dict(self) -> dict:
        return {
            "layer": self.layer,
            "row_index": self.row_index,
            "version": self.version,
            "stage_errors": [item.to_dict() for item in self.stage_errors],
            "first_exceed_stage": self.first_exceed_stage,
            "flip": self.flip,
            "truth_argmax": self.truth_argmax,
            "fixed_argmax": self.fixed_argmax,
        }


def compare_tiers(
    coarse_argmax: int,
    fine_argmax: int,
    degenerate: bool,
    cross_layer_inconsistent: bool,
) -> str:
    """两档位下下游结论（argmax）不一致时的归因。"""

    if coarse_argmax == fine_argmax:
        return "NONE"
    rounding_flipped = coarse_argmax != fine_argmax and not degenerate
    return attribute_flip(degenerate, rounding_flipped, cross_layer_inconsistent)
