"""领域数据模型与交换格式码表。"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum

# ---------------------------------------------------------------------- #
# v1 定宽文本行字段规格（总宽 64）
# ---------------------------------------------------------------------- #
V1_FIELD_WIDTHS = {
    "layer": 24,
    "row": 6,
    "scale_permille": 12,
    "bias_code": 12,
    "bit_width": 10,
}
V1_LINE_WIDTH = sum(V1_FIELD_WIDTHS.values())
V1_SCALE_MIN, V1_SCALE_MAX = 0, 9_999_999
V1_BIAS_MIN, V1_BIAS_MAX = -9_999_999, 9_999_999

# ---------------------------------------------------------------------- #
# 量化方案 / 舍入模式 / 位宽码表
# ---------------------------------------------------------------------- #
class Scheme(str, Enum):
    Q4_3_NORM = "Q4.3_NORM"        # 8 位，启用偏置
    Q8_8_NORM = "Q8.8_NORM"        # 16 位，启用偏置
    Q16_16_NORM = "Q16.16_NORM"    # 32 位，启用偏置
    Q4_3_NORM_NO_BIAS = "Q4.3_NORM_NO_BIAS"
    Q8_8_NORM_NO_BIAS = "Q8.8_NORM_NO_BIAS"
    Q16_16_NORM_NO_BIAS = "Q16.16_NORM_NO_BIAS"

    @property
    def bit_width(self) -> int:
        return {
            Scheme.Q4_3_NORM: 8,
            Scheme.Q8_8_NORM: 16,
            Scheme.Q16_16_NORM: 32,
            Scheme.Q4_3_NORM_NO_BIAS: 8,
            Scheme.Q8_8_NORM_NO_BIAS: 16,
            Scheme.Q16_16_NORM_NO_BIAS: 32,
        }[self]

    @property
    def frac_bits(self) -> int:
        return {
            Scheme.Q4_3_NORM: 3,
            Scheme.Q8_8_NORM: 8,
            Scheme.Q16_16_NORM: 16,
            Scheme.Q4_3_NORM_NO_BIAS: 3,
            Scheme.Q8_8_NORM_NO_BIAS: 8,
            Scheme.Q16_16_NORM_NO_BIAS: 16,
        }[self]

    @property
    def use_bias(self) -> bool:
        return not self.name.endswith("NO_BIAS")


BIT_WIDTH_TO_SCHEME = {
    8: Scheme.Q4_3_NORM,
    16: Scheme.Q8_8_NORM,
    32: Scheme.Q16_16_NORM,
}
BIT_WIDTH_TO_NO_BIAS_SCHEME = {
    8: Scheme.Q4_3_NORM_NO_BIAS,
    16: Scheme.Q8_8_NORM_NO_BIAS,
    32: Scheme.Q16_16_NORM_NO_BIAS,
}
NAMED_ROUNDING_MODES = ("HALF_EVEN", "HALF_UP", "TRUNC", "FLOOR")
DEFAULT_ROUNDING = "HALF_EVEN"
TIERS = ("coarse", "standard", "fine")


@dataclass
class TensorPackage:
    """多层激活张量包：layers 为层名 -> 浮点激活矩阵（行 x 列）。"""

    layers: dict[str, list[list[float]]]
    depends_on: dict[str, list[str]] = field(default_factory=dict)
    seed: int = 0

    def dims(self, layer: str) -> tuple[int, int]:
        matrix = self.layers[layer]
        return len(matrix), len(matrix[0])


@dataclass
class OpConfig:
    """算子与调度配置。"""

    frac_bits: int
    tier: str
    rounding_mode: str
    max_workers: int
    element_budget: int          # 单批算子元素预算
    layer_row_quota: int         # 单层单批行数配额
    tolerances: dict[str, float]
    config_version: int = 1

    def fingerprint_inputs(self) -> tuple[int, str, str, int, int, int]:
        return (
            self.frac_bits,
            self.tier,
            self.rounding_mode,
            self.max_workers,
            self.element_budget,
            self.layer_row_quota,
        )


@dataclass
class V1WeightRow:
    layer: str
    row_index: int
    scale_permille: int
    bias_code: int
    bit_width: int


@dataclass
class V2WeightRow:
    layer: str
    row_index: int
    scale: float
    scheme: str
    rounding_mode: str


@dataclass
class EditCommand:
    """编辑命令流中的单条命令。

    kind 取值：
      switch_version / replace_layer_params / replace_row_activation /
      set_frac_bits / set_tier
    """

    seq: int
    kind: str
    payload: dict


@dataclass
class InjectionItem:
    inject_id: str
    fault: str          # non_finite_scale / zero_variance / bad_bitwidth / duplicate_layer
    target_layer: str
    target_row: int | None = None
    payload: dict = field(default_factory=dict)


def dataclass_to_dict(obj) -> dict:
    return asdict(obj)
