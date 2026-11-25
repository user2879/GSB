"""整数定点算术：量化、截断方向、溢出饱和与误差区间。

全部代价计算只使用 :class:`int`，绝不使用浮点比较决定胜负。

约定：
- 会话缩放因子 ``scale`` 为正整数（10 的幂），表示一个定点单位的细分粒度；
- 除法统一采用向负无穷截断（floor），并在算子处显式标注；
- 乘法在合成前先反量化（先除后乘），所有中间量以 63 位有符号饱和；
- 误差区间以缩放后的定点整数表示，记录量化截断累积上界。
"""

from .errors import SpecError

SAT_MAX = (1 << 62) - 1
SAT_MIN = -(1 << 62)


def validate_scale(scale) -> int:
    if not isinstance(scale, int) or isinstance(scale, bool):
        raise SpecError("缩放因子必须为正整数", field="scale")
    if scale <= 0:
        raise SpecError("缩放因子必须为正整数", field="scale")
    return scale


def sat(value: int) -> int:
    """63 位有符号饱和：超出范围时夹到边界并保留饱和事实由调用方记录。"""
    if value > SAT_MAX:
        return SAT_MAX
    if value < SAT_MIN:
        return SAT_MIN
    return value


def mul_fixed(a: int, b: int, scale: int) -> int:
    """两个同尺度定点整数相乘，先反量化再饱和，截断方向 floor。"""
    return sat((a * b) // scale)


def div_fixed(a: int, b: int, scale: int = 1) -> int:
    """定点/整数除法，floor 截断；``b`` 为正整数。"""
    if b <= 0:
        raise SpecError("定点除法分母必须为正整数", field="denominator")
    return sat(a // b)


def quantize(value: int, scale: int) -> int:
    """把整数行数量化为定点单位（行计数本身为整数，量化即乘尺度并饱和）。"""
    if not isinstance(value, int) or isinstance(value, bool):
        raise SpecError("待量化行计数必须为整数", field="rows")
    return sat(value * scale)


def ceil_div(a: int, b: int) -> int:
    if b <= 0:
        raise SpecError("ceil_div 分母必须为正整数", field="denominator")
    return -((-a) // b)


def add(a: int, b: int) -> int:
    return sat(a + b)


def sub(a: int, b: int) -> int:
    return sat(a - b)


class ErrorBudget:
    """合成误差区间（定点整数）。

    每个 floor 除法最多引入 1 个“反量化单位”的截断误差；
    误差随算子线性累加，误差区间对称表示为 ``[-epsilon, +epsilon]``。
    """

    def __init__(self, scale: int):
        self.scale = scale
        self.epsilon = 0
        self.sources = []

    def record_div(self, op: str, field: str):
        self.epsilon = add(self.epsilon, 1)
        self.sources.append({"op": op, "field": field, "kind": "floor_div", "unit": 1})

    def record_quant(self, op: str, field: str, units: int = 1):
        self.epsilon = sat(self.epsilon + units)
        self.sources.append({"op": op, "field": field, "kind": "quantize", "unit": units})

    def interval(self):
        return {"low": -self.epsilon, "high": self.epsilon, "epsilon": self.epsilon}
