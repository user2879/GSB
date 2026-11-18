"""定点整数量化工具。

缩放因子 S 必须是 10 的整数次幂（k>=0）。所有比例统一量化为定点整数：
    q = floor(numerator * S / denominator)        # 向负无穷截断；本域均为非负，即向下截断
单点绝对误差上界（单位：1 个定点单位 / ulp）恒为 1：
    0 <= 真值*S - q < 1
比例比较的误差界为两侧 ulp 之和：|q_a - q_b| <= 2 时判定为不可区分。
"""

from .errors import SpecError


def power_of_ten(scale):
    """判断 scale 是否为 10 的非负整数次幂，返回指数 k，否则报错。"""
    if not isinstance(scale, int) or isinstance(scale, bool) or scale < 1:
        raise SpecError("scale 必须是正整数（10 的整数次幂）", path="scale", value=scale)
    s = scale
    k = 0
    while s > 1:
        if s % 10 != 0:
            raise SpecError("scale 必须是 10 的整数次幂", path="scale", value=scale)
        s //= 10
        k += 1
    return k


def scaled_ratio(numerator, denominator, scale):
    """floor(numerator * scale / denominator)，纯整数运算，分母为 0 时返回 0。"""
    if denominator <= 0:
        return 0
    return (int(numerator) * int(scale)) // int(denominator)


def single_value_error_bound():
    """单值定点量化误差上界（向下截断，误差区间 [0,1) ulp，上界取 1）。"""
    return 1


def comparison_error_bound():
    """两个定点比例做差比较时的合成误差上界（两侧 ulp 之和）。"""
    return 2


def indistinguishable(q_a, q_b):
    """差值落在比较误差界内即判定不可区分。"""
    return abs(int(q_a) - int(q_b)) <= comparison_error_bound()
