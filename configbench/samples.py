"""固定种子确定性样本生成。

使用 NumPy 的新 Generator（PCG64）与固定种子，同种子/同参数在任意机器、
任意进程重复执行得到完全相同的整数数组；样本仅在本地生成，不访问网络。
"""

import numpy as np


def generate_samples(seed, n_samples, n_groups, value_range, declared_scale):
    """返回 (groups, values)，均为 int64 一维数组，长度 n_samples。

    groups[i] 取值 [0, n_groups)，values[i] 取值 [0, value_range)。
    values 的定点缩放因子为 declared_scale（raw 即生成整数）。
    """
    rng = np.random.default_rng(seed)
    groups = rng.integers(0, n_groups, size=n_samples, dtype=np.int64)
    values = rng.integers(0, value_range, size=n_samples, dtype=np.int64)
    return groups, values


def rescale_int(raw, from_scale, to_scale):
    """把定点 raw 从 from_scale 重缩放至 to_scale，向零截断（Python 实现）。"""
    from .fixedpoint import trunc_div

    if to_scale >= from_scale:
        return int(raw) * 10 ** (to_scale - from_scale)
    return trunc_div(int(raw), 10 ** (from_scale - to_scale))
