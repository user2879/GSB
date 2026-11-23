"""量化方案、舍入模式、位宽等枚举与码表。"""
from __future__ import annotations

# 允许的整数量化位宽
ALLOWED_QBITS = (4, 8, 16, 32)

# v1 定宽文本字段宽度（字符数）
V1_FIELD_WIDTHS = {
    "group": 8,
    "column": 16,
    "scale_permille": 12,
    "zero_point": 8,
    "qbits": 4,
}
V1_LINE_WIDTH = sum(V1_FIELD_WIDTHS.values())  # 48

# 缺失标记
MISSING_TOKEN = None

# 量化方案：v1 单字符码 -> v2 命名
SCHEME_CODE_TO_NAME = {
    "S": "symmetric",
    "A": "asymmetric",
}
SCHEME_NAME_TO_CODE = {v: k for k, v in SCHEME_CODE_TO_NAME.items()}

# 舍入模式枚举（v2 命名，v1 不携带、以默认值补齐）
ROUNDING_MODES = (
    "half_up",       # 四舍五入（远离零，用于 v2->v1 permille 还原）
    "half_even",     # 银行家舍入
    "truncate",      # 向零截断
    "floor",         # 向负无穷
)
DEFAULT_ROUNDING = "half_up"

# 饱和截断策略
SATURATION_STRATEGIES = ("clip", "reject")
DEFAULT_SATURATION = "clip"

# v1 精度标签（permille 固定精度）
V1_PRECISION_LABEL = "permille-v1"

# 交换版本号（参与缓存键派生）
EXCHANGE_VERSIONS = (1, 2)


def qint_range(qbits: int, scheme: str) -> tuple[int, int]:
    """返回量化位宽下的整数可表示区间（含端点）。"""
    if qbits not in ALLOWED_QBITS:
        raise ValueError(f"qbits {qbits} not in {ALLOWED_QBITS}")
    if scheme == "symmetric":
        bound = (1 << (qbits - 1)) - 1
        return -bound, bound
    # asymmetric: 无符号整数
    return 0, (1 << qbits) - 1
