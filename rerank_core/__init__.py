"""本地确定性搜索精排与重排领域组件。

仅依赖 Python 标准库（生成样本时可选 NumPy）。所有融合打分与折减均以
定点整数完成，保证任意并发度与交互命令序列下结果逐字段可复现。
"""

from .errors import (
    RerankError,
    ConfigInvalidError,
    RecordInvalidError,
    ConfigMismatchError,
    BudgetPendingError,
    BUSINESS_CODES,
)

__all__ = [
    "RerankError",
    "ConfigInvalidError",
    "RecordInvalidError",
    "ConfigMismatchError",
    "BudgetPendingError",
    "BUSINESS_CODES",
]
