"""丢包与网络故障诊断工作台。

仅使用 Python 标准库与 NumPy，面向本地 Windows / 回环环境运行，
不依赖任何外部网络服务或数据集。
"""

__all__ = [
    "errors",
    "constants",
    "wire",
    "eventlog",
    "model",
    "merge",
    "gaps",
    "statistics",
    "replay",
    "snapshot",
    "ledger",
    "topology",
    "pipeline",
]

MODEL_VERSION = "3.0.0"
ARBITRATION_RULE_VERSION = "1.0.0"
