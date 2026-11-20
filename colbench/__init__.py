"""列式存储压缩与扫描诊断工作台。

仅依赖 Python 3.11+ 标准库与 NumPy；无外部服务、无网络访问。
"""

__all__ = ["errors", "codecs", "schema", "segment", "planner", "scheduler",
           "ledger", "replay", "versioning", "snapshot", "inject", "view",
           "gui", "sample_data"]
__version__ = "1.0.0"
