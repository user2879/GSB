"""本地确定性跨模态对齐与检索数据处理管线。

仅依赖 Python 标准库与 NumPy，所有产物均在本地生成，运行结果与时间、
机器无关：缓存键、快照版本号与报告内容全部由输入内容与配置派生。
"""

__all__ = ["errors", "hashing", "png_codec", "features", "formats",
           "alignment", "convert", "cache", "snapshot", "injection",
           "report", "pipeline", "cli"]
