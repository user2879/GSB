"""数字乐谱与演奏表达数据处理管线。

仅依赖 NumPy（生成器随机源）与 Python 标准库；所有持久化文件均为
确定性 JSON，跨平台（Windows / Linux）可重复。
"""

__version__ = "1.0.0"
SCHEMA = "score-pipeline/1"
SHARD_BEATS = 8
BEAT_UNIT_MS = 500.0
EPS = 1e-9

