"""faultdiag: 面向 Windows 本地环境的故障隔离与恢复诊断组件。

以故障域标记为核心, 从本地 JSON 声明加载实例/故障域, 依据 NDJSON
故障事件流推导健康与隔离状态, 通过 127.0.0.1 回环 TCP 的 JSON 行
协议向确定性模拟服务子进程下发隔离/恢复指令, 并将全部决策与回执
持久化到本地 SQLite 账本。
"""

from .errors import DiagError

__all__ = ["DiagError"]
__version__ = "1.0.0"
