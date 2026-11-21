"""冷热分层与迁移组件 (Hot/Cold Tier Mover).

纯 Python 标准库实现（numpy 可选），面向 Windows 本地环境：
- SQLite 作为元数据账本
- 本地目录作为对象数据载体
- JSON 清单驱动，固定整数种子生成确定性对象内容
"""

__version__ = "1.0.0"
LEDGER_FORMAT_VERSION = 3
