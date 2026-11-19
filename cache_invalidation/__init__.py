"""本地缓存失效传播中间件（仅依赖 Python 标准库）。

模块划分：
- errors: 业务错误码与结构化异常
- model: 命名空间、条目、注释依赖图等领域模型
- jsonpos: 带行号的 JSON 解析器
- loader: 本地声明文件加载与强校验
- snapshot: v1/v2/v3 版本化快照编解码与归一
- engine: 失效传播引擎、SQLite 持久化、恢复与重放
- netloop: 回环端口上的确定性 JSON Lines 服务
- gui: Tkinter 诊断界面（延迟导入，无显示环境也可导入包）
- logging_json: 结构化日志
- fault_injection: 只修改输入文件/参数的异常注入
- cli: 命令行入口
"""

from .errors import (
    BusinessError,
    EGraph,
    EFormat,
    EClock,
    ESnapshot,
    EInherit,
    EReplay,
)
from .model import EntryID, Entry, Namespace, Graph

STATE_VERSION = 1
SUPPORTED_FORMAT_VERSIONS = (1, 2, 3)
SUPPORTED_DIGEST_ALGORITHMS = ("none", "crc32", "sha1", "sha256")

__all__ = [
    "BusinessError",
    "EGraph",
    "EFormat",
    "EClock",
    "ESnapshot",
    "EInherit",
    "EReplay",
    "EntryID",
    "Entry",
    "Namespace",
    "Graph",
    "STATE_VERSION",
    "SUPPORTED_FORMAT_VERSIONS",
    "SUPPORTED_DIGEST_ALGORITHMS",
]
