"""layerbench - 本地容器镜像分层构建与构建缓存诊断工作台（无 Docker，纯标准库）。

模块划分：
- errors     统一业务码与异常
- digest     确定性哈希组合、稳定序列化、换行标准化
- manifest   JSON 清单加载与强校验
- executor   跨平台本地命令执行（Windows cmd / POSIX sh）
- cache      跨构建缓存索引、稳定淘汰
- ledger     SQLite 账本
- scheduler  双重预算并行就绪调度
- report     确定性报告与预算审计
- snapshot   会话快照、整体替换恢复、强制终止恢复
- replay     恢复后重放校验与异常注入
- gui        可选 Tkinter 诊断窗口
- cli        命令行入口
"""

__version__ = "1.0.0"
