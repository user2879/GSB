"""边缘缓存分片下发诊断工作台。

模块划分：
- errors      统一错误码与异常
- clock       单调时钟封装（真实 time.monotonic_ns，毫秒整数）
- determinism 确定性分片内容与 NumPy 定点校验摘要
- spec        声明/清单/事件流的 JSON 加载与强校验
- protocol    JSON 行协议的收发与真实套接字超时
- node        缓存节点子进程
- ledger      SQLite 原子账本、快照版本与恢复
- scheduler   三重约束并行调度与水位维护
- engine      编排：事件循环、租约生命周期、双路回收唯一归属裁决
- replay      独立只读重放校验
- cli         命令行入口
- ui          Tkinter 只读水位界面
"""

__version__ = "1.0.0"
