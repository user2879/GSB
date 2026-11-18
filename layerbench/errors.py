"""统一业务码与异常。

业务码（与需求条目一一对应）：
- E_MANIFEST 清单强校验失败（整体拒绝加载，不部分生效）
- E_TIMEOUT  单指令执行超时
- E_COMMAND  执行类指令非零退出
- E_AUDIT    构建报告预算/一致性审计发现偏差
- E_REPLAY   快照恢复后重放与快照记录不一致
- E_SNAPSHOT 快照内容摘要/版本/缓存索引不匹配或损坏
- E_BUDGET   预算参数非法（例如并行槽位上限为 0，属注入类异常）
"""
from __future__ import annotations

# 指令生命周期状态（写入账本 status）
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"      # 已占槽位、执行中（崩溃后判定为未知）
STATUS_COMMITTED = "committed"  # 已完整提交（成功或失败均落库后）
STATUS_SKIPPED = "skipped"      # 上游失败传播跳过
STATUS_UNKNOWN = "unknown"      # 崩溃恢复时发现执行中未提交，要求重跑

# 指令业务结果码
OK = "OK"
E_MANIFEST = "E_MANIFEST"
E_TIMEOUT = "E_TIMEOUT"
E_COMMAND = "E_COMMAND"
E_AUDIT = "E_AUDIT"
E_REPLAY = "E_REPLAY"
E_SNAPSHOT = "E_SNAPSHOT"
E_BUDGET = "E_BUDGET"

ALL_ERROR_CODES = (
    E_MANIFEST, E_TIMEOUT, E_COMMAND, E_AUDIT,
    E_REPLAY, E_SNAPSHOT, E_BUDGET,
)


class LayerBenchError(Exception):
    """所有可预期业务异常的基类。"""

    def __init__(self, code: str, message: str, location: str | None = None,
                 details: dict | None = None):
        self.code = code
        self.location = location
        self.details = details or {}
        full = f"[{code}] {message}"
        if location:
            full = f"[{code}] {location}: {message}"
        super().__init__(full)


class ManifestError(LayerBenchError):
    def __init__(self, message: str, location: str | None = None,
                 problems: list | None = None):
        # problems 为收集到的全部错误，保证一次性整体拒绝
        self.problems = problems or []
        detail = message
        if problems:
            detail = message + " | " + " ; ".join(problems)
        super().__init__(E_MANIFEST, detail, location,
                         {"problems": self.problems})
