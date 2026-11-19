"""业务错误码与失败清单记录。

所有校验/运行期问题都以 FailureRecord 形式收集，校验阶段不会因为单个
配置项非法而中断，其余条目继续校验。
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any


# ---- 业务码 ---------------------------------------------------------------
E_CONFIG_INVALID = "E_CONFIG_INVALID"
E_CONFIG_MISMATCH = "E_CONFIG_MISMATCH"
E_BUDGET_EXHAUSTED = "E_BUDGET_EXHAUSTED"
E_TASK_QUOTA_EXCEEDED = "E_TASK_QUOTA_EXCEEDED"
E_TASK_FAILED = "E_TASK_FAILED"
E_STAGE_WRITE_FAILED = "E_STAGE_WRITE_FAILED"
E_GRACEFUL_STOP = "E_GRACEFUL_STOP"


@dataclass
class FailureRecord:
    """失败清单中的单条记录（可 JSON 序列化）。"""

    code: str
    stage: str
    message: str
    task_key: str | None = None
    epoch: int | None = None
    exception_type: str | None = None
    subject: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ConfigError(ValueError):
    """配置非法，携带业务码与具体条目。"""

    def __init__(self, code: str, message: str, subject: str | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.subject = subject


class ConfigMismatchError(RuntimeError):
    """恢复时快照配置哈希/版本与当前配置不一致。"""

    def __init__(self, message: str):
        super().__init__(message)
        self.code = E_CONFIG_MISMATCH


class BudgetExhaustedError(RuntimeError):
    """全局轮次预算耗尽，任务未启动。"""

    def __init__(self, message: str):
        super().__init__(message)
        self.code = E_BUDGET_EXHAUSTED


class TaskQuotaExceededError(RuntimeError):
    """单方案轮次配额被触及，任务按配额中止（属于受控状态而非崩溃）。"""

    def __init__(self, message: str, consumed_epochs: int):
        super().__init__(message)
        self.code = E_TASK_QUOTA_EXCEEDED
        self.consumed_epochs = consumed_epochs


class StageWriteError(RuntimeError):
    """快照阶段提交失败。"""

    def __init__(self, message: str):
        super().__init__(message)
        self.code = E_STAGE_WRITE_FAILED


@dataclass
class FailureLog:
    """线程安全的失败清单收集器。"""

    records: list[FailureRecord] = field(default_factory=list)

    def add(
        self,
        code: str,
        stage: str,
        message: str,
        task_key: str | None = None,
        epoch: int | None = None,
        exception_type: str | None = None,
        subject: str | None = None,
    ) -> None:
        self.records.append(
            FailureRecord(
                code=code,
                stage=stage,
                message=message,
                task_key=task_key,
                epoch=epoch,
                exception_type=exception_type,
                subject=subject,
            )
        )

    def extend(self, records: list[FailureRecord]) -> None:
        self.records.extend(records)

    def to_list(self) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self.records]

    def codes(self) -> list[str]:
        return [r.code for r in self.records]
