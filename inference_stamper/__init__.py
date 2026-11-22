"""本地推理服务身份摘要与推理令牌盖章组件。

所有随机量均由整数种子经纯 Python 实现的 splitmix64 派生，
不依赖模型下载或任何外部服务。
"""

from .errors import (
    ITSError,
    SpecError,
    StaleError,
    IdentityError,
    SnapshotError,
    ReplayError,
    AuditError,
)

__all__ = [
    "ITSError",
    "SpecError",
    "StaleError",
    "IdentityError",
    "SnapshotError",
    "ReplayError",
    "AuditError",
]
