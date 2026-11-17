"""业务错误码与领域异常。"""

from __future__ import annotations

from typing import Any


class DomainError(Exception):
    """所有业务错误的基类，携带稳定业务码与结构化明细。"""

    code = "E_DOMAIN"

    def __init__(self, message: str, details: list[dict[str, Any]] | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or []

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": self.details}


class GraphError(DomainError):
    """E_GRAPH：图文档或变更集非法。"""

    code = "E_GRAPH"


class IndexIntegrityError(DomainError):
    """E_INDEX：索引定义或快照中的索引条目完整性校验失败。"""

    code = "E_INDEX"


class DependencyError(DomainError):
    """E_DEPEND：派生视图依赖成环或被引用删除。"""

    code = "E_DEPEND"


class BudgetError(DomainError):
    """E_BUDGET：重算预算不足（含预算为零）。"""

    code = "E_BUDGET"


class StaleError(DomainError):
    """E_STALE：查询引用了处于失效状态的索引条目或视图。"""

    code = "E_STALE"


class ReplayError(DomainError):
    """E_REPLAY：恢复后重放校验发现首个不一致。"""

    code = "E_REPLAY"
