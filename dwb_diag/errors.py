"""业务码与结构化异常。

业务码约定:
  E_PAGE      双写缓冲镜像结构性非法(页号/页大小/摘要/序号)
  E_CORRUPT   主数据文件页内容损坏(主文件摘要不匹配)
  E_LOST      主文件与缓冲中均无可用镜像, 页无法恢复
  E_DEPEND    页版本映射与视图依赖成环
  E_SNAPSHOT  快照缓冲摘要与当前磁盘不一致
  E_VIEW      视图重建/校验不一致(版本未递增或摘要异常)
  E_REPLAY    按快照重放修复/失效轨迹时出现差异
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class DiagError(Exception):
    """所有组件业务异常的基类。"""

    code = "E_INTERNAL"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        if code is not None:
            self.code = code
        self.message = message
        self.detail: dict[str, Any] = dict(detail or {})

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"code": self.code, "message": self.message}
        out.update(self.detail)
        return out


class PageFormatError(DiagError):
    """E_PAGE: 双写缓冲镜像结构性非法。"""

    code = "E_PAGE"

    def __init__(
        self,
        message: str,
        *,
        offset: int,
        page_no: int | None,
        reason: str | None = None,
        **kw: Any,
    ) -> None:
        detail = {"offset": offset, "page_no": page_no}
        if reason is not None:
            detail["reason"] = reason
        detail.update(kw.pop("detail", {}) or {})
        super().__init__(message, detail=detail, **kw)


class MainCorruptError(DiagError):
    """E_CORRUPT: 主数据文件页损坏。"""

    code = "E_CORRUPT"


class LostPageError(DiagError):
    """E_LOST: 页在主文件与缓冲中均无可用镜像。"""

    code = "E_LOST"

    def __init__(self, message: str, *, page_no: int, max_seq: int, **kw: Any) -> None:
        super().__init__(message, detail={"page_no": page_no, "max_seq": max_seq}, **kw)


class DependCycleError(DiagError):
    """E_DEPEND: 依赖关系成环。"""

    code = "E_DEPEND"

    def __init__(self, message: str, *, cycle: list[str], **kw: Any) -> None:
        super().__init__(message, detail={"cycle": cycle}, **kw)


class SnapshotMismatchError(DiagError):
    """E_SNAPSHOT: 快照与当前磁盘状态不一致(触发整体替换, 非致命)。"""

    code = "E_SNAPSHOT"


class ViewError(DiagError):
    """E_VIEW: 视图重建/校验不一致。"""

    code = "E_VIEW"

    def __init__(self, message: str, *, view: str, field: str, **kw: Any) -> None:
        super().__init__(message, detail={"view": view, "field": field}, **kw)


class ReplayError(DiagError):
    """E_REPLAY: 按快照重放时出现差异。"""

    code = "E_REPLAY"

    def __init__(self, message: str, *, page_no: int | None, field: str, **kw: Any) -> None:
        super().__init__(message, detail={"page_no": page_no, "field": field}, **kw)


@dataclass
class ErrorRecord:
    """结构化日志中记录的一次业务码事件。"""

    code: str
    message: str
    detail: dict[str, Any] = field(default_factory=dict)
