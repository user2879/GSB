"""业务错误码与异常定义。

所有可预期的业务违规都抛出 :class:`KvError`，其 ``code`` 为稳定的业务码：

* E_RECORD    键值/索引记录非法（键格式、过期戳、写入序号等）
* E_TRACE     轨迹文件损坏或操作约束违规
* E_EXPIRED   逻辑时钟判定键已过期
* E_INDEX     增量计数视图与按键集合重算结果不一致
* E_VERSION   索引条目版本回退
* E_SNAPSHOT  快照内容摘要/版本不匹配
* E_VISIBLE   快速查询结论与权威重放结论不一致
* E_REPLAY    快照恢复后重放出现分歧
"""

from __future__ import annotations

from typing import Any, Mapping, Optional


class KvError(Exception):
    """带稳定业务码的异常基类。"""

    code: str = "E_GENERIC"

    def __init__(
        self,
        message: str,
        *,
        code: Optional[str] = None,
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        if code is not None:
            self.code = code
        self.message = message
        self.details: dict[str, Any] = dict(details or {})

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": self.details}

    def __str__(self) -> str:  # pragma: no cover - 调试便利
        return f"{self.code}: {self.message} {self.details}".rstrip()


class RecordError(KvError):
    code = "E_RECORD"


class TraceError(KvError):
    code = "E_TRACE"


class ExpiredError(KvError):
    code = "E_EXPIRED"


class IndexError(KvError):
    code = "E_INDEX"


class VersionError(KvError):
    code = "E_VERSION"


class SnapshotError(KvError):
    code = "E_SNAPSHOT"


class VisibleError(KvError):
    code = "E_VISIBLE"


class ReplayError(KvError):
    code = "E_REPLAY"
