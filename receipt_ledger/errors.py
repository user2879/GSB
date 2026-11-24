"""统一错误类型。错误码作为异常的稳定属性，供 CLI 与测试精确断言。"""


class LedgerError(Exception):
    """所有业务错误的基类，携带稳定错误码与结构化定位信息。"""

    code = "E_INTERNAL"

    def __init__(self, message, *, code=None, **location):
        super().__init__(message)
        if code is not None:
            self.code = code
        self.location = location

    def __str__(self):
        if self.location:
            detail = ", ".join(f"{k}={v!r}" for k, v in sorted(self.location.items()))
            return f"[{self.code}] {super().__str__()} ({detail})"
        return f"[{self.code}] {super().__str__()}"


class SpecError(LedgerError):
    """规格声明违规，定位首个违规项与字段。"""

    code = "E_SPEC"


class CodecError(LedgerError):
    """回执编解码失败：长度字段不自洽或字段类型非法。"""

    code = "E_CODEC"


class PreviewError(LedgerError):
    """快速缓存结论与权威重算不逐字节一致。"""

    code = "E_PREVIEW"


class ReplayError(LedgerError):
    """恢复重放时账本与重推导结论出现逐项差异。"""

    code = "E_REPLAY"


class SnapshotError(LedgerError):
    """快照版本或内容摘要与磁盘状态不一致。"""

    code = "E_SNAPSHOT"

