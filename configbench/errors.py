"""统一错误码与异常类型。

所有面向业务的失败均携带稳定的错误码字符串（E_*），便于账本、
测试与技术说明中按码判定，而不是依赖异常类名或本地化文案。
"""


class BenchError(Exception):
    """实验台所有受控异常的基类。"""

    code = "E_UNKNOWN"

    def __init__(self, message, *, field=None, basis=None, extra=None):
        super().__init__(message)
        self.message = message
        # 首个违规字段（如 manifest.versions[1].items["x"].value）
        self.field = field
        # 判定依据（期望/实际/规则描述）
        self.basis = basis
        self.extra = dict(extra or {})

    def to_dict(self):
        data = {"code": self.code, "message": self.message}
        if self.field is not None:
            data["field"] = self.field
        if self.basis is not None:
            data["basis"] = self.basis
        if self.extra:
            data["extra"] = self.extra
        return data


class ManifestError(BenchError):
    code = "E_MANIFEST"


class PortError(BenchError):
    code = "E_PORT"


class HandshakeError(BenchError):
    code = "E_HANDSHAKE"


class FieldError(BenchError):
    code = "E_FIELD"


class PreviewError(BenchError):
    code = "E_PREVIEW"


class LatencyError(BenchError):
    code = "E_LATENCY"


class ReplayError(BenchError):
    code = "E_REPLAY"


class LedgerError(BenchError):
    code = "E_LEDGER"


class TimeoutError(BenchError):
    code = "E_TIMEOUT"


class SchedulerError(BenchError):
    code = "E_SCHEDULER"
