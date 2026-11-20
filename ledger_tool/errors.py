"""结构化业务错误码."""

E_FORMAT = "E_FORMAT"
E_UPGRADE = "E_UPGRADE"
E_DIVERGE = "E_DIVERGE"
E_SNAPSHOT = "E_SNAPSHOT"
E_REPLAY = "E_REPLAY"
E_INTEGRITY = "E_INTEGRITY"
E_CHAIN = "E_CHAIN"


class LedgerError(Exception):
    """携带业务码、记录标识、字段与检出阶段的异常."""

    def __init__(self, code, message, record_id=None, field=None, stage=None, extra=None):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.record_id = record_id
        self.field = field
        self.stage = stage
        self.extra = extra or {}

    def to_dict(self):
        return {
            "code": self.code,
            "message": self.message,
            "record_id": self.record_id,
            "field": self.field,
            "stage": self.stage,
            "extra": self.extra,
        }
