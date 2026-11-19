"""业务错误码与解析异常定义。"""

E_OK = "E_OK"
E_HEADER = "E_HEADER"
E_PART = "E_PART"
E_OVERLAP = "E_OVERLAP"
E_SELFREF = "E_SELFREF"
E_PATCH_HEADER = "E_PATCH_HEADER"
E_PATCH_BOUNDS = "E_PATCH_BOUNDS"
E_SLOT_ESCALATION = "E_SLOT_ESCALATION"
E_WRITE_CHAIN = "E_WRITE_CHAIN"
E_STALE_BASE = "E_STALE_BASE"
E_TIMEOUT = "E_TIMEOUT"
E_OPS = "E_OPS"
E_LOCK = "E_LOCK"
E_ARGS = "E_ARGS"
E_UNKNOWN = "E_UNKNOWN"

ALL_CODES = frozenset(
    [
        E_OK, E_HEADER, E_PART, E_OVERLAP, E_SELFREF,
        E_PATCH_HEADER, E_PATCH_BOUNDS, E_SLOT_ESCALATION,
        E_WRITE_CHAIN, E_STALE_BASE, E_TIMEOUT, E_OPS,
        E_LOCK, E_ARGS, E_UNKNOWN,
    ]
)

# 结构性/策略性结论（非解析偏移类）
STRUCTURAL_CODES = frozenset(
    [E_OVERLAP, E_SELFREF, E_WRITE_CHAIN, E_STALE_BASE, E_TIMEOUT, E_OPS]
)


class AuditError(Exception):
    """带业务码与字节偏移的审计异常。"""

    def __init__(self, code, message, offset=None):
        super().__init__(message)
        if code not in ALL_CODES:
            raise ValueError("unknown error code: %r" % (code,))
        self.code = code
        self.offset = offset
        self.message = message

    def as_dict(self):
        return {"code": self.code, "message": str(self), "offset": self.offset}
