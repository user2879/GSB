"""统一错误码与管线异常。"""

E_SCHEMA_TOO_NEW = "E_SCHEMA_TOO_NEW"
E_FORMAT_FIELD = "E_FORMAT_FIELD"
E_PEER_GAP = "E_PEER_GAP"
E_BASE_VERSION = "E_BASE_VERSION"
E_SCHEMA_FIELD = "E_SCHEMA_FIELD"


class PipelineError(Exception):
    """携带业务错误码的管线异常。"""

    def __init__(self, code, message, *, offset=None, field=None):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.offset = offset
        self.field = field

    def to_dict(self):
        return {
            "code": self.code,
            "message": self.message,
            "offset": self.offset,
            "field": self.field,
        }
