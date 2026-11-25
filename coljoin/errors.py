"""业务错误码与异常定义。

所有可预期的非法输入/状态均抛出 :class:`ColjoinError` 的子类，
异常携带稳定的业务码、首个非法对象标识、字段名与人类可读信息。
"""


class ColjoinError(Exception):
    code = "E_INTERNAL"

    def __init__(self, message, *, object_id=None, field=None, line=None):
        super().__init__(message)
        self.message = message
        self.object_id = object_id
        self.field = field
        self.line = line

    def to_dict(self):
        return {
            "code": self.code,
            "message": self.message,
            "object": self.object_id,
            "field": self.field,
            "line": self.line,
        }

    def __str__(self):
        loc = []
        if self.object_id is not None:
            loc.append(f"object={self.object_id}")
        if self.field is not None:
            loc.append(f"field={self.field}")
        if self.line is not None:
            loc.append(f"line={self.line}")
        suffix = (" [" + ", ".join(loc) + "]") if loc else ""
        return f"{self.code}: {self.message}{suffix}"


class SpecError(ColjoinError):
    """声明/查询/权重等静态规格非法。"""

    code = "E_SPEC"


class FormatError(ColjoinError):
    """回执文件编解码与跨版本归一时的格式非法。"""

    code = "E_FORMAT"


class ReplayError(ColjoinError):
    """强制终止恢复后的一致性重放发现分歧。"""

    code = "E_REPLAY"


class PortError(ColjoinError):
    """127.0.0.1 高位端口探测/绑定失败。"""

    code = "E_PORT"


class ProtocolError(ColjoinError):
    """执行器 TCP 协议握手或收发超时/非法帧。"""

    code = "E_PROTOCOL"
