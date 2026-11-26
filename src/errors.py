"""业务错误码。

E_FIELD   回执字段级格式错误（宽度/类型/取值域），定位源行号与字段名。
E_REF     回执引用级错误（名次唯一性/越界、重复投递引用悬空）。
E_TIMEOUT 真实套接字超时。
E_PORT    回环端口被占用或不可连接。
E_REPLAY  恢复后重放校验发现的首个不一致。
"""


class BizError(Exception):
    code = "E_UNKNOWN"

    def __init__(self, message, **loc):
        super().__init__(message)
        self.message = message
        self.loc = loc

    def __str__(self):
        loc = " ".join(f"{k}={v}" for k, v in self.loc.items())
        return f"{self.code}: {self.message}" + (f" ({loc})" if loc else "")


class FieldError(BizError):
    code = "E_FIELD"


class RefError(BizError):
    code = "E_REF"


class SocketTimeoutError(BizError):
    code = "E_TIMEOUT"


class PortError(BizError):
    code = "E_PORT"


class ReplayError(BizError):
    code = "E_REPLAY"
