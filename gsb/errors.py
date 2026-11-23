"""统一错误类型：E_SPEC / E_FORMAT / E_MIXED / E_REPLAY。"""

from __future__ import annotations


class GSBError(Exception):
    """携带错误码与结构化上下文的业务错误。"""

    def __init__(self, code: str, message: str, **context):
        self.code = code
        self.message = message
        self.context = dict(context)
        ctx = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
        super().__init__(f"[{code}] {message}" + (f" | {ctx}" if ctx else ""))


def e_spec(message: str, *, field: str, owner: str) -> GSBError:
    return GSBError("E_SPEC", message, field=field, owner=owner)


def e_format(message: str, *, batch_id: str, field: str) -> GSBError:
    return GSBError("E_FORMAT", message, batch_id=batch_id, field=field)


def e_mixed(message: str, *, batch_id: str, versions) -> GSBError:
    return GSBError("E_MIXED", message, batch_id=batch_id, versions=list(versions))


def e_replay(message: str, *, batch_id: str, request_id: str, field: str) -> GSBError:
    return GSBError(
        "E_REPLAY", message, batch_id=batch_id, request_id=request_id, field=field
    )
