"""平台统一错误码与异常类型。

错误码约定：
  E_FORMAT   清单/视图格式校验失败（整体拒绝）
  E_PORT     节点回环端口被占用
  E_TIMEOUT  握手或下发套接字超时
  E_PROTO    节点回执格式不合法
  E_CONVERGE 期望摘要与回执摘要不一致或节点未回执
  E_PREVIEW  界面快速结论与账本权威重算不一致
  E_LATENCY  交互延迟超出预算阈值
  E_REPLAY   账本重放与记录不一致
"""

E_FORMAT = "E_FORMAT"
E_PORT = "E_PORT"
E_TIMEOUT = "E_TIMEOUT"
E_PROTO = "E_PROTO"
E_CONVERGE = "E_CONVERGE"
E_PREVIEW = "E_PREVIEW"
E_LATENCY = "E_LATENCY"
E_REPLAY = "E_REPLAY"


class PlatformError(Exception):
    code = "E_UNKNOWN"

    def __init__(self, message, **fields):
        super().__init__(message)
        self.message = message
        self.fields = fields

    def to_dict(self):
        return {"code": self.code, "message": self.message, **self.fields}

    def __str__(self):
        extras = " ".join(f"{k}={v!r}" for k, v in sorted(self.fields.items()))
        return f"{self.code}: {self.message}" + (f" ({extras})" if extras else "")


class FormatError(PlatformError):
    code = E_FORMAT

    def __init__(self, message, path, source_version):
        super().__init__(message, path=path, source_version=source_version)


class PortError(PlatformError):
    code = E_PORT

    def __init__(self, node, port):
        super().__init__("节点回环端口被占用", node=node, port=port)


class TimeoutError_(PlatformError):
    code = E_TIMEOUT

    def __init__(self, node, timeout_ms, phase):
        super().__init__("套接字超时", node=node, timeout_ms=timeout_ms, phase=phase)


class ProtoError(PlatformError):
    code = E_PROTO

    def __init__(self, node, detail):
        super().__init__("回执格式不合法", node=node, detail=detail)


class ConvergeError(PlatformError):
    code = E_CONVERGE

    def __init__(self, deviations):
        # deviations: [{"node","key","expected","actual"}] 已按 (node, key) 稳定排序
        super().__init__("期望摘要与回执摘要不一致", deviations=deviations)


class PreviewError(PlatformError):
    code = E_PREVIEW

    def __init__(self, key, node):
        super().__init__("快速结论与权威重算不一致", key=key, node=node)


class LatencyError(PlatformError):
    code = E_LATENCY

    def __init__(self, kind, p95_ms, budget_ms):
        super().__init__("交互延迟超出预算", kind=kind, p95_ms=p95_ms, budget_ms=budget_ms)


class ReplayError(PlatformError):
    code = E_REPLAY

    def __init__(self, key, node, field, expected, actual, basis):
        super().__init__(
            "账本重放不一致",
            key=key, node=node, field=field,
            expected=expected, actual=actual, basis=basis,
        )
