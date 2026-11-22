"""统一错误码与结构化异常。"""


class DiagError(Exception):
    """所有诊断台错误携带稳定错误码与结构化细节。"""

    code = "E_UNKNOWN"

    def __init__(self, message, **details):
        super().__init__(message)
        self.message = message
        self.details = details

    def to_dict(self):
        return {"code": self.code, "message": self.message, "details": self.details}


class SpecError(DiagError):
    code = "E_SPEC"


class PortError(DiagError):
    code = "E_PORT"


class HandshakeError(DiagError):
    code = "E_HANDSHAKE"


class ProtoError(DiagError):
    code = "E_PROTO"


class LeaseError(DiagError):
    code = "E_LEASE"


class GraphError(DiagError):
    code = "E_GRAPH"


class SnapshotError(DiagError):
    code = "E_SNAPSHOT"


class VersionError(DiagError):
    code = "E_VERSION"


class ReplayError(DiagError):
    code = "E_REPLAY"
