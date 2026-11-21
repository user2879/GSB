"""Business error codes used across the hotgray package."""

CODES = {
    "E_SPEC": "invalid gateway declaration",
    "E_PORT": "loopback port unavailable",
    "E_HANDSHAKE": "backend handshake/switch timeout",
    "E_REPLY": "backend reply missing or malformed",
    "E_SNAPSHOT": "snapshot disagrees with authoritative recompute",
    "E_REPLAY": "recovery replay not reproducible",
    "E_STATE": "persisted state format or content digest mismatch",
}


class HotGrayError(Exception):
    """Base class carrying a stable business error code and offending field."""

    code = "E_STATE"

    def __init__(self, message="", *, field=None, detail=None):
        super().__init__(message or CODES.get(self.code, ""))
        self.field = field
        self.detail = detail

    def to_dict(self):
        payload = {"code": self.code, "message": str(self)}
        if self.field is not None:
            payload["field"] = self.field
        if self.detail is not None:
            payload["detail"] = self.detail
        return payload


class SpecError(HotGrayError):
    code = "E_SPEC"


class PortError(HotGrayError):
    code = "E_PORT"


class HandshakeError(HotGrayError):
    code = "E_HANDSHAKE"


class ReplyError(HotGrayError):
    code = "E_REPLY"


class SnapshotError(HotGrayError):
    code = "E_SNAPSHOT"


class ReplayError(HotGrayError):
    code = "E_REPLAY"


class StateError(HotGrayError):
    code = "E_STATE"
