"""Structured error codes used across the platform."""

E_PORT = "E_PORT"            # loopback port occupied / unavailable
E_NODE = "E_NODE"            # node subprocess failed to start or died
E_HANDSHAKE = "E_HANDSHAKE"  # handshake timeout / invalid hello
E_SCOPE = "E_SCOPE"          # access outside lease sector scope
E_SEQ = "E_SEQ"              # operation sequence out of lease range
E_OWNER = "E_OWNER"          # write to a sector not owned by the node
E_FIELD = "E_FIELD"          # whitelist field validation failure
E_VER = "E_VER"              # summary / version mismatch on verification
E_RECOVER = "E_RECOVER"      # recovery consistency check failure
E_REPLAY = "E_REPLAY"        # snapshot replay divergence
E_TIMEOUT = "E_TIMEOUT"      # adjudication attempt timed out
E_BUSY = "E_BUSY"            # sector still covered by a non-terminal lease


class PlatformError(Exception):
    def __init__(self, code, **detail):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")

    def to_dict(self):
        return {"code": self.code, "detail": self.detail}
