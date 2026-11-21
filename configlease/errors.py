"""Structured error codes shared by the platform, nodes and validation tools.

Every failure raised inside the system is an :class:`LeaseError` carrying a
stable machine readable ``code`` (the ``E_*`` family required by the task) plus
context fields that are written verbatim to the structured log.
"""
from __future__ import annotations


class LeaseError(Exception):
    code = "E_UNKNOWN"

    def __init__(self, message: str = "", **context):
        super().__init__(message or self.code)
        self.message = message or self.code
        self.context = context

    def to_dict(self) -> dict:
        data = {"code": self.code, "message": self.message}
        data.update(self.context)
        return data


class ManifestError(LeaseError):
    code = "E_MANIFEST"


class PortInUseError(LeaseError):
    code = "E_PORT"


class NodeStartError(LeaseError):
    code = "E_START"


class HandshakeTimeoutError(LeaseError):
    code = "E_HANDSHAKE"


class KeyNotCoveredError(LeaseError):
    code = "E_KEY"


class RevisionOutOfRangeError(LeaseError):
    code = "E_REV"


class RollbackRevisionError(LeaseError):
    code = "E_ROLLBACK"


class ApplyMismatchError(LeaseError):
    code = "E_APPLY"


class LeaseExpiredError(LeaseError):
    code = "E_EXPIRED"


class RecoverError(LeaseError):
    code = "E_RECOVER"


class VersionConflictError(LeaseError):
    code = "E_VERSION"


class ReplayMismatchError(LeaseError):
    code = "E_REPLAY"


class ArbitrationTimeoutError(LeaseError):
    code = "E_TIMEOUT"


ERROR_CODES = (
    "E_MANIFEST",
    "E_PORT",
    "E_START",
    "E_HANDSHAKE",
    "E_KEY",
    "E_REV",
    "E_ROLLBACK",
    "E_APPLY",
    "E_EXPIRED",
    "E_RECOVER",
    "E_VERSION",
    "E_REPLAY",
    "E_TIMEOUT",
)
