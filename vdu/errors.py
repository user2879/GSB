"""Business error codes and exception types for the VDU pipeline.

Every validation failure is raised as :class:`VDUError` carrying a stable
business code so callers can branch on machine-readable identifiers instead of
parsing human readable messages.
"""


class VDUError(Exception):
    """Base exception carrying a business code and a file/line locator."""

    def __init__(self, code, message, path=None, line=None):
        self.code = code
        self.path = path
        self.line = line
        located = message
        if path is not None:
            located = f"{path}:{line if line is not None else '?'} {message}"
        super().__init__(f"[{code}] {located}")


class AudioFormatError(VDUError):
    """Raised when a WAV container does not satisfy the 16k/mono/16bit rules."""

    def __init__(self, message, path=None, line=None):
        super().__init__("E_AUDIO_FORMAT", message, path, line)


class InputInvalidError(VDUError):
    """Raised when transcript / config content fails deterministic validation."""

    def __init__(self, message, path=None, line=None):
        super().__init__("E_INPUT_INVALID", message, path, line)


class EvidenceQuotaError(VDUError):
    """Raised (and recorded on a segment) when a segment exceeds its quota."""

    def __init__(self, message, path=None, line=None):
        super().__init__("E_EVIDENCE_QUOTA_EXCEEDED", message, path, line)


class EvidenceBudgetPending(VDUError):
    """Raised (and recorded) when the batch budget cannot cover the segment."""

    def __init__(self, message, path=None, line=None):
        super().__init__("E_EVIDENCE_BUDGET_PENDING", message, path, line)


class SnapshotError(VDUError):
    """Raised when persisted snapshot state is corrupt or incompatible."""

    def __init__(self, message, path=None, line=None):
        super().__init__("E_SNAPSHOT_INVALID", message, path, line)
