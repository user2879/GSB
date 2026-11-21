"""Domain error hierarchy for the delay queue component.

All component-level failures derive from :class:`DelayQueueError`.  Each
business error carries a stable machine code (``E_SPEC`` ...) so callers can
branch deterministically.
"""


class DelayQueueError(Exception):
    """Base class for every error raised by the package."""

    code = "E_DQ"

    def __init__(self, message: str = "", **details):
        super().__init__(message)
        self.message = message
        self.details = details

    def __str__(self) -> str:
        if self.details:
            extra = ", ".join(f"{k}={v!r}" for k, v in self.details.items())
            return f"[{self.code}] {self.message} ({extra})"
        return f"[{self.code}] {self.message}"


class SpecError(DelayQueueError):
    """Raised when a topic declaration or message stream fails validation.

    The whole load is rejected; ``line`` (0 for declaration level) and
    ``field`` point at the first offending location.
    """

    code = "E_SPEC"

    def __init__(self, message: str, line: int = 0, field: str = ""):
        super().__init__(message, line=line, field=field)
        self.line = line
        self.field = field


class IllegalTransitionError(DelayQueueError):
    """Raised when a state transition is not present in the permission table."""

    code = "E_TRANSITION"


class SnapshotMismatchError(DelayQueueError):
    """Raised when on-disk format version or content digest disagrees.

    The mismatch report lists message-count delta, version delta and the number
    of migration records that would be discarded by a full replacement.
    """

    code = "E_SNAPSHOT"

    def __init__(self, message: str, report: dict | None = None):
        super().__init__(message)
        self.report = report or {}


class ReplayError(DelayQueueError):
    """Raised when ledger replay diverges from the persisted record."""

    code = "E_REPLAY"

    def __init__(self, message: str, message_id: str = "", field: str = "",
                 expected=None, actual=None):
        super().__init__(message, message_id=message_id, field=field,
                         expected=expected, actual=actual)
        self.message_id = message_id
        self.field = field
        self.expected = expected
        self.actual = actual


class TimeoutConfigError(DelayQueueError):
    """Raised when a visibility-timeout parameter is illegal (E_TIMEOUT)."""

    code = "E_TIMEOUT"


class NotReproducibleError(DelayQueueError):
    """Replay cannot be reproduced because a versioned input disagrees."""

    code = "E_NOT_REPRODUCIBLE"
