"""Domain error codes for the service-mesh workbench.

Every user-facing failure is expressed as a :class:`BenchError` carrying a
stable machine-readable ``code`` (for example ``E_RULE``).  Structured logs and
the CLI use these codes so that injected faults and genuine defects share one
classification vocabulary.
"""


class BenchError(Exception):
    """Base class carrying a stable error ``code`` and structured context."""

    code = "E_UNKNOWN"

    def __init__(self, message, code=None, **context):
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        self.context = context

    def as_dict(self):
        data = {"code": self.code, "message": self.message}
        data.update(self.context)
        return data


class RuleError(BenchError):
    """A rule declaration failed strong validation (E_RULE)."""

    code = "E_RULE"


class CircuitError(BenchError):
    """A fast-fail because the target instance is open (E_CIRCUIT)."""

    code = "E_CIRCUIT"


class UpstreamError(BenchError):
    """A real loopback upstream is unreachable (E_UPSTREAM)."""

    code = "E_UPSTREAM"


class PrevailError(BenchError):
    """Cached receipt differs from the authoritative recomputation (E_PREVAIL)."""

    code = "E_PREVAIL"


class ReplayError(BenchError):
    """Ledger replay cannot reproduce a stored receipt (E_REPLAY)."""

    code = "E_REPLAY"


class SnapshotError(BenchError):
    """Session snapshot/ledger content digest mismatch (E_SNAPSHOT)."""

    code = "E_SNAPSHOT"
