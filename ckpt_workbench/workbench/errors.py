"""Stable error codes used across the checkpoint workbench."""


class WorkbenchError(Exception):
    """Base class carrying a stable machine-readable code and details."""

    code = "E_UNKNOWN"

    def __init__(self, message, details=None):
        super().__init__(message)
        self.message = message
        self.details = details or []

    def __str__(self):
        lines = [f"{self.code}: {self.message}"]
        for item in self.details:
            lines.append(f"  - {item}")
        return "\n".join(lines)


class LedgerError(WorkbenchError):
    code = "E_LEDGER"


class DependencyError(WorkbenchError):
    code = "E_DEPEND"


class IncrementalError(WorkbenchError):
    code = "E_INCR"


class IntegrityError(WorkbenchError):
    code = "E_INTEGRITY"


class ChainError(WorkbenchError):
    code = "E_CHAIN"


class SnapshotError(WorkbenchError):
    code = "E_SNAPSHOT"


class ReplayError(WorkbenchError):
    code = "E_REPLAY"


CODE_BY_NAME = {
    "E_LEDGER": LedgerError,
    "E_DEPEND": DependencyError,
    "E_INCR": IncrementalError,
    "E_INTEGRITY": IntegrityError,
    "E_CHAIN": ChainError,
    "E_SNAPSHOT": SnapshotError,
    "E_REPLAY": ReplayError,
}
