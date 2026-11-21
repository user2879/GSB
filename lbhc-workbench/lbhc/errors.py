"""Business error codes surfaced by the workbench.

Every abnormal exit carries one of these codes, a human readable message,
an optional JSON field path, and a process exit code so scripts can branch.
"""

EXIT_OK = 0
EXIT_SPEC = 2
EXIT_PORT = 3
EXIT_LEDGER = 4
EXIT_REPLAY = 5


class BenchError(Exception):
    CODE = "E_UNKNOWN"
    EXIT = 1

    def __init__(self, message, path=None, detail=None):
        super().__init__(message)
        self.message = message
        self.path = path
        self.detail = detail

    def to_dict(self):
        return {
            "code": self.CODE,
            "message": self.message,
            "path": self.path,
            "detail": self.detail,
        }

    def __str__(self):
        loc = f" at {self.path}" if self.path else ""
        return f"{self.CODE}{loc}: {self.message}"


class SpecError(BenchError):
    CODE = "E_SPEC"
    EXIT = EXIT_SPEC


class PortError(BenchError):
    CODE = "E_PORT"
    EXIT = EXIT_PORT


class LedgerError(BenchError):
    CODE = "E_LEDGER"
    EXIT = EXIT_LEDGER


class ReplayError(BenchError):
    CODE = "E_REPLAY"
    EXIT = EXIT_REPLAY
