"""统一错误码。所有业务错误携带机器可读代码与首个非法定位。"""


class DiagError(Exception):
    def __init__(self, code, message, location=None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.location = location

    def __str__(self):
        if self.location:
            return f"{self.code}: {self.message} (at {self.location})"
        return f"{self.code}: {self.message}"


class SpecError(DiagError):
    def __init__(self, message, location):
        super().__init__("E_SPEC", message, location)


class PortError(DiagError):
    def __init__(self, message, location=None):
        super().__init__("E_PORT", message, location)


class TimeoutError(DiagError):
    def __init__(self, message, location=None):
        super().__init__("E_TIMEOUT", message, location)


class LeaseError(DiagError):
    def __init__(self, message, location=None):
        super().__init__("E_LEASE", message, location)


class BudgetError(DiagError):
    def __init__(self, message, location=None):
        super().__init__("E_BUDGET", message, location)


class ChecksumError(DiagError):
    def __init__(self, message, location=None):
        super().__init__("E_CHECKSUM", message, location)


class VerifyError(DiagError):
    def __init__(self, message, location):
        super().__init__("E_VERIFY", message, location)


class LedgerError(DiagError):
    def __init__(self, message, location=None):
        super().__init__("E_LEDGER", message, location)
