"""Business error codes used across the workbench.

All codes are stable strings so reports written by older runs remain
readable.  Per-sample failures never abort a run; they are collected in a
deterministically ordered failure list.
"""

E_CONFIG_INVALID = "E_CONFIG_INVALID"
E_SAMPLE_INVALID = "E_SAMPLE_INVALID"
E_REGION_OUT_OF_BOUND = "E_REGION_OUT_OF_BOUND"
E_CONFIG_MISMATCH = "E_CONFIG_MISMATCH"


class WorkbenchError(Exception):
    """Base exception carrying a stable business code and context."""

    def __init__(self, code, message, context=None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.context = context or {}

    def to_record(self):
        return {
            "code": self.code,
            "message": self.message,
            "context": self.context,
        }


class ConfigError(WorkbenchError):
    def __init__(self, message, context=None):
        super().__init__(E_CONFIG_INVALID, message, context)


class SampleError(WorkbenchError):
    def __init__(self, message, context=None, code=E_SAMPLE_INVALID):
        super().__init__(code, message, context)


class ConfigMismatchError(WorkbenchError):
    def __init__(self, message, context=None):
        super().__init__(E_CONFIG_MISMATCH, message, context)
