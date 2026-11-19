"""Business error codes for the probe-GI component.

Every abnormal chain raises a ProbeGIError subclass carrying a stable
machine-readable ``code``.  The CLI serialises these to JSON on stderr.
"""

E_PROBE_GRID = "E_PROBE_GRID"
E_SH_ORDER = "E_SH_ORDER"
E_SAMPLE_DIR = "E_SAMPLE_DIR"
E_BUDGET_DENIED = "E_BUDGET_DENIED"
E_SNAPSHOT_STALE = "E_SNAPSHOT_STALE"
E_STRATEGY_DIVERGE = "E_STRATEGY_DIVERGE"
E_GL_CONTEXT = "E_GL_CONTEXT"


class ProbeGIError(Exception):
    code = "E_INTERNAL"

    def __init__(self, message="", **context):
        super().__init__(message or self.code)
        self.message = message or self.code
        self.context = context

    def to_dict(self):
        return {"code": self.code, "message": self.message, "context": self.context}


class ProbeGridError(ProbeGIError):
    code = E_PROBE_GRID


class SHOrderError(ProbeGIError):
    code = E_SH_ORDER


class SampleDirectionError(ProbeGIError):
    code = E_SAMPLE_DIR


class BudgetDeniedError(ProbeGIError):
    code = E_BUDGET_DENIED


class SnapshotStaleError(ProbeGIError):
    code = E_SNAPSHOT_STALE


class StrategyDivergeError(ProbeGIError):
    code = E_STRATEGY_DIVERGE


class GLContextError(ProbeGIError):
    code = E_GL_CONTEXT
