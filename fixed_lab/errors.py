"""Stable business and input-validation codes."""

E_INPUT_INVALID = "E_INPUT_INVALID"
E_BUDGET_DEFERRED = "E_BUDGET_DEFERRED"
E_COLUMN_QUOTA_DEFERRED = "E_COLUMN_QUOTA_DEFERRED"
E_SOLVER_LOW_PIVOT = "E_SOLVER_LOW_PIVOT"
E_SOLVER_NON_CONVERGED = "E_SOLVER_NON_CONVERGED"
E_SOLVER_CURVATURE = "E_SOLVER_CURVATURE"
E_SNAPSHOT_COMMIT = "E_SNAPSHOT_COMMIT"


class InputValidationError(ValueError):
    """Raised when a whole configuration or command stream cannot be loaded."""

    code = E_INPUT_INVALID

    def __init__(self, message: str, *, field: str | None = None, errors: list | None = None):
        super().__init__(message)
        self.field = field
        self.errors = errors or []


class SolverError(RuntimeError):
    """Deterministic solver failure with a stable business code."""

    def __init__(self, code: str, message: str, details: dict | None = None):
        super().__init__(message)
        self.code = code
        self.details = details or {}
