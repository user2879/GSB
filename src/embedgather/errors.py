"""Business error codes shared across the component."""

E_EMBED_INVALID = "E_EMBED_INVALID"
E_ID_MISSING = "E_ID_MISSING"
E_SNAPSHOT_MISMATCH = "E_SNAPSHOT_MISMATCH"
E_POINT_BUDGET_EXHAUSTED = "E_POINT_BUDGET_EXHAUSTED"

ALL_CODES = (
    E_EMBED_INVALID,
    E_ID_MISSING,
    E_SNAPSHOT_MISMATCH,
    E_POINT_BUDGET_EXHAUSTED,
)


class BusinessError(Exception):
    """An error carrying a stable business code."""

    def __init__(self, code, detail=None):
        if code not in ALL_CODES:
            raise ValueError(f"unknown business code: {code}")
        self.code = code
        self.detail = detail or {}
        super().__init__(f"{code}: {self.detail}")


class GracefulExit(Exception):
    """Raised when a handleable exit signal was received and the current
    stage artifacts have been committed atomically."""

    def __init__(self, progress):
        self.progress = progress
        super().__init__(f"graceful exit at {progress}")
