"""Business errors used by the deterministic layout pipeline."""


class BusinessError(Exception):
    """Base class for user-facing business failures."""

    code = "E_BUSINESS"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code


class ImageInvalidError(BusinessError):
    code = "E_IMAGE_INVALID"


class TruthInvalidError(BusinessError):
    code = "E_TRUTH_INVALID"


class ConfigError(BusinessError):
    code = "E_CONFIG_INVALID"


class MatchBudgetPending(BusinessError):
    code = "E_MATCH_BUDGET_PENDING"


class SnapshotStale(BusinessError):
    code = "E_SNAPSHOT_STALE"
