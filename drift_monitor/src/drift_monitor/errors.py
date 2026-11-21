"""Business error codes for the drift monitoring component."""


class DriftError(Exception):
    """Base error carrying a stable business code and offending field."""

    code = "E_INTERNAL"

    def __init__(self, message: str, field: str | None = None):
        super().__init__(message)
        self.field = field
        self.message = message

    def to_dict(self) -> dict:
        return {"code": self.code, "field": self.field, "message": self.message}


class ConfigInvalidError(DriftError):
    code = "E_CONFIG_INVALID"


class DataGenError(DriftError):
    code = "E_DATA_GEN_FAILED"


class SnapshotStaleError(DriftError):
    code = "E_SNAPSHOT_STALE"


class SnapshotCorruptError(DriftError):
    code = "E_SNAPSHOT_CORRUPT"


class MetricFailedError(DriftError):
    code = "E_METRIC_FAILED"
