"""Stable error codes used by the report and API."""

E_SAMPLE_INVALID = "E_SAMPLE_INVALID"
E_BUDGET_PENDING = "E_BUDGET_PENDING"
E_CONFIG_INVALID = "E_CONFIG_INVALID"
E_SNAPSHOT_INVALID = "E_SNAPSHOT_INVALID"


class FusionError(Exception):
    """Base exception carrying a stable diagnostic code."""

    code = "E_FUSION"

    def __init__(self, message: str, *, details: list[str] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = sorted(details or [])


class ConfigError(FusionError):
    code = E_CONFIG_INVALID


class SnapshotError(FusionError):
    code = E_SNAPSHOT_INVALID
