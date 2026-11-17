"""Business error codes and failure records for the time-series workbench."""

from __future__ import annotations


E_CONFIG_INVALID = "E_CONFIG_INVALID"
E_SERIES_INVALID = "E_SERIES_INVALID"
E_CONFIG_MISMATCH = "E_CONFIG_MISMATCH"


class ConfigError(Exception):
    """Fatal configuration error (the run cannot start)."""

    def __init__(self, message: str, code: str = E_CONFIG_INVALID):
        super().__init__(message)
        self.code = code
        self.message = message


class SeriesFailure(Exception):
    """Per-series validation failure collected into the failure list."""

    def __init__(self, series_id: str, message: str, code: str = E_SERIES_INVALID):
        super().__init__(f"{code}: {series_id}: {message}")
        self.series_id = series_id
        self.message = message
        self.code = code

    def to_record(self) -> dict:
        return {
            "series_id": self.series_id,
            "code": self.code,
            "message": self.message,
        }


def sort_failures(records: list[dict]) -> list[dict]:
    """Deterministic failure-list ordering used by every run mode."""
    return sorted(records, key=lambda r: (r["series_id"], r["code"], r["message"]))
