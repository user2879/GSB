"""Stable error codes used across the diarization component."""


class DiarizError(Exception):
    """Base error carrying a stable machine readable code."""

    code = "E_INTERNAL"

    def __init__(self, message: str = "", details=None):
        super().__init__(message or self.code)
        self.message = message or self.code
        self.details = details if details is not None else {}


class AudioFormatError(DiarizError):
    code = "E_AUDIO_FORMAT"


class ConfigInvalidError(DiarizError):
    code = "E_CONFIG_INVALID"


class SnapshotConflictError(DiarizError):
    code = "E_SNAPSHOT_CONFLICT"


class InjectedFaultError(DiarizError):
    code = "E_INJECTED_FAULT"


FAIL_CODES = {
    "audio_format": AudioFormatError.code,
    "config_invalid": ConfigInvalidError.code,
    "snapshot_conflict": SnapshotConflictError.code,
    "injected_fault": InjectedFaultError.code,
}
