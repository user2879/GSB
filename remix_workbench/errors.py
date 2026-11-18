from __future__ import annotations

from dataclasses import dataclass


E_TRACK_FORMAT = "E_TRACK_FORMAT"
E_MIX_CONFIG = "E_MIX_CONFIG"


@dataclass(frozen=True)
class BusinessError(Exception):
    code: str
    message: str
    track: str | None = None
    injection_id: str | None = None

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"


class TrackFormatError(BusinessError):
    def __init__(
        self,
        message: str,
        *,
        track: str | None = None,
        injection_id: str | None = None,
    ) -> None:
        super().__init__(E_TRACK_FORMAT, message, track, injection_id)


class MixConfigError(BusinessError):
    def __init__(
        self,
        message: str,
        *,
        track: str | None = None,
        injection_id: str | None = None,
    ) -> None:
        super().__init__(E_MIX_CONFIG, message, track, injection_id)

