"""Business error codes and exception hierarchy.

Two validation codes are required by the specification:

* ``E_AUDIO_FORMAT`` - the RIFF/WAV container or its PCM payload is malformed
  (bad header, wrong rate/channels/bit depth, data length mismatch, ...).
* ``E_FIELD_INVALID`` - a structural field is missing or out of the allowed
  range/enumeration (negative timestamps, bad frame geometry, bad thresholds,
  unsupported fixed-point width or quantization mode, ...).

``E_STALE_RESULT`` is *not* an exceptional condition: it is returned on a
normal response object to distinguish a stale summary from a fresh one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


# Canonical, stable business codes (never renumber: ledgers reference them).
E_AUDIO_FORMAT = "E_AUDIO_FORMAT"
E_FIELD_INVALID = "E_FIELD_INVALID"
E_STALE_RESULT = "E_STALE_RESULT"
E_INJECTION = "E_INJECTION"
E_NOT_FOUND = "E_NOT_FOUND"

VAD_ERROR_CODES = (
    E_AUDIO_FORMAT,
    E_FIELD_INVALID,
    E_STALE_RESULT,
    E_INJECTION,
    E_NOT_FOUND,
)


@dataclass
class FailureRecord:
    """A single, ledger-worthy validation or injection failure.

    ``item_id`` is the offending segment/frame/window identifier (or ``None``
    when the failure is configuration level).  ``seq`` is the deterministic,
    zero-based order in which the failure was observed.
    """

    seq: int
    code: str
    stage: str
    message: str
    item_id: Optional[str] = None
    injection_id: Optional[str] = None
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "seq": self.seq,
            "code": self.code,
            "stage": self.stage,
            "message": self.message,
        }
        if self.item_id is not None:
            out["item_id"] = self.item_id
        if self.injection_id is not None:
            out["injection_id"] = self.injection_id
        if self.detail:
            out["detail"] = self.detail
        return out


class VADError(Exception):
    """Base class for all VAD component errors."""

    code = E_NOT_FOUND


class VADValidationError(VADError):
    """Raised for unrecoverable configuration / field validation problems."""

    def __init__(self, message: str, code: str = E_FIELD_INVALID,
                 item_id: Optional[str] = None,
                 stage: str = "load", detail: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.code = code
        self.item_id = item_id
        self.stage = stage
        self.detail = detail or {}


class VADFieldError(VADError):
    """Raised when a single segment field is invalid while collecting records.

    The loader converts these into :class:`FailureRecord` entries rather than
    aborting the whole run, so a bad segment is isolated instead of silently
    dropped.
    """

    def __init__(self, message: str, code: str = E_FIELD_INVALID,
                 item_id: Optional[str] = None, stage: str = "load",
                 detail: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.code = code
        self.item_id = item_id
        self.stage = stage
        self.detail = detail or {}


class VADInjectionError(VADError):
    """Raised by an active injection to simulate a real input fault."""

    def __init__(self, message: str, injection_id: str, stage: str,
                 item_id: Optional[str] = None,
                 code: str = E_INJECTION,
                 detail: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.code = code
        self.injection_id = injection_id
        self.stage = stage
        self.item_id = item_id
        self.detail = detail or {}
