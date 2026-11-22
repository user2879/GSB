"""Unified error codes for the platform.

E_FORMAT  : block format validation failure (unknown version, missing field,
            bad null bitmap length, unsupported column type, ...).
E_PLAN    : batch split / plan compilation failure.
E_DIVERGE : upgraded block diverges from directly-written current-version block.
E_REPLAY  : ledger replay / recovery divergence.
"""


class ColValError(Exception):
    code = "E_UNKNOWN"

    def __init__(self, message, **ctx):
        self.context = ctx
        detail = ", ".join(f"{k}={v!r}" for k, v in sorted(ctx.items()))
        super().__init__(f"{self.code}: {message}" + (f" [{detail}]" if detail else ""))


class FormatError(ColValError):
    code = "E_FORMAT"


class PlanError(ColValError):
    code = "E_PLAN"


class DivergeError(ColValError):
    code = "E_DIVERGE"


class ReplayError(ColValError):
    code = "E_REPLAY"
