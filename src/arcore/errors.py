"""Typed error codes used across the alert correlation component."""


class ArcoreError(Exception):
    """Base class carrying a stable machine readable error code."""

    code = "E_INTERNAL"

    def __init__(self, message: str, *, source_id: str | None = None,
                 field: str | None = None):
        super().__init__(message)
        self.source_id = source_id
        self.field = field

    def __str__(self) -> str:  # pragma: no cover - trivial formatting
        loc = ""
        if self.source_id is not None or self.field is not None:
            loc = f" [source={self.source_id!r}, field={self.field!r}]"
        return f"{self.code}: {self.args[0]}{loc}"


class SpecError(ArcoreError):
    """Raised for the first declaration / stream specification violation."""

    code = "E_SPEC"


class LeaseError(ArcoreError):
    """Raised on lease cap, optimistic version, holder or expiry violations."""

    code = "E_LEASE"


class DivergeError(ArcoreError):
    """Raised when exact and approximate correlation paths disagree."""

    code = "E_DIVERGE"

    def __init__(self, message: str, *, pair: tuple[str, str] | None = None):
        super().__init__(message)
        self.pair = pair


class LedgerError(ArcoreError):
    """Raised on ledger format or integrity problems."""

    code = "E_LEDGER"
