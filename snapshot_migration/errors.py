"""Domain error hierarchy.

Every rule violation reported by the component carries a stable ``E_*`` code
plus enough location context (generation / fragment / field) to be diagnosed
deterministically from the Tkinter view and the command line.
"""


class MigrationError(Exception):
    code = "E_UNKNOWN"

    def __init__(self, message, *, generation_id=None, fragment_id=None,
                 field=None, range_id=None, **extra):
        super().__init__(message)
        self.message = message
        self.generation_id = generation_id
        self.fragment_id = fragment_id
        self.field = field
        self.range_id = range_id
        self.extra = extra

    @property
    def location(self):
        loc = {}
        if self.generation_id is not None:
            loc["generation_id"] = self.generation_id
        if self.range_id is not None:
            loc["range_id"] = self.range_id
        if self.fragment_id is not None:
            loc["fragment_id"] = self.fragment_id
        if self.field is not None:
            loc["field"] = self.field
        loc.update(self.extra)
        return loc

    def to_dict(self):
        return {"code": self.code, "message": self.message, "location": self.location}

    def __str__(self):
        parts = [self.code, self.message]
        if self.location:
            parts.append("location=" + repr(self.location))
        return " | ".join(parts)


class ChainError(MigrationError):
    """Snapshot generation chain declaration is invalid (whole load rejected)."""

    code = "E_CHAIN"


class BoundError(MigrationError):
    """Fragment boundary / key-count invariant is violated."""

    code = "E_BOUND"


class InheritError(MigrationError):
    """Inheritance source generation or fragment cannot be resolved."""

    code = "E_INHERIT"


class CleanError(MigrationError):
    """Cleanup ledger conflicts with generation pointers / active read views."""

    code = "E_CLEAN"


class CursorError(MigrationError):
    """Two-dimensional cursor can no longer be anchored safely."""

    code = "E_CURSOR"
