"""Typed pipeline errors with stable, deterministic formatting."""


class PipeError(Exception):
    """Base error. Carries optional stable locators (node/key/field)."""

    code = "E_PIPE"

    def __init__(self, message, *, key=None, field=None, node=None):
        super().__init__(message)
        self.message = message
        self.key = key
        self.field = field
        self.node = node

    def __str__(self):
        loc = []
        if self.node is not None:
            loc.append("node=%s" % self.node)
        if self.key is not None:
            loc.append("key=%s" % self.key)
        if self.field is not None:
            loc.append("field=%s" % self.field)
        suffix = (" [" + ", ".join(loc) + "]") if loc else ""
        return "%s: %s%s" % (self.code, self.message, suffix)


class SpecError(PipeError):
    """Declaration / changeset validation failure."""

    code = "E_SPEC"


class ConvergeError(PipeError):
    """Receipt convergence check failure."""

    code = "E_CONVERGE"


class ReplayError(PipeError):
    """Post-recovery replay verification failure."""

    code = "E_REPLAY"


class DispatchError(PipeError):
    """Dispatch transport failure after all retries."""

    code = "E_DISPATCH"
