"""Stable error codes for the convergence diagnostics component."""


class ConvError(Exception):
    """Base error carrying a stable machine-readable code."""

    code = "E_CONV"

    def __init__(self, message, **fields):
        super().__init__(message)
        self.message = message
        self.fields = fields

    def to_dict(self):
        payload = {"code": self.code, "message": self.message}
        payload.update(self.fields)
        return payload


class SceneError(ConvError):
    """E_SCENE: scenario / executor-set declaration is invalid."""

    code = "E_SCENE"

    def __init__(self, field, message):
        super().__init__(message, field=field)
        self.field = field


class WireError(ConvError):
    """E_WIRE: datagram failed structural or semantic decode."""

    code = "E_WIRE"

    def __init__(self, offset, message):
        super().__init__(message, offset=offset)
        self.offset = offset


class StaleError(ConvError):
    """E_STALE: pulse round regressed on the executor."""

    code = "E_STALE"


class ReplayError(ConvError):
    """E_REPLAY: replayed conclusion diverges from the committed ledger."""

    code = "E_REPLAY"

    def __init__(self, round_no, executor_id, field, expected, actual):
        super().__init__(
            "replay mismatch at round=%s executor=%s field=%s"
            % (round_no, executor_id, field),
            round=round_no,
            executor_id=executor_id,
            field=field,
            expected=expected,
            actual=actual,
        )
