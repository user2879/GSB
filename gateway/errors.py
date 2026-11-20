"""Error types for the inference gateway diagnostics console."""


class ValidationError(Exception):
    """Raised when a declaration or request stream fails validation.

    Carries the JSON-style field path of the first offending field.
    """

    def __init__(self, path: str, message: str):
        self.path = path
        self.message = message
        super().__init__(f"{path}: {message}")


class ReplayError(Exception):
    """Raised when replay after recovery diverges from the committed ledger.

    Always reported with code E_REPLAY plus the first diverging request
    sequence number and field.
    """

    CODE = "E_REPLAY"

    def __init__(self, request_seq: int, field: str, expected, actual):
        self.request_seq = request_seq
        self.field = field
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"{self.CODE} request_seq={request_seq} field={field} "
            f"expected={expected!r} actual={actual!r}"
        )
