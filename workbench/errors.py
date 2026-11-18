"""Business error codes used across the workbench."""

E_DECODE_FAIL = "E_DECODE_FAIL"
E_BUDGET_EXHAUSTED = "E_BUDGET_EXHAUSTED"
E_CONFIG_MISMATCH = "E_CONFIG_MISMATCH"
E_INTERNAL = "E_INTERNAL"

ALL_CODES = (E_DECODE_FAIL, E_BUDGET_EXHAUSTED, E_CONFIG_MISMATCH, E_INTERNAL)


class BusinessError(Exception):
    """Exception carrying a stable business code."""

    def __init__(self, code, message):
        super().__init__(message)
        if code not in ALL_CODES:
            raise ValueError(f"unknown business code: {code}")
        self.code = code
        self.message = message
