"""Business error codes shared across the workbench."""

E_CONFIG = "E_CONFIG"
E_LEASE = "E_LEASE"
E_SNAPSHOT = "E_SNAPSHOT"
E_PREDICATE = "E_PREDICATE"
E_STATE = "E_STATE"


class BizError(Exception):
    """Deterministic business error carrying a stable code and first offending field."""

    def __init__(self, code, message, field=None, detail=None):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.field = field
        self.detail = detail

    def to_dict(self):
        return {
            "code": self.code,
            "message": self.message,
            "field": self.field,
            "detail": self.detail,
        }
