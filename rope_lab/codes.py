"""Business diagnostic codes used by validation, scheduling and injection."""

E_SCHEMA_UNKNOWN = "E_SCHEMA_UNKNOWN"
E_DIM_MISMATCH = "E_DIM_MISMATCH"
E_BLOCK_INCOMPATIBLE = "E_BLOCK_INCOMPATIBLE"
E_SAMPLE_DUPLICATE = "E_SAMPLE_DUPLICATE"
E_KEY_POSITION_DUPLICATE = "E_KEY_POSITION_DUPLICATE"
E_VALUE_NONFINITE = "E_VALUE_NONFINITE"
E_POSITION_INVALID = "E_POSITION_INVALID"
E_FIELD_INVALID = "E_FIELD_INVALID"
E_JSON_LINE = "E_JSON_LINE"
E_POINT_QUOTA_EXCEEDED = "E_POINT_QUOTA_EXCEEDED"
E_BUDGET_PENDING = "E_BUDGET_PENDING"
E_FIXED_OVERFLOW = "E_FIXED_OVERFLOW"
E_SNAPSHOT_VERSION = "E_SNAPSHOT_VERSION"
E_INJECTION_NOT_HIT = "E_INJECTION_NOT_HIT"
E_STAGE_FAILURE = "E_STAGE_FAILURE"


class LabValidationError(ValueError):
    def __init__(self, code, message, details=None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}

    def as_dict(self):
        return {"code": self.code, "message": self.message, "details": self.details}
