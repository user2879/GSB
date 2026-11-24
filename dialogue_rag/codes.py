"""Stable business codes used in failure and budget lists."""


class Code:
    VALIDATION_ERROR = "E_VALIDATION"
    INVALID_JSON = "E_INVALID_JSON"
    FIELD_TYPE = "E_FIELD_TYPE"
    FIELD_REQUIRED = "E_FIELD_REQUIRED"
    FIELD_NON_FINITE = "E_FIELD_NON_FINITE"
    FIELD_OUT_OF_RANGE = "E_FIELD_OUT_OF_RANGE"
    FIELD_NOT_ALLOWED = "E_FIELD_NOT_ALLOWED"
    TURN_SEQ_GAP = "E_TURN_SEQ_GAP"
    TURN_SEQ_ORDER = "E_TURN_SEQ_ORDER"
    DUPLICATE_ID = "E_DUPLICATE_ID"
    UNKNOWN_REFERENCE = "E_UNKNOWN_REFERENCE"
    UNKNOWN_COMMAND = "E_UNKNOWN_COMMAND"
    COMMAND_TARGET_MISSING = "E_COMMAND_TARGET_MISSING"
    BUDGET_PENDING = "E_BUDGET_PENDING"
    SNAPSHOT_REBUILD = "E_SNAPSHOT_REBUILD"
    SNAPSHOT_RESUME = "I_SNAPSHOT_RESUME"
    STOPPING = "I_STOP_AFTER_CURRENT"


def failure(code, message, *, field=None, session_id=None, turn_id=None, **extra):
    item = {
        "code": code,
        "severity": "error",
        "message": message,
        "session_id": session_id,
        "turn_id": turn_id,
        "field": field,
    }
    item.update(extra)
    return {key: value for key, value in item.items() if value is not None}
