from __future__ import annotations


OK = "E_OK"
INPUT_INVALID = "E_INPUT_INVALID"
FIELD_TRUNCATED = "E_FIELD_TRUNCATED"
MATCH_BUDGET_PENDING = "E_MATCH_BUDGET_PENDING"
SNAPSHOT_STALE = "E_SNAPSHOT_STALE"
TEMPLATE_VERSION_UNKNOWN = "E_TEMPLATE_VERSION_UNKNOWN"
STAGE_EXCEPTION = "E_STAGE_EXCEPTION"


CODE_DESCRIPTIONS = {
    OK: "ok",
    INPUT_INVALID: "input validation failed",
    FIELD_TRUNCATED: "exchange field is truncated or malformed",
    MATCH_BUDGET_PENDING: "character waits for another template-match budget batch",
    SNAPSHOT_STALE: "committed snapshot does not match current version or config",
    TEMPLATE_VERSION_UNKNOWN: "dot-matrix template version is not registered",
    STAGE_EXCEPTION: "isolated stage exception",
}
