"""Business error codes and the pipeline exception type.

Error codes are stable identifiers that appear verbatim in failure records
and in the audit report; they must never be renamed.
"""

E_EMPTY_TEXT = "E_EMPTY_TEXT"
E_BAD_JSON = "E_BAD_JSON"
E_DUP_ID = "E_DUP_ID"
E_CONFIG_INVALID = "E_CONFIG_INVALID"
E_RULE_WEIGHT = "E_RULE_WEIGHT"
E_TIER_LIMIT_ZERO = "E_TIER_LIMIT_ZERO"
E_NON_FINITE = "E_NON_FINITE"
E_THRESHOLD_BOUNDARY = "E_THRESHOLD_BOUNDARY"
E_CONFIG_MISMATCH = "E_CONFIG_MISMATCH"
E_STAGE_FAILED = "E_STAGE_FAILED"


class PipelineError(Exception):
    """An exception carrying a stable business code and an optional sample id."""

    def __init__(self, code, message, sample_id=None, stage=None):
        super().__init__(message)
        self.code = code
        self.sample_id = sample_id
        self.stage = stage

    def to_record(self):
        record = {
            "code": self.code,
            "message": str(self),
            "exception_type": type(self).__name__,
        }
        if self.sample_id is not None:
            record["sample_id"] = self.sample_id
        if self.stage is not None:
            record["stage"] = self.stage
        return record
