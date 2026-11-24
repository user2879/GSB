"""Stable business error codes for the news summarization pipeline."""

E_INPUT_INVALID = "E_INPUT_INVALID"
E_REWRITE_RULE_INVALID = "E_REWRITE_RULE_INVALID"
E_SNAPSHOT_STALE = "E_SNAPSHOT_STALE"
E_SENTENCE_BUDGET_PENDING = "E_SENTENCE_BUDGET_PENDING"

ALL_CODES = (
    E_INPUT_INVALID,
    E_REWRITE_RULE_INVALID,
    E_SNAPSHOT_STALE,
    E_SENTENCE_BUDGET_PENDING,
)


class PipelineError(Exception):
    """Error carrying a stable business code and the offending object id."""

    def __init__(self, code, message, object_id=None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.object_id = object_id

    def to_record(self):
        record = {"code": self.code, "message": self.message}
        if self.object_id is not None:
            record["object_id"] = self.object_id
        return record
