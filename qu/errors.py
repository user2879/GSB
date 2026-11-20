"""Business error codes and typed pipeline exceptions.

Error codes are part of the persisted report contract and must stay stable.
"""

E_INVALID_JSON = "E_INVALID_JSON"
E_EMPTY_QUERY = "E_EMPTY_QUERY"
E_RULE_DANGLING = "E_RULE_DANGLING"
E_INVALID_WEIGHT = "E_INVALID_WEIGHT"
E_DUPLICATE_QUERY_ID = "E_DUPLICATE_QUERY_ID"
E_CONFIG_MISMATCH = "E_CONFIG_MISMATCH"
E_PIPELINE_STAGE = "E_PIPELINE_STAGE"

ALL_CODES = (
    E_INVALID_JSON,
    E_EMPTY_QUERY,
    E_RULE_DANGLING,
    E_INVALID_WEIGHT,
    E_DUPLICATE_QUERY_ID,
    E_CONFIG_MISMATCH,
    E_PIPELINE_STAGE,
)


class QUError(Exception):
    """Base exception carrying a stable business code."""

    code = E_PIPELINE_STAGE

    def __init__(self, message, code=None, stage=None):
        super().__init__(message)
        if code is not None:
            self.code = code
        self.stage = stage


class InvalidJsonError(QUError):
    code = E_INVALID_JSON


class EmptyQueryError(QUError):
    code = E_EMPTY_QUERY


class RuleDanglingError(QUError):
    code = E_RULE_DANGLING


class InvalidWeightError(QUError):
    code = E_INVALID_WEIGHT


class DuplicateQueryIdError(QUError):
    code = E_DUPLICATE_QUERY_ID


class ConfigMismatchError(QUError):
    code = E_CONFIG_MISMATCH
