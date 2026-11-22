"""Business error codes and exception hierarchy.

Every input fault or pipeline failure is mapped to a stable business code so
that failure records can be compared field-by-field across runs.
"""

E_SAMPLE_FORMAT = "E_SAMPLE_FORMAT"
E_LABEL_UNKNOWN = "E_LABEL_UNKNOWN"
E_BBOX_OUT_OF_RANGE = "E_BBOX_OUT_OF_RANGE"
E_CONFIG = "E_CONFIG"
E_FEATURE_BUDGET_PENDING = "E_FEATURE_BUDGET_PENDING"
E_STAGE = "E_STAGE"

SAMPLE_ERROR_CODES = (
    E_SAMPLE_FORMAT,
    E_LABEL_UNKNOWN,
    E_BBOX_OUT_OF_RANGE,
)


class PipelineError(Exception):
    """Base class for all pipeline errors carrying a business code."""

    code = E_STAGE

    def __init__(self, message, *, code=None, stage="", sample="", injection_id=""):
        super().__init__(message)
        if code is not None:
            self.code = code
        self.stage = stage
        self.sample = sample
        self.injection_id = injection_id


class SampleError(PipelineError):
    """A single-sample input failure; it must never abort the whole batch."""

    code = E_SAMPLE_FORMAT


class ConfigError(PipelineError):
    """Invalid pipeline configuration."""

    code = E_CONFIG


class BudgetPending(PipelineError):
    """Raised by the scheduler when the per-batch feature budget is exhausted."""

    code = E_FEATURE_BUDGET_PENDING


class StageError(PipelineError):
    """Unexpected failure inside a stage for one sample/task."""

    code = E_STAGE
