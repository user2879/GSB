"""Business and pipeline errors used by the transfer pipeline."""


class PipelineError(Exception):
    """Base class for errors that carry a stable business code."""

    code = "E_PIPELINE"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        if code is not None:
            self.code = code


class PipelineConfigError(PipelineError):
    code = "E_CONFIG_INVALID"


class TaskDescriptionError(PipelineError):
    code = "E_TASK_DESC_INVALID"


class AdapterConfigError(PipelineError):
    code = "E_ADAPTER_CONFIG_INVALID"


class FeatureLibraryError(PipelineError):
    code = "E_FEATURE_LIBRARY_INVALID"


class EncoderError(PipelineError):
    code = "E_ENCODER_INVALID"


class SnapshotError(PipelineError):
    code = "E_SNAPSHOT_INVALID"


class SnapshotMismatchError(PipelineError):
    code = "E_CONFIG_MISMATCH"


class TrainingError(PipelineError):
    code = "E_TRAINING_FAILED"


class EvaluationError(PipelineError):
    code = "E_EVALUATION_FAILED"


class SharedForwardError(PipelineError):
    code = "E_SHARED_FORWARD_FAILED"
