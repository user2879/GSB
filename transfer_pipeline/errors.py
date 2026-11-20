"""Business errors used by the pipeline."""


class PipelineError(Exception):
    business_code = "E_PIPELINE"

    def __init__(self, message: str, *, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigError(PipelineError):
    business_code = "E_CONFIG_INVALID"


class PipelineConfigMismatchError(PipelineError):
    business_code = "E_CONFIG_MISMATCH"


class EncoderValidationError(PipelineError):
    business_code = "E_ENCODER_INVALID"


class TaskValidationError(PipelineError):
    business_code = "E_TASK_INVALID"


class AdapterExecutionError(PipelineError):
    business_code = "E_ADAPTER_INVALID"


class RankInvalidError(AdapterExecutionError):
    business_code = "E_RANK_INVALID"


class TrainingExecutionError(AdapterExecutionError):
    business_code = "E_TRAINING_EXCEPTION"


class ValidationExecutionError(AdapterExecutionError):
    business_code = "E_VALIDATION_EXCEPTION"


class SimulatedCrash(PipelineError):
    business_code = "E_SIMULATED_CRASH"

