"""Business error codes and typed exceptions.

All codes are stable strings shared by the failure list, the CLI and the
verification report (requirement 8).
"""

E_CAND_FORMAT = "E_CAND_FORMAT"
E_CONFIG_MISMATCH = "E_CONFIG_MISMATCH"
E_IMAGE_TRUNCATED = "E_IMAGE_TRUNCATED"
E_WEIGHT_NONFINITE = "E_WEIGHT_NONFINITE"
E_CLASS_UNKNOWN = "E_CLASS_UNKNOWN"
E_STAGE_FAILED = "E_STAGE_FAILED"

ALL_CODES = (
    E_CAND_FORMAT,
    E_CONFIG_MISMATCH,
    E_IMAGE_TRUNCATED,
    E_WEIGHT_NONFINITE,
    E_CLASS_UNKNOWN,
    E_STAGE_FAILED,
)


class DetboxError(Exception):
    """Base error carrying a stable business code."""

    code = E_STAGE_FAILED

    def __init__(self, message: str, *, code: str | None = None):
        super().__init__(message)
        if code is not None:
            self.code = code


class CandidateFormatError(DetboxError):
    code = E_CAND_FORMAT


class ConfigMismatchError(DetboxError):
    code = E_CONFIG_MISMATCH


class ImageTruncatedError(DetboxError):
    code = E_IMAGE_TRUNCATED


class WeightNonFiniteError(DetboxError):
    code = E_WEIGHT_NONFINITE


class UnknownClassError(DetboxError):
    code = E_CLASS_UNKNOWN
