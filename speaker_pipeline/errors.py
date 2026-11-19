"""Business error codes and typed pipeline exceptions."""


class BusinessError(Exception):
    """Base exception carrying a stable business error code."""

    code = "E_INTERNAL"

    def __init__(self, message, *, stage=None, interval_index=None,
                 injection_id=None):
        super().__init__(message)
        self.message = message
        self.stage = stage
        self.interval_index = interval_index
        self.injection_id = injection_id

    def to_record(self, file_name=None):
        record = {
            "file": file_name,
            "interval_index": self.interval_index,
            "stage": self.stage,
            "code": self.code,
            "error_type": type(self).__name__,
            "message": str(self.message),
        }
        if self.injection_id is not None:
            record["injection_id"] = self.injection_id
        return record


class AudioFormatError(BusinessError):
    code = "E_AUDIO_FORMAT"


class NonFiniteSampleError(BusinessError):
    code = "E_NON_FINITE"


class InvalidThresholdError(BusinessError):
    code = "E_INVALID_THRESHOLD"


class OutOfRangeError(BusinessError):
    code = "E_OUT_OF_RANGE"


class FeatureError(BusinessError):
    code = "E_FEATURE"


class SegmentationError(BusinessError):
    code = "E_SEGMENTATION"


class AttributionError(BusinessError):
    code = "E_ATTRIBUTION"


class SnapshotError(BusinessError):
    code = "E_SNAPSHOT"


class CommandError(BusinessError):
    code = "E_COMMAND"
