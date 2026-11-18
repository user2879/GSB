"""业务错误码与管线异常定义。"""


class ErrorCode:
    SAMPLE_INVALID = "E_SAMPLE_INVALID"
    RETRIEVAL_LOW_SCORE = "E_RETRIEVAL_LOW_SCORE"
    FORMAT_LOSSY = "E_FORMAT_LOSSY"

    ALL = (SAMPLE_INVALID, RETRIEVAL_LOW_SCORE, FORMAT_LOSSY)


class PipelineError(Exception):
    """携带业务码与阶段名的管线异常。"""

    def __init__(self, code, message, exc_type=None):
        super().__init__(message)
        self.code = code
        self.exc_type = exc_type or type(self).__name__

    def to_record(self, sample, stage):
        return {
            "sample": sample,
            "stage": stage,
            "code": self.code,
            "exc_type": self.exc_type,
            "message": str(self),
        }


def wrap_exception(code, exc):
    """把底层异常包装成携带业务码的 PipelineError。"""
    return PipelineError(code, str(exc), type(exc).__name__)
