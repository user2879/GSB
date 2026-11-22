"""业务错误码与管线异常定义。

所有可追溯的失败最终都会映射到这里的稳定业务码字符串，写入诊断报告。
"""

E_AUDIO_FORMAT = "E_AUDIO_FORMAT"
E_AUDIO_TRUNCATED = "E_AUDIO_TRUNCATED"
E_NONFINITE_SAMPLE = "E_NONFINITE_SAMPLE"
E_THRESHOLD_ZERO = "E_THRESHOLD_ZERO"
E_RANGE_VIOLATION = "E_RANGE_VIOLATION"
E_TRUTH_MISMATCH = "E_TRUTH_MISMATCH"
E_SNAPSHOT_STALE = "E_SNAPSHOT_STALE"


class PipelineError(Exception):
    """携带业务码与处理阶段的管线异常。"""

    def __init__(self, code, stage, message, injection_id=None):
        super().__init__(message)
        self.code = code
        self.stage = stage
        self.message = message
        self.injection_id = injection_id

    def to_failure(self, audio_name):
        return {
            "audio": audio_name,
            "stage": self.stage,
            "code": self.code,
            "message": self.message,
            "injection_id": self.injection_id,
        }
