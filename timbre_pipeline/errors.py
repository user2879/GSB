"""业务错误码与异常类型。

错误码以 ``E_`` 开头，可被序列化进失败清单与诊断报告；
单音频/单段故障以 :class:`PipelineError` 抛出，由流水线捕获隔离，
不影响其余音频继续处理。
"""

E_AUDIO_FORMAT = "E_AUDIO_FORMAT"
E_DIM_QUANT_CONFLICT = "E_DIM_QUANT_CONFLICT"
E_SEGMENT_OVERFLOW = "E_SEGMENT_OVERFLOW"
E_LATENCY_FALLBACK = "E_LATENCY_FALLBACK"
E_INJECT_NON_FINITE = "E_INJECT_NON_FINITE"
E_INJECT_DIM_MISMATCH = "E_INJECT_DIM_MISMATCH"
E_INJECT_THRESHOLD_RANGE = "E_INJECT_THRESHOLD_RANGE"
E_INJECT_SEGMENT_OUT_OF_RANGE = "E_INJECT_SEGMENT_OUT_OF_RANGE"
E_STAGE_FAILED = "E_STAGE_FAILED"
E_SNAPSHOT_VERSION = "E_SNAPSHOT_VERSION"

ALL_CODES = (
    E_AUDIO_FORMAT,
    E_DIM_QUANT_CONFLICT,
    E_SEGMENT_OVERFLOW,
    E_LATENCY_FALLBACK,
    E_INJECT_NON_FINITE,
    E_INJECT_DIM_MISMATCH,
    E_INJECT_THRESHOLD_RANGE,
    E_INJECT_SEGMENT_OUT_OF_RANGE,
    E_STAGE_FAILED,
    E_SNAPSHOT_VERSION,
)


class PipelineError(Exception):
    """携带业务码、阶段名与目标标识的可隔离故障。"""

    def __init__(self, code, message, stage=None, target=None, extra=None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.stage = stage
        self.target = target
        self.extra = dict(extra or {})

    def to_record(self):
        record = {
            "code": self.code,
            "message": str(self.message),
            "stage": self.stage,
            "target": self.target,
        }
        record.update(self.extra)
        return record


def fail_record(code, message, stage=None, target=None, **extra):
    """构造进入失败清单的纯字典记录。"""
    record = {"code": code, "message": message, "stage": stage, "target": target}
    record.update(extra)
    return record
