"""全链路业务错误码。"""


class BusinessError(Exception):
    """携带业务码的确定性错误，业务码进入失败清单与诊断报告。"""

    def __init__(self, code: str, message: str, *, stage: str = "",
                 fragment_seq: int | None = None,
                 window_index: int | None = None,
                 injection_id: str | None = None,
                 evidence: str = ""):
        super().__init__(message)
        self.code = code
        self.message = message
        self.stage = stage
        self.fragment_seq = fragment_seq
        self.window_index = window_index
        self.injection_id = injection_id
        self.evidence = evidence

    def to_record(self) -> dict:
        record = {
            "code": self.code,
            "message": self.message,
            "stage": self.stage,
        }
        if self.fragment_seq is not None:
            record["fragment_seq"] = self.fragment_seq
        if self.window_index is not None:
            record["window_index"] = self.window_index
        if self.injection_id is not None:
            record["injection_id"] = self.injection_id
        if self.evidence:
            record["evidence"] = self.evidence
        return record


# RIFF/WAV 容器或音频参数不合规
E_AUDIO_FORMAT = "E_AUDIO_FORMAT"
# 片段序号、到达/采集毫秒、命令词项等元数据不合规
E_FRAGMENT_INVALID = "E_FRAGMENT_INVALID"
# 采样值含 NaN/Inf
E_SAMPLE_NON_FINITE = "E_SAMPLE_NON_FINITE"
# 窗引用的采样区间越界
E_SPAN_OUT_OF_RANGE = "E_SPAN_OUT_OF_RANGE"
# 量化档位不在允许枚举内
E_QUANT_LEVEL_INVALID = "E_QUANT_LEVEL_INVALID"
# 后验输入维度与投影矩阵不符
E_DIMENSION_MISMATCH = "E_DIMENSION_MISMATCH"
# 单批重算预算耗尽，窗保留在待办队列
E_WINDOW_BUDGET_PENDING = "E_WINDOW_BUDGET_PENDING"
