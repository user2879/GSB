"""业务码与管线异常定义。"""

E_IMAGE_FORMAT = "E_IMAGE_FORMAT"          # 图像容器/尺寸/分块不合法
E_SAMPLE_INVALID = "E_SAMPLE_INVALID"      # 样本描述不合法或特征值非有限
E_CONFIG_INVALID = "E_CONFIG_INVALID"      # 配置不合法
E_BUDGET_EXHAUSTED = "E_BUDGET_EXHAUSTED"  # 预算/配额耗尽
E_CONFIG_MISMATCH = "E_CONFIG_MISMATCH"    # 快照版本或配置哈希不一致

ALL_CODES = (
    E_IMAGE_FORMAT,
    E_SAMPLE_INVALID,
    E_CONFIG_INVALID,
    E_BUDGET_EXHAUSTED,
    E_CONFIG_MISMATCH,
)

# 阶段固定顺序（用于失败清单与排序键的确定性）
STAGES = ("load_validate", "feature_extract", "fusion_score", "summarize")


class PipelineError(Exception):
    """携带业务码与阶段名的可恢复样本级异常。"""

    def __init__(self, code, stage, message, injection_id=None):
        super().__init__(message)
        self.code = code
        self.stage = stage
        self.message = message
        self.injection_id = injection_id

    def to_record(self, sample_id):
        return {
            "sample_id": sample_id,
            "stage": self.stage,
            "code": self.code,
            "message": self.message,
            "injection_id": self.injection_id,
        }
