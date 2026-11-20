"""业务码与异常定义。"""

BUSINESS_CODES = {
    "E_CONFIG_INVALID": "配置或特征分组表非法",
    "E_RECORD_INVALID": "候选记录非法",
    "E_CONFIG_MISMATCH": "快照版本或配置哈希不一致，拒绝续跑",
    "E_INTERACT_BUDGET_PENDING": "交互延迟预算内无法完成，返回中间状态",
}


class RerankError(Exception):
    """所有领域异常的基类。"""

    code = "E_INTERNAL"

    def __init__(self, message: str, *, details=None):
        super().__init__(message)
        self.message = message
        self.details = details if details is not None else {}

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "details": self.details}


class ConfigInvalidError(RerankError):
    code = "E_CONFIG_INVALID"


class RecordInvalidError(RerankError):
    code = "E_RECORD_INVALID"


class ConfigMismatchError(RerankError):
    code = "E_CONFIG_MISMATCH"


class BudgetPendingError(RerankError):
    code = "E_INTERACT_BUDGET_PENDING"
