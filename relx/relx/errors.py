"""业务错误码与失败清单记录。

错误码以 E_ 开头，按输入类别划分。单句级错误进入失败清单，不中断其余句子；
配置与模型级错误属于装载阶段致命错误，抛出 RelxError。
"""

E_CONFIG_INVALID = "E_CONFIG_INVALID"
E_MODEL_INVALID = "E_MODEL_INVALID"
E_VOCAB_INVALID = "E_VOCAB_INVALID"
E_ENTITY_REF_INVALID = "E_ENTITY_REF_INVALID"
E_ENTITY_STRUCTURE_INVALID = "E_ENTITY_STRUCTURE_INVALID"
E_RELATION_REF_INVALID = "E_RELATION_REF_INVALID"
E_JSONL_INVALID = "E_JSONL_INVALID"
E_SENTENCE_DUPLICATE = "E_SENTENCE_DUPLICATE"
E_TOKEN_OOV = "E_TOKEN_OOV"
E_FEATURE_BUDGET_PENDING = "E_FEATURE_BUDGET_PENDING"
E_FEATURE_BUDGET_OVERSIZE = "E_FEATURE_BUDGET_OVERSIZE"
E_COMMAND_INVALID = "E_COMMAND_INVALID"


class RelxError(Exception):
    """装载或配置阶段的致命错误，携带业务码与上下文。"""

    def __init__(self, code: str, message: str, context=None):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.context = context


class FailureRecord:
    """失败清单单条记录，顺序由调用方按稳定键排序。"""

    __slots__ = ("code", "sentence_id", "stage", "message", "detail")

    def __init__(self, code, sentence_id, stage, message, detail=None):
        self.code = code
        self.sentence_id = sentence_id
        self.stage = stage
        self.message = message
        self.detail = detail

    def to_dict(self):
        return {
            "code": self.code,
            "sentence_id": self.sentence_id,
            "stage": self.stage,
            "message": self.message,
            "detail": self.detail,
        }

    @staticmethod
    def sort_key(record):
        return (
            record.sentence_id or "",
            record.stage or "",
            record.code,
            record.message,
        )
