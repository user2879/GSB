"""业务异常码与统一异常类型。"""


class DepEditError(Exception):
    """所有可预期业务异常的基类。"""

    code = "E_UNKNOWN"

    def __init__(self, message: str, *, code: str | None = None):
        super().__init__(message)
        if code is not None:
            self.code = code


class EHeadStructure(DepEditError):
    """头词结构非法：越界 / 成环 / 根数量不为一。"""

    code = "E_HEAD_STRUCTURE"


class ESnapshotStale(DepEditError):
    """快照清单与当前统一表示不一致，拒绝覆盖。"""

    code = "E_SNAPSHOT_STALE"


class EFixedWidthField(DepEditError):
    """定宽行字段缺失或无法按宽度切分。"""

    code = "E_FIXED_WIDTH_FIELD"


class EHeadOutOfRange(DepEditError):
    """头词序号越界。"""

    code = "E_HEAD_OUT_OF_RANGE"


class EScoreRange(DepEditError):
    """千分比得分超出 [0, 1000]。"""

    code = "E_SCORE_RANGE"


class ERelationUnknown(DepEditError):
    """关系名不在映射表内。"""

    code = "E_RELATION_UNKNOWN"


class EScoreNonFinite(DepEditError):
    """v2 浮点得分为非有限数。"""

    code = "E_SCORE_NON_FINITE"


class ECycleDetected(DepEditError):
    """句子出现环依赖。"""

    code = "E_CYCLE_DETECTED"


class EMultipleRoots(DepEditError):
    """句子根节点数量不为一。"""

    code = "E_MULTIPLE_ROOTS"


CODE_TO_EXCEPTION: dict[str, type[DepEditError]] = {
    cls.code: cls
    for cls in (
        EHeadStructure,
        ESnapshotStale,
        EFixedWidthField,
        EHeadOutOfRange,
        EScoreRange,
        ERelationUnknown,
        EScoreNonFinite,
        ECycleDetected,
        EMultipleRoots,
    )
}
