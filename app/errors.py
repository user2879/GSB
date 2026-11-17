"""业务错误码与失败记录。

失败清单中的每一条记录都携带实体 ID、阶段名、业务码与异常类型，
单实体失败不阻断其余实体的处理（失败隔离）。
"""
from __future__ import annotations

from dataclasses import dataclass

E_FIELD_WIDTH = "E_FIELD_WIDTH"
E_PARTITION_OOB = "E_PARTITION_OOB"
E_STEP_UNKNOWN = "E_STEP_UNKNOWN"
E_STEP_MAP_MISSING = "E_STEP_MAP_MISSING"
E_VERSION_UNSUPPORTED = "E_VERSION_UNSUPPORTED"
E_SNAPSHOT_STALE = "E_SNAPSHOT_STALE"
E_PRECISION_MISSING = "E_PRECISION_MISSING"
E_MERGE_CONFLICT = "E_MERGE_CONFLICT"


@dataclass
class Failure:
    entity_id: str | None
    stage: str
    code: str
    exc_type: str
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "entity_id": self.entity_id,
            "stage": self.stage,
            "code": self.code,
            "exc_type": self.exc_type,
            "detail": self.detail,
        }


class PipelineError(Exception):
    """携带业务码的异常，用于失败隔离与清单记录。"""

    def __init__(self, code: str, detail: str = "", entity_id: str | None = None):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail
        self.entity_id = entity_id


class EditRejected(PipelineError):
    """编辑被版本语义约束拒绝。"""


def failure_from_exc(exc: Exception, stage: str, entity_id: str | None = None,
                     code: str | None = None) -> Failure:
    resolved = code or (exc.code if isinstance(exc, PipelineError) else "E_INTERNAL")
    ent = entity_id or (exc.entity_id if isinstance(exc, PipelineError) else None)
    detail = exc.detail if isinstance(exc, PipelineError) else str(exc)
    return Failure(entity_id=ent, stage=stage, code=resolved,
                   exc_type=type(exc).__name__, detail=detail)
