"""领域错误码与失败清单项。

所有加载期 / 运行期可恢复错误统一使用 :class:`ProofreadError` 表达，
并由调用方写入失败清单，保证失败项可回放、可排序、逐字段一致。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


E_EMPTY_TEXT = "E_EMPTY_TEXT"
E_PARSE_INVALID = "E_PARSE_INVALID"
E_RULE_UNKNOWN = "E_RULE_UNKNOWN"
E_CONFIG_INVALID = "E_CONFIG_INVALID"
E_CONFIG_MISMATCH = "E_CONFIG_MISMATCH"
E_DUPLICATE_ID = "E_DUPLICATE_ID"
E_RESOURCE_INVALID = "E_RESOURCE_INVALID"
E_STATE_CORRUPT = "E_STATE_CORRUPT"


class ProofreadError(Exception):
    """携带稳定错误码的领域异常。"""

    def __init__(self, code: str, message: str, *, sample_id: Optional[str] = None,
                 source: str = "") -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.sample_id = sample_id
        self.source = source

    def to_item(self) -> Dict[str, Any]:
        return failure_item(self.code, self.message, self.sample_id, self.source)


@dataclass(frozen=True)
class Failure:
    code: str
    message: str
    sample_id: Optional[str]
    source: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def failure_item(code: str, message: str, sample_id: Optional[str] = None,
                 source: str = "") -> Dict[str, Any]:
    """构造顺序固定的失败清单字典，保证 JSON 字段顺序稳定。"""
    return {
        "code": code,
        "message": message,
        "sample_id": sample_id,
        "source": source,
    }
