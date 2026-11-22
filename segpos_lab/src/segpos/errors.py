"""业务错误码与失败清单条目。"""
from __future__ import annotations

from dataclasses import dataclass


E_SENTENCE_INVALID = "E_SENTENCE_INVALID"
E_LEXICON_INVALID = "E_LEXICON_INVALID"
E_PROB_TABLE_INVALID = "E_PROB_TABLE_INVALID"
E_CONFIG_INVALID = "E_CONFIG_INVALID"
E_COMMAND_INVALID = "E_COMMAND_INVALID"
E_INTERACT_BUDGET_PENDING = "E_INTERACT_BUDGET_PENDING"
E_SNAPSHOT_RECOVERED = "E_SNAPSHOT_RECOVERED"


@dataclass(frozen=True)
class Failure:
    """进入失败清单的单条记录，顺序由 (source_line, code, key) 稳定决定。"""

    code: str
    key: str
    reason: str
    source: str
    source_line: int

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "key": self.key,
            "reason": self.reason,
            "source": self.source,
            "source_line": self.source_line,
        }
