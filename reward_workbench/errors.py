"""业务错误码与确定性失败清单条目。

所有错误均以结构化条目进入失败清单，绝不抛出后中断整批处理。
失败清单始终按稳定排序键排列，保证并发/串行结果逐字段一致。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

# ---- 业务错误码枚举（题面约定 + 工作台补充） ----
E_PAIR_INVALID = "E_PAIR_INVALID"
E_CONFIG_INVALID = "E_CONFIG_INVALID"
E_TRAIN_BUDGET_EXHAUSTED = "E_TRAIN_BUDGET_EXHAUSTED"
E_SNAPSHOT_STALE = "E_SNAPSHOT_STALE"
# 命令流引用的对 ID 不存在（替换目标缺失等）
E_PAIR_NOT_FOUND = "E_PAIR_NOT_FOUND"
# 命令流本身结构非法
E_COMMAND_INVALID = "E_COMMAND_INVALID"

ALL_ERROR_CODES = (
    E_PAIR_INVALID,
    E_CONFIG_INVALID,
    E_TRAIN_BUDGET_EXHAUSTED,
    E_SNAPSHOT_STALE,
    E_PAIR_NOT_FOUND,
    E_COMMAND_INVALID,
)


@dataclass(frozen=True)
class FailureEntry:
    """单条失败记录。注入编号 injection_id 用于异常注入审计。"""

    sort_key: str          # 稳定排序键（通常为 pair_id/注入项/阶段）
    code: str
    stage: str             # load_config / load_pair / boundary / snapshot ...
    message: str
    injection_id: str = ""  # 异常注入项编号，非注入场景为空
    context: str = ""      # 额外可审计上下文（命令序号、行号等）

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FailureLog:
    """确定性失败清单：去重 + 固定排序。"""

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str, str], FailureEntry] = {}

    def add(self, entry: FailureEntry) -> None:
        key = (entry.sort_key, entry.code, entry.stage)
        if key not in self._entries:
            self._entries[key] = entry

    def entries(self) -> list[FailureEntry]:
        return sorted(
            self._entries.values(),
            key=lambda e: (e.sort_key, e.stage, e.code, e.injection_id, e.context),
        )

    def codes(self) -> list[str]:
        return [e.code for e in self.entries()]

    def to_list(self) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self.entries()]

    def __len__(self) -> int:
        return len(self._entries)

    def has(self, code: str) -> bool:
        return any(e.code == code for e in self._entries.values())
