"""业务错误码与失败清单条目。"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


E_SEED_INVALID = "E_SEED_INVALID"
E_CONFIG_INVALID = "E_CONFIG_INVALID"
E_VOCAB_INVALID = "E_VOCAB_INVALID"
E_PARAMS_INVALID = "E_PARAMS_INVALID"
E_BATCH_INVALID = "E_BATCH_INVALID"
E_CONFIG_MISMATCH = "E_CONFIG_MISMATCH"
E_SNAPSHOT_CORRUPT = "E_SNAPSHOT_CORRUPT"
E_IO_ERROR = "E_IO_ERROR"
OK = "OK"


@dataclass(frozen=True)
class Failure:
    """单条加载/处理失败记录，顺序即追加顺序，保证可复现。"""

    code: str
    source: str
    ref: str
    message: str
    batch_no: int = 0
    stage: str = "load"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
