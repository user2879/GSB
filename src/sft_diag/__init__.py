"""确定性指令微调样本构造与增量失效诊断工作台。"""

__version__ = "1.0.0"

from .errors import (  # noqa: F401
    E_BATCH_INVALID,
    E_CONFIG_INVALID,
    E_CONFIG_MISMATCH,
    E_PARAMS_INVALID,
    E_SEED_INVALID,
    E_SNAPSHOT_CORRUPT,
    E_VOCAB_INVALID,
    OK,
)
from .runner import run_workbench  # noqa: F401

__all__ = [
    "E_BATCH_INVALID",
    "E_CONFIG_INVALID",
    "E_CONFIG_MISMATCH",
    "E_PARAMS_INVALID",
    "E_SEED_INVALID",
    "E_SNAPSHOT_CORRUPT",
    "E_VOCAB_INVALID",
    "OK",
    "run_workbench",
]
