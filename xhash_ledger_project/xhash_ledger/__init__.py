"""跨模态哈希签名与近重复簇账本组件。"""
from .config import Config, parse_config
from .engine import Engine
from .errors import (
    E_CONFIG_INVALID,
    E_SAMPLE_INVALID,
    E_SIGNATURE_BUDGET_PENDING,
    E_SIGNATURE_QUOTA_EXCEEDED,
)

__all__ = [
    "Config",
    "Engine",
    "parse_config",
    "E_CONFIG_INVALID",
    "E_SAMPLE_INVALID",
    "E_SIGNATURE_BUDGET_PENDING",
    "E_SIGNATURE_QUOTA_EXCEEDED",
]
