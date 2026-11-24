from __future__ import annotations


class MMWindowError(Exception):
    code = "E_MMWIN"


class ConfigError(MMWindowError):
    code = "E_CONFIG_INVALID"


class PlanError(MMWindowError):
    code = "E_PLAN_INVALID"


class CheckpointError(MMWindowError):
    code = "E_CHECKPOINT_INVALID"
