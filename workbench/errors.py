from __future__ import annotations


class WorkbenchError(Exception):
    code = "E_WORKBENCH"

    def __init__(self, message: str, *, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def as_dict(self) -> dict:
        return {"code": self.code, "message": self.message, **self.details}


class ConfigError(WorkbenchError):
    code = "E_CONFIG_INVALID"


class ShardError(WorkbenchError):
    code = "E_SHARD_INVALID"


class SnapshotStaleError(WorkbenchError):
    code = "E_SNAPSHOT_STALE"


class PlanError(WorkbenchError):
    code = "E_PLAN_INVALID"
