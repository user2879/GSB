from __future__ import annotations

from dataclasses import asdict, dataclass


E_MASK_FORMAT_INVALID = "E_MASK_FORMAT_INVALID"
E_INTERACT_BUDGET_PENDING = "E_INTERACT_BUDGET_PENDING"
E_BOUNDARY_SENSITIVE = "E_BOUNDARY_SENSITIVE"
E_SNAPSHOT_INVALID = "E_SNAPSHOT_INVALID"
E_PARAM_INVALID = "E_PARAM_INVALID"


class PipelineError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class SnapshotRecoveryError(PipelineError):
    def __init__(self, message: str):
        super().__init__(E_SNAPSHOT_INVALID, message)


@dataclass(frozen=True)
class Failure:
    instance_key: str
    code: str
    stage: str
    message: str

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def stable_sort_key(self) -> tuple:
        try:
            source_format, raw_id = self.instance_key.split(":", 1)
            instance_id = int(raw_id)
            key_order = (0, source_format, instance_id, self.instance_key)
        except (TypeError, ValueError):
            key_order = (1, self.instance_key, 0, self.instance_key)
        return (*key_order, self.stage, self.code, self.message)
