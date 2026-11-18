"""Business error codes and structured failure records.

All failure paths in the workbench produce a stable, code-driven record so that
runs are reproducible and the diagnostic report can be compared field by field.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict


E_LOG_INVALID = "E_LOG_INVALID"
E_HASH_CONFIG_INVALID = "E_HASH_CONFIG_INVALID"
E_CONFIG_MISMATCH = "E_CONFIG_MISMATCH"
E_ATTRIBUTION_DRIFT = "E_ATTRIBUTION_DRIFT"

ALL_CODES = (
    E_LOG_INVALID,
    E_HASH_CONFIG_INVALID,
    E_CONFIG_MISMATCH,
    E_ATTRIBUTION_DRIFT,
)


@dataclass(frozen=True)
class FailureRecord:
    """One deterministic failure entry.

    Ordering is by (stage, code, request_id, item_id, reason) so that concurrent
    workers can be merged into an identical sequence regardless of scheduling.
    """

    stage: str
    code: str
    reason: str
    request_id: str = ""
    item_id: str = ""
    group: str = ""
    injection_id: str = ""
    detail: str = ""

    def sort_key(self) -> tuple:
        return (
            self.stage,
            self.code,
            self.request_id,
            self.item_id,
            self.group,
            self.reason,
            self.injection_id,
            self.detail,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
