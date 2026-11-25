from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Failure:
    code: str
    message: str
    scope: str
    field: str | None = None
    batch_seq: int | None = None
    log_id: str | None = None
    injection_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass
class AcceptedRecord:
    log_id: str
    timestamp: int
    device_id: str
    severity: str
    template_id: str
    metric: float
    arrival_seq: int
    record_index: int
    injection_ids: tuple[str, ...] = ()

    @property
    def bucket_id(self) -> str:
        return f"{self.device_id}|{self.severity}"

    @property
    def order_key(self) -> tuple[int, int, str]:
        return (self.arrival_seq, self.record_index, self.log_id)

    def content_dict(self) -> dict[str, Any]:
        return {
            "log_id": self.log_id,
            "timestamp": self.timestamp,
            "device_id": self.device_id,
            "severity": self.severity,
            "template_id": self.template_id,
            "metric": self.metric,
        }


@dataclass
class ValidationResult:
    accepted_records: list[AcceptedRecord]
    failures: list[Failure]
    accepted_batches: list[int]
    rejected_batches: list[int]
    completed_batch_cursor: int
    injections_applied: list[dict[str, Any]]
    config_issues: list[Failure]

    @property
    def config_valid(self) -> bool:
        return not self.config_issues

