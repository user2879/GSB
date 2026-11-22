"""Shared request/attempt/result value objects."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Request:
    request_id: str
    route_id: str
    client_key: str
    sequence: int


@dataclass
class AttemptRecord:
    attempt: int
    instance_id: str
    started_mono_ms: int
    elapsed_ms: int
    outcome: str          # success | slow | failure | timeout
    error_code: str = ""  # E_CIRCUIT / E_UPSTREAM / ""
    detail: str = ""


@dataclass
class RequestResult:
    request_id: str
    route_id: str
    client_key: str
    sequence: int
    service_id: str
    instance_id: str
    final_outcome: str
    attempts: list = field(default_factory=list)
    total_elapsed_ms: int = 0
    started_clock_ms: int = 0
    finished_clock_ms: int = 0

    def as_ledger_dict(self):
        return {
            "request_id": self.request_id,
            "route_id": self.route_id,
            "client_key": self.client_key,
            "sequence": self.sequence,
            "service_id": self.service_id,
            "instance_id": self.instance_id,
            "final_outcome": self.final_outcome,
            "attempts": [vars(item) for item in self.attempts],
            "total_elapsed_ms": self.total_elapsed_ms,
            "started_clock_ms": self.started_clock_ms,
            "finished_clock_ms": self.finished_clock_ms,
        }
