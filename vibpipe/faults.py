from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class InjectedFault(RuntimeError):
    def __init__(self, injection_id: str, kind: str, batch: int, stage: str, code: str):
        super().__init__(f"injected {kind} fault {injection_id} at batch={batch}, stage={stage}")
        self.injection_id = injection_id
        self.kind = kind
        self.batch = batch
        self.stage = stage
        self.code = code


@dataclass(frozen=True)
class FaultInjector:
    compute_faults: tuple[dict[str, Any], ...] = ()
    disk_faults: tuple[dict[str, Any], ...] = ()

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "FaultInjector":
        if not payload:
            return cls()
        return cls(
            compute_faults=tuple(payload.get("compute_faults", [])),
            disk_faults=tuple(payload.get("disk_faults", [])),
        )

    @property
    def injection_ids(self) -> list[str]:
        return [str(x["injection_id"]) for x in (*self.compute_faults, *self.disk_faults)]

    def check_compute(self, batch: int, stage: str) -> None:
        for spec in self.compute_faults:
            if int(spec["batch"]) == batch and spec["stage"] == stage:
                raise InjectedFault(
                    str(spec["injection_id"]),
                    "compute",
                    batch,
                    stage,
                    str(spec.get("code", "E_BATCH_COMPUTE")),
                )

    def check_disk(self, batch: int, stage: str) -> None:
        for spec in self.disk_faults:
            if int(spec["batch"]) == batch and spec["stage"] == stage:
                raise InjectedFault(
                    str(spec["injection_id"]),
                    "disk",
                    batch,
                    stage,
                    str(spec.get("code", "E_SNAPSHOT_WRITE")),
                )

    def hit_ids(self, failures: list[dict[str, Any]]) -> list[str]:
        return sorted({f["injection_id"] for f in failures if f.get("injection_id")})
