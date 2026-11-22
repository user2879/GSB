from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .model import InstanceSpec, MeshSpec, RouteRevision


@dataclass(frozen=True)
class WeightRange:
    instance: InstanceSpec
    start: int
    end: int


@dataclass(frozen=True)
class Selection:
    instance: InstanceSpec
    position: int
    total_weight: int
    ranges: tuple[WeightRange, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "instance_id": self.instance.id,
            "position": self.position,
            "total_weight": self.total_weight,
            "range": [
                item.instance.id
                for item in self.ranges
                if item.instance.id == self.instance.id
            ]
            and [
                next(
                    (item.start, item.end)
                    for item in self.ranges
                    if item.instance.id == self.instance.id
                )
            ],
        }


class WeightedSelector:
    """Deterministic weighted target selection over explicit cumulative ranges.

    Instance order is declaration order from the validated spec. A SHA-256
    digest of ``service, request id, attempt number`` gives an integer position
    in the cumulative weight interval. No Python hash, dict order, wall clock or
    request arrival order participates in the decision.
    """

    def __init__(self, spec: MeshSpec, revision: RouteRevision) -> None:
        self.spec = spec
        self.revision = revision
        self._ranges: dict[str, tuple[WeightRange, ...]] = {}
        for service in spec.ordered_services():
            weights = spec.effective_weights(service.id, revision)
            ranges: list[WeightRange] = []
            cumulative = 0
            for instance in service.instances:
                weight = weights.get(instance.id, 0)
                if weight < 0:
                    raise ValueError(f"negative effective weight for {instance.id}")
                start = cumulative
                cumulative += weight
                ranges.append(WeightRange(instance, start, cumulative))
            if cumulative <= 0:
                raise ValueError(f"service {service.id} has no positive effective weight")
            self._ranges[service.id] = tuple(ranges)

    def ranges_for(self, service_id: str) -> tuple[WeightRange, ...]:
        return self._ranges[service_id]

    def total_weight(self, service_id: str) -> int:
        ranges = self._ranges[service_id]
        return ranges[-1].end if ranges else 0

    def select(self, service_id: str, request_id: str, attempt_no: int) -> Selection:
        total = self.total_weight(service_id)
        material = f"{service_id}\x1f{request_id}\x1f{attempt_no}".encode("utf-8")
        digest = hashlib.sha256(material).digest()
        position = int.from_bytes(digest[:8], "big") % total
        ranges = self._ranges[service_id]
        for item in ranges:
            if item.start <= position < item.end:
                return Selection(item.instance, position, total, ranges)
        raise AssertionError("weighted range lookup failed")

