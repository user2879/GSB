from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .errors import RuleError
from .rules import InstanceRule, RequestSpec, RuleSet


@dataclass(frozen=True)
class Selection:
    target: InstanceRule
    ordinal: int
    algorithm: str = "smooth-weighted-round-robin-v1"


class SmoothWeightedRoundRobin:
    """A deterministic, explicitly ordered smooth weighted round-robin picker.

    Instances are sorted by port before any state is updated. Ties are resolved
    by the sorted position, never by dict/hash iteration order.
    """

    ALGORITHM = "smooth-weighted-round-robin-v1"

    def __init__(self, instances: Sequence[InstanceRule]) -> None:
        if not instances:
            raise RuleError("at least one instance is required", "$.instances")
        self.instances: tuple[InstanceRule, ...] = tuple(
            sorted(instances, key=lambda item: (item.port, item.name))
        )
        eligible = [item for item in self.instances if item.weight > 0]
        if not eligible:
            raise RuleError(
                "sum of instance weights must be greater than zero", "$.instances"
            )
        self._eligible = tuple(eligible)
        self._current = [0 for _ in self._eligible]
        self._total_weight = sum(item.weight for item in self._eligible)
        self._ordinal = 0

    @property
    def total_weight(self) -> int:
        return self._total_weight

    def choose(self) -> Selection:
        for index, instance in enumerate(self._eligible):
            self._current[index] += instance.weight
        best_index = 0
        # The explicit max comparison with sorted positions resolves ties.
        for index in range(1, len(self._eligible)):
            if self._current[index] > self._current[best_index]:
                best_index = index
        self._current[best_index] -= self._total_weight
        selection = Selection(
            target=self._eligible[best_index],
            ordinal=self._ordinal,
            algorithm=self.ALGORITHM,
        )
        self._ordinal += 1
        return selection


class SelectionPlanner:
    """Precomputes every attempt target for a round in stable attempt layers.

    Requests are ordered by their input index. Attempt layer 1 is assigned for
    every request before layer 2, which mirrors a sequential first-attempt
    round and avoids biasing all initial requests toward the first SWRR slot.
    This is the key to deterministic parallel execution: thread scheduling can
    never alter weighted-selection ordinals. An attempt that is never executed
    simply leaves an unused slot in the per-service deterministic plan.
    """

    def __init__(self, rules: RuleSet, requests: Iterable[RequestSpec]) -> None:
        request_list = tuple(requests)
        selectors = {
            service.name: SmoothWeightedRoundRobin(service.instances)
            for service in rules.services
        }
        plans: dict[tuple[str, int], Selection] = {}
        by_service: dict[str, list[RequestSpec]] = {
            service.name: [] for service in rules.services
        }
        for request in request_list:
            by_service[request.route.service].append(request)
        for service_name in sorted(by_service):
            selector = selectors[service_name]
            service_requests = sorted(by_service[service_name], key=lambda item: item.index)
            max_attempts = max(
                (request.route.max_retries + 1 for request in service_requests),
                default=0,
            )
            for attempt_index in range(max_attempts):
                for request in service_requests:
                    if attempt_index <= request.route.max_retries:
                        plans[(request.request_id, attempt_index)] = selector.choose()
        self._plans = plans
        self.algorithm = SmoothWeightedRoundRobin.ALGORITHM

    def get(self, request_id: str, attempt_index: int) -> Selection:
        try:
            return self._plans[(request_id, attempt_index)]
        except KeyError as exc:
            raise KeyError(
                f"no selection plan for {request_id!r} attempt {attempt_index}"
            ) from exc

    def as_dict(self) -> dict[str, dict[str, str | int]]:
        return {
            f"{request_id}#{attempt_index}": {
                "instance": selection.target.name,
                "port": selection.target.port,
                "ordinal": selection.ordinal,
                "algorithm": selection.algorithm,
            }
            for (request_id, attempt_index), selection in sorted(
                self._plans.items(), key=lambda item: (item[0][0], item[0][1])
            )
        }
