"""Plain data models for scenes, observations and per-round conclusions."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

ROLE_PRIMARY = "primary"
ROLE_REPLICA = "replica"
ROLES = (ROLE_PRIMARY, ROLE_REPLICA)

STATE_UNKNOWN = "unknown"
STATE_HEALTHY = "healthy"
STATE_UNHEALTHY = "unhealthy"
STATES = (STATE_UNKNOWN, STATE_HEALTHY, STATE_UNHEALTHY)

OBS_SUCCESS = "success"
OBS_REFUSED = "connection_refused"
OBS_TIMEOUT = "timeout"
OBS_ERROR = "error_return"
OBSERVATIONS = (OBS_SUCCESS, OBS_REFUSED, OBS_TIMEOUT, OBS_ERROR)
FAILURE_OBS = frozenset((OBS_REFUSED, OBS_TIMEOUT, OBS_ERROR))

STRATEGY_AUTO = "auto"
STRATEGY_MANUAL = "manual"
STRATEGIES = (STRATEGY_AUTO, STRATEGY_MANUAL)

CODE_OK = "OK"
CODE_MANUAL = "E_MANUAL"
CODE_COOLDOWN = "E_COOLDOWN"
CODE_NO_TARGET = "E_NO_TARGET"
CODE_SWITCHED = "SWITCHED"
CODE_BUSY = "E_BUSY"


@dataclass(frozen=True)
class ServiceConfig:
    service_id: str
    strategy: str
    cooldown_ms: int
    min_healthy_replicas: int
    success_threshold: int
    failure_threshold: int


@dataclass(frozen=True)
class InstanceConfig:
    service_id: str
    instance_id: str
    role: str
    version: int


@dataclass(frozen=True)
class Observation:
    service_id: str
    instance_id: str
    round_index: int
    observed_at_ms: int
    result: str


@dataclass
class Scene:
    scene_id: str
    scene_version: int
    services: dict[str, ServiceConfig]
    instances: dict[str, InstanceConfig]
    observations: list[Observation]
    by_instance_round: dict[str, dict[int, Observation]] = field(default_factory=dict)
    rounds: tuple[int, ...] = ()
    service_members: dict[str, list[str]] = field(default_factory=dict)

    def scale(self) -> dict[str, int]:
        return {
            "service_count": len(self.services),
            "instance_count": len(self.instances),
            "observation_rows": len(self.observations),
            "round_span": (self.rounds[-1] - self.rounds[0] + 1) if self.rounds else 0,
            "max_round": self.rounds[-1] if self.rounds else 0,
        }

    def instance_service(self, instance_id: str) -> str:
        return self.instances[instance_id].service_id


@dataclass
class InstanceState:
    """Mutable health bookkeeping for one instance."""

    instance_id: str
    service_id: str
    state: str = STATE_UNKNOWN
    success_streak: int = 0
    failure_streak: int = 0
    last_observed_at_ms: int | None = None
    transitions: list[dict[str, Any]] = field(default_factory=list)
    last_observation: Observation | None = None

    def snapshot(self, round_index: int, observed: bool) -> dict[str, Any]:
        """Immutable per-round record embedded in conclusions."""

        return {
            "instance_id": self.instance_id,
            "service_id": self.service_id,
            "state": self.state,
            "success_streak": self.success_streak,
            "failure_streak": self.failure_streak,
            "last_observed_at_ms": self.last_observed_at_ms,
            "observed_in_round": bool(observed),
            "no_observation": not observed,
            "transitions": [dict(item) for item in self.transitions],
        }


@dataclass
class RuntimeInstance:
    """Runtime role/version bookkeeping that switch transactions mutate."""

    instance_id: str
    service_id: str
    role: str
    version: int

    def clone(self) -> "RuntimeInstance":
        return RuntimeInstance(self.instance_id, self.service_id, self.role, self.version)


def clone_runtime(runtime: dict[str, RuntimeInstance]) -> dict[str, RuntimeInstance]:
    return {key: value.clone() for key, value in runtime.items()}


def primary_of(runtime: dict[str, RuntimeInstance], service_id: str) -> str:
    for instance_id in sorted(
        item.instance_id for item in runtime.values() if item.service_id == service_id
    ):
        if runtime[instance_id].role == ROLE_PRIMARY:
            return instance_id
    raise KeyError(f"no primary instance for service {service_id}")


def healthy_members(
    instance_states: dict[str, InstanceState],
    runtime: dict[str, RuntimeInstance],
    service_id: str,
) -> list[str]:
    members = [
        instance_id
        for instance_id, item in runtime.items()
        if item.service_id == service_id
        and instance_states[instance_id].state == STATE_HEALTHY
    ]
    return sorted(members)


def replace_state(snapshot: dict[str, Any]) -> InstanceState:
    state = InstanceState(
        instance_id=snapshot["instance_id"],
        service_id=snapshot["service_id"],
        state=snapshot["state"],
        success_streak=snapshot["success_streak"],
        failure_streak=snapshot["failure_streak"],
        last_observed_at_ms=snapshot["last_observed_at_ms"],
        transitions=[dict(item) for item in snapshot["transitions"]],
    )
    return state
