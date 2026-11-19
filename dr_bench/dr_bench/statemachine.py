"""Per-instance health state machine driven by observation rows."""

from __future__ import annotations

from .models import (
    OBS_SUCCESS,
    STATE_HEALTHY,
    STATE_UNHEALTHY,
    STATE_UNKNOWN,
    InstanceState,
    Observation,
    ServiceConfig,
)


def effective_threshold(value: int) -> int:
    """A zero threshold behaves like a single required observation."""

    return max(1, value)


def apply_observation(
    state: InstanceState,
    config: ServiceConfig,
    observation: Observation,
) -> dict | None:
    """Advance one instance with a fresh observation.

    Returns the transition record when the health state changed, otherwise
    ``None``. The transition is appended to the instance history as well.
    """

    previous_state = state.state
    if observation.result == OBS_SUCCESS:
        state.success_streak += 1
        state.failure_streak = 0
        if state.success_streak >= effective_threshold(config.success_threshold):
            state.state = STATE_HEALTHY
    else:
        state.failure_streak += 1
        state.success_streak = 0
        if state.failure_streak >= effective_threshold(config.failure_threshold):
            state.state = STATE_UNHEALTHY
    state.last_observed_at_ms = observation.observed_at_ms
    state.last_observation = observation
    if state.state != previous_state:
        transition = {
            "instance_id": state.instance_id,
            "from_state": previous_state,
            "to_state": state.state,
            "round_index": observation.round_index,
            "observed_at_ms": observation.observed_at_ms,
            "reason": f"threshold_met:{observation.result}",
        }
        state.transitions.append(transition)
        return transition
    return None


def initial_states(scene) -> dict[str, InstanceState]:
    return {
        instance_id: InstanceState(
            instance_id=instance_id,
            service_id=instance.service_id,
            state=STATE_UNKNOWN,
        )
        for instance_id, instance in sorted(scene.instances.items())
    }
