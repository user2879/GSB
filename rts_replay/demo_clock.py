"""Fixed deterministic per-tick simulated workload costs for the demo."""
from __future__ import annotations


def build_scripted_costs(
    last_tick: int, budget_ms: float, spike_factor: float
) -> dict[int, float]:
    """Return per-tick costs in milliseconds.

    Regular ticks cost half the budget so frames normally fit; every seventh
    tick simulates a latency spike several times the budget, forcing the player
    to halve its per-frame allowance and skip a refresh. Tick 0 carries the
    default cost used when a tick is absent from the map.
    """
    regular_ms = budget_ms * 0.5
    spike_ms = budget_ms * spike_factor
    costs = {-1: regular_ms}
    for tick in range(1, last_tick + 1):
        costs[tick] = spike_ms if tick % 7 == 0 else regular_ms
    return costs
