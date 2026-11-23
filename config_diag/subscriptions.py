from __future__ import annotations

import copy
from typing import Any

import numpy as np

from .engine import ConfigEngine
from .errors import DiagnosticError
from .model import SubscriptionSample


class SubscriptionSimulator:
    """Deterministic polling model with whole-view replacement semantics."""

    def __init__(self, engine: ConfigEngine, sample: SubscriptionSample) -> None:
        self.engine = engine
        self.sample = sample
        self.cache: dict[str, dict[str, Any] | None] = {
            client.client_id: None for client in sample.clients
        }
        self.stale_events: list[dict[str, Any]] = []

    def poll(self, rounds: int = 1) -> dict[str, Any]:
        if rounds <= 0:
            raise DiagnosticError("E_STATE", "poll rounds 必须为正整数", field="rounds")
        current = self.engine.current_view()
        active = {item["key"]: item for item in current["active"]}
        rounds_payload: list[dict[str, Any]] = []
        for round_index in range(rounds):
            client_payload: list[dict[str, Any]] = []
            for client in self.sample.clients:
                previous = self.cache[client.client_id]
                replaced = False
                stale = False
                if previous is None:
                    replaced = True
                elif (
                    previous["config_set_version"] != current["config_set_version"]
                    or previous["derivation_version"] != current["derivation_version"]
                    or previous["view_digest"] != current["view_digest"]
                ):
                    stale = True
                    replaced = True
                    event = {
                        "client_id": client.client_id,
                        "old_view": previous["view_version"],
                        "old_set_version": previous["config_set_version"],
                        "old_derivation_version": previous["derivation_version"],
                        "new_view": current["view_version"],
                        "new_set_version": current["config_set_version"],
                        "new_derivation_version": current["derivation_version"],
                        "action": "whole_view_replace",
                    }
                    self.stale_events.append(event)
                # Replacement, when needed, copies the complete current view and
                # never patches the previous cache key by key.
                if replaced:
                    self.cache[client.client_id] = copy.deepcopy(current)

                seed = (client.seed + round_index * 2_654_435_761) & 0xFFFFFFFF
                rng = np.random.RandomState(seed)
                size = min(client.sample_size, len(client.keys))
                indices = sorted(int(value) for value in rng.choice(len(client.keys), size=size, replace=False))
                observations = []
                unavailable = []
                visible_values = []
                for index in indices:
                    key = client.keys[index]
                    record = active.get(key)
                    if record is None:
                        unavailable.append(key)
                        continue
                    observations.append(
                        {
                            "key": key,
                            "value": record["value"],
                            "version": record["version"],
                            "content_digest": record["content_digest"],
                            "view_version": current["view_version"],
                        }
                    )
                    visible_values.append(record["value"])
                client_payload.append(
                    {
                        "client_id": client.client_id,
                        "seed": client.seed,
                        "poll_interval_ms": client.poll_interval_ms,
                        "view_version": current["view_version"],
                        "set_version": current["config_set_version"],
                        "derivation_version": current["derivation_version"],
                        "view_digest": current["view_digest"],
                        "replaced_whole_view": replaced,
                        "stale_cache_replaced": stale,
                        "observations": observations,
                        "unavailable": unavailable,
                        "visible_count": len(observations),
                        "visible_sum": sum(visible_values),
                    }
                )
            rounds_payload.append({"round": round_index + 1, "clients": client_payload})
        return {
            "ok": True,
            "rounds": rounds_payload,
            "stale_events": self.stale_events,
            "rng": "numpy.random.RandomState(seed + round*2654435761)",
            "statistics_source": "current-view-only",
            "current_view_version": current["view_version"],
            "current_set_version": current["config_set_version"],
            "current_derivation_version": current["derivation_version"],
        }
