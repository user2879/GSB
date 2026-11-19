"""Faction rule configuration and range validation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .errors import ErrorCode, RtsError


@dataclass(frozen=True)
class FactionRules:
    faction: str
    resource_cap: int
    collect_rate: int
    unit_cost: int
    max_concurrent_commands: int

    def validate(self) -> None:
        checks = {
            "resource_cap": self.resource_cap,
            "collect_rate": self.collect_rate,
            "unit_cost": self.unit_cost,
            "max_concurrent_commands": self.max_concurrent_commands,
        }
        for name, value in checks.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise RtsError(
                    ErrorCode.E_RULE_RANGE,
                    f"faction '{self.faction}' rule '{name}' must be a positive integer",
                    actor=self.faction,
                    context={"field": name, "value": value},
                )
        if self.unit_cost > self.resource_cap:
            raise RtsError(
                ErrorCode.E_RULE_RANGE,
                f"faction '{self.faction}' unit_cost exceeds resource_cap",
                actor=self.faction,
                context={
                    "unit_cost": self.unit_cost,
                    "resource_cap": self.resource_cap,
                },
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "faction": self.faction,
            "resource_cap": self.resource_cap,
            "collect_rate": self.collect_rate,
            "unit_cost": self.unit_cost,
            "max_concurrent_commands": self.max_concurrent_commands,
        }


@dataclass(frozen=True)
class RuleConfig:
    match_seed: int
    rng_chain_version: int
    queue_ticks: int
    factions: tuple[FactionRules, ...]

    def validate(self) -> None:
        if isinstance(self.match_seed, bool) or not isinstance(self.match_seed, int):
            raise RtsError(
                ErrorCode.E_RULE_RANGE, "match_seed must be an integer"
            )
        if self.rng_chain_version < 1:
            raise RtsError(
                ErrorCode.E_RULE_RANGE,
                "rng_chain_version must be >= 1",
                context={"value": self.rng_chain_version},
            )
        if self.queue_ticks < 1:
            raise RtsError(
                ErrorCode.E_RULE_RANGE,
                "queue_ticks must be >= 1",
                context={"value": self.queue_ticks},
            )
        if not self.factions:
            raise RtsError(ErrorCode.E_RULE_RANGE, "at least one faction is required")
        names = [faction.faction for faction in self.factions]
        if len(set(names)) != len(names):
            raise RtsError(
                ErrorCode.E_RULE_RANGE, "duplicate faction names", context={"names": names}
            )
        for faction in self.factions:
            faction.validate()

    def faction_names(self) -> tuple[str, ...]:
        return tuple(faction.faction for faction in self.factions)

    def get(self, faction: str) -> FactionRules:
        for item in self.factions:
            if item.faction == faction:
                return item
        raise RtsError(
            ErrorCode.E_RULE_RANGE,
            f"no rules configured for actor faction '{faction}'",
            actor=faction,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "match_seed": self.match_seed,
            "rng_chain_version": self.rng_chain_version,
            "queue_ticks": self.queue_ticks,
            "factions": [faction.to_dict() for faction in self.factions],
        }


def rules_from_dict(raw: Any) -> RuleConfig:
    if not isinstance(raw, dict):
        raise RtsError(ErrorCode.E_INPUT_MALFORMED, "rules must be a JSON object")
    try:
        faction_items = tuple(
            FactionRules(
                faction=item["faction"],
                resource_cap=item["resource_cap"],
                collect_rate=item["collect_rate"],
                unit_cost=item["unit_cost"],
                max_concurrent_commands=item["max_concurrent_commands"],
            )
            for item in raw["factions"]
        )
        config = RuleConfig(
            match_seed=raw["match_seed"],
            rng_chain_version=raw["rng_chain_version"],
            queue_ticks=raw["queue_ticks"],
            factions=faction_items,
        )
    except KeyError as exc:
        raise RtsError(
            ErrorCode.E_INPUT_MALFORMED,
            f"rules missing field: {exc.args[0]}",
            context={"field": exc.args[0]},
        ) from exc
    config.validate()
    return config
