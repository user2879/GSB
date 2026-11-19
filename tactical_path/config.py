"""YAML configuration loading with strict, reproducible defaults."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .errors import ErrorCode, TacticalError
from .models import Budgets, MobilityRule, TierSpec


@dataclass(frozen=True)
class AppConfig:
    seed: int
    mobility: dict[str, MobilityRule]
    tiers: dict[str, TierSpec]
    budgets: dict[str, Budgets]
    heuristic: str
    cost_tolerance: float
    sequence_tolerance: int
    cjp_min_cells: int
    estimator: dict[str, float]
    tight_total_memory_mib: float

    @property
    def tier_order(self) -> tuple[str, ...]:
        return tuple(self.tiers)


def _number(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TacticalError(
            ErrorCode.BUDGET_DENIED,
            f"{label} must be numeric",
            context={"value": repr(value)},
        ) from exc
    return result


def load_config(path: str | Path) -> AppConfig:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except OSError as exc:
        raise TacticalError(
            ErrorCode.SNAPSHOT_VERSION,
            "unable to read configuration",
            context={"path": str(path)},
        ) from exc
    except yaml.YAMLError as exc:
        raise TacticalError(
            ErrorCode.SNAPSHOT_VERSION,
            "invalid YAML configuration",
            context={"path": str(path)},
        ) from exc

    if not isinstance(raw, dict):
        raise TacticalError(
            ErrorCode.MAP_SHAPE, "configuration root must be a mapping"
        )

    mobility: dict[str, MobilityRule] = {}
    for name, item in sorted((raw.get("mobility") or {}).items()):
        mobility[name] = MobilityRule(
            name=str(name),
            cost_multiplier=int(item["cost_multiplier"]),
            max_terrain=int(item["max_terrain"]),
        )
    if not mobility:
        raise TacticalError(ErrorCode.UNIT_INVALID, "no movement classes configured")

    tiers: dict[str, TierSpec] = {}
    for name, item in (raw.get("tiers") or {}).items():
        tiers[str(name)] = TierSpec(
            name=str(name),
            size=int(item["size"]),
            region_size=int(item["region_size"]),
            unit_count=int(item["unit_count"]),
            query_count=int(item["query_count"]),
            query_mode=str(item.get("query_mode", "cross_region")),
            target_groups=int(item.get("target_groups", item["query_count"])),
            interior_walls=bool(item.get("interior_walls", True)),
            mobility_classes=tuple(str(value) for value in item.get("mobility_classes", [])),
        )
    if not tiers:
        raise TacticalError(ErrorCode.MAP_SHAPE, "no tiers configured")

    budgets: dict[str, Budgets] = {}
    for name, item in (raw.get("budgets") or {}).items():
        budget = Budgets(
            tier=str(name),
            query_time_ms=_number(item["query_time_ms"], f"{name}.query_time_ms"),
            query_memory_mib=_number(
                item["query_memory_mib"], f"{name}.query_memory_mib"
            ),
            total_memory_mib=_number(
                item["total_memory_mib"], f"{name}.total_memory_mib"
            ),
            max_threads=int(item["max_threads"]),
            precompute_time_ms=_number(
                item["precompute_time_ms"], f"{name}.precompute_time_ms"
            ),
        )
        budget.validate()
        budgets[str(name)] = budget
    if set(tiers) != set(budgets):
        raise TacticalError(
            ErrorCode.BUDGET_DENIED,
            "each tier needs exactly one budget block",
            context={"tiers": sorted(tiers), "budgets": sorted(budgets)},
        )

    solver = raw.get("solver") or {}
    estimator = {
        "ns_per_expansion": _number(
            solver.get("ns_per_expansion", 4000), "solver.ns_per_expansion"
        ),
        "flow_ns_per_cell": _number(
            solver.get("flow_ns_per_cell", 800), "solver.flow_ns_per_cell"
        ),
        "jump_ns_per_cell": _number(
            solver.get("jump_ns_per_cell", 500), "solver.jump_ns_per_cell"
        ),
    }

    return AppConfig(
        seed=int(raw.get("seed", 20261119)),
        mobility=mobility,
        tiers=tiers,
        budgets=budgets,
        heuristic=str(solver.get("heuristic", "manhattan")),
        cost_tolerance=_number(
            solver.get("cost_tolerance", 0.0), "solver.cost_tolerance"
        ),
        sequence_tolerance=int(solver.get("sequence_tolerance", 0)),
        cjp_min_cells=int(solver.get("cjp_min_cells", 100000)),
        estimator=estimator,
        tight_total_memory_mib=_number(
            raw.get("demo", {}).get("tight_total_memory_mib", 0.000001),
            "demo.tight_total_memory_mib",
        ),
    )
