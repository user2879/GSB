"""Independent fault-injection configuration.

Injections are isolated from the regular prefilter/fixed-point configs so a
normal run is never affected. Every fired injection carries its injection id
into the failure list and the diagnostic report.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .errors import E_CONFIG_INVALID, FailureList

VALID_KINDS = ("nonfinite_scale", "force_overflow", "truncate_pack",
               "tamper_snapshot")


@dataclass
class Injection:
    injection_id: str
    kind: str
    target: dict[str, Any] = field(default_factory=dict)

    def matches_block(self, qb: int, kb: int) -> bool:
        return self.target.get("qb") == qb and self.target.get("kb") == kb


@dataclass
class InjectionPlan:
    injections: list[Injection] = field(default_factory=list)

    def block_injections(self, qb: int, kb: int) -> list[Injection]:
        return [item for item in self.injections if item.matches_block(qb, kb)]

    def has(self, kind: str) -> bool:
        return any(item.kind == kind for item in self.injections)

    def of_kind(self, kind: str) -> list[Injection]:
        return [item for item in self.injections if item.kind == kind]


def load_injections(path: str | None, failures: FailureList) -> InjectionPlan:
    if not path:
        return InjectionPlan()
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        failures.add(E_CONFIG_INVALID, "injection_load", path, exc, detail=str(exc))
        return InjectionPlan()
    items = raw.get("injections", []) if isinstance(raw, dict) else []
    plan = InjectionPlan()
    for pos, item in enumerate(items):
        target = f"{path}#injection[{pos}]"
        if not isinstance(item, dict) or item.get("kind") not in VALID_KINDS:
            failures.add(E_CONFIG_INVALID, "injection_validate", target,
                         ValueError(f"kind must be one of {VALID_KINDS}"))
            continue
        plan.injections.append(
            Injection(
                injection_id=str(item.get("id", f"inj-{pos}")),
                kind=item["kind"],
                target=item.get("target", {}) if isinstance(item.get("target", {}), dict)
                else {},
            )
        )
    return plan
