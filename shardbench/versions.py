"""Monotonic version registry for rules, hash ring, dep graph and budgets."""

from __future__ import annotations

from dataclasses import dataclass, asdict

from .errors import BizError, E_STATE


@dataclass(frozen=True)
class VersionSet:
    rule_version: int
    ring_version: int
    dep_graph_version: int
    budget_version: int

    def as_dict(self):
        return asdict(self)


class VersionRegistry:
    def __init__(self, initial: VersionSet):
        self._v = initial

    def current(self) -> VersionSet:
        return self._v

    def _bump(self, field, new_value) -> VersionSet:
        old = getattr(self._v, field)
        if not isinstance(new_value, int) or new_value <= old:
            raise BizError(
                E_STATE,
                f"{field} must increase monotonically ({old} -> {new_value})",
                field=field,
            )
        self._v = VersionSet(**{**self._v.as_dict(), field: new_value})
        return self._v

    def bump_rule(self, v):
        return self._bump("rule_version", v)

    def bump_ring(self, v):
        return self._bump("ring_version", v)

    def bump_dep_graph(self, v):
        return self._bump("dep_graph_version", v)

    def bump_budget(self, v):
        return self._bump("budget_version", v)


def diff_units(old_config, new_config):
    """Stable (added, removed, upgraded) routing-unit lists between configs."""
    old_ids = set(old_config.units) if old_config else set()
    new_ids = set(new_config.units)
    added = sorted(new_ids - old_ids)
    removed = sorted(old_ids - new_ids)
    upgraded = sorted(
        u
        for u in (old_ids & new_ids)
        if old_config.units[u].rule_view() != new_config.units[u].rule_view()
    )
    return {"added": added, "removed": removed, "upgraded": upgraded}
