"""Staleness guard and whole-view replacement on spec evolution."""
from __future__ import annotations

from dataclasses import dataclass

from .errors import SpecError
from .util import canonical_digest


@dataclass(frozen=True)
class ViewDiff:
    old_version: int
    new_version: int
    added: list[str]
    removed: list[str]
    changed: list[str]

    def to_record(self) -> dict:
        return {
            "old_version": self.old_version,
            "new_version": self.new_version,
            "added": self.added,
            "removed": self.removed,
            "changed": self.changed,
        }


def assert_version_monotonic(old_version: int, new_version: int) -> None:
    if new_version < old_version:
        raise SpecError(
            "declared_version must be monotonically non-decreasing", 0,
            "declared_version", f"new={new_version} < old={old_version}")


def _index(records: list[dict]) -> dict[str, dict]:
    return {r["aggregate_id"]: r for r in records}


def diff_views(old_aggregates: list[dict], old_version: int,
               new_aggregates: list[dict], new_version: int) -> ViewDiff:
    """Whole-view replacement diff (no incremental merge)."""
    old = _index(old_aggregates)
    new = _index(new_aggregates)
    added = sorted(set(new) - set(old))
    removed = sorted(set(old) - set(new))
    changed = sorted(
        aid for aid in set(old) & set(new)
        if canonical_digest(old[aid]) != canonical_digest(new[aid]))
    return ViewDiff(old_version=old_version, new_version=new_version,
                    added=added, removed=removed, changed=changed)
