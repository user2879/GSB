"""View version registry and whole-view replacement semantics.

A view version increases monotonically whenever its structural fingerprint
changes (participating environments or per-key policy) or when a view is added.
Deleted views are removed from the active registry. Conclusions produced under
an older fingerprint are marked stale and excluded from the current display;
re-arbitration always replaces a view conclusion wholesale and reports the
stable added / removed / changed key sets.
"""
from __future__ import annotations

from dataclasses import dataclass

from .canonical import fingerprint_view


@dataclass
class ViewVersion:
    view_id: str
    version: int
    fingerprint: str


@dataclass
class ViewDiff:
    added_keys: list[str]
    removed_keys: list[str]
    changed_keys: list[str]
    version_bumped: bool


class VersionRegistry:
    def __init__(self):
        # view_id -> ViewVersion
        self._versions: dict[str, ViewVersion] = {}

    def reconcile(self, views: dict) -> dict[str, ViewVersion]:
        """Reconcile registry against freshly loaded views.

        Deleted views are dropped; new views start at version 1; edited views
        get version + 1; unchanged views keep their version.
        """
        next_registry: dict[str, ViewVersion] = {}
        for view_id in sorted(views):
            view = views[view_id]
            fingerprint = fingerprint_view(view)
            previous = self._versions.get(view_id)
            if previous is None:
                next_registry[view_id] = ViewVersion(view_id, 1, fingerprint)
            elif previous.fingerprint != fingerprint:
                next_registry[view_id] = ViewVersion(
                    view_id, previous.version + 1, fingerprint
                )
            else:
                next_registry[view_id] = previous
        self._versions = next_registry
        return dict(next_registry)

    def version_of(self, view_id: str) -> int:
        return self._versions[view_id].version

    def active_view_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._versions))

    def snapshot(self) -> dict:
        return {
            view_id: {"version": vv.version, "fingerprint": vv.fingerprint}
            for view_id, vv in sorted(self._versions.items())
        }

    @classmethod
    def from_snapshot(cls, payload: dict) -> "VersionRegistry":
        registry = cls()
        for view_id, row in sorted(payload.items()):
            registry._versions[view_id] = ViewVersion(
                view_id=view_id, version=row["version"], fingerprint=row["fingerprint"]
            )
        return registry


def diff_keys(old_policies: dict[str, str], new_policies: dict[str, str]) -> ViewDiff:
    old_set, new_set = set(old_policies), set(new_policies)
    added = sorted(new_set - old_set)
    removed = sorted(old_set - new_set)
    changed = sorted(
        key for key in old_set & new_set if old_policies[key] != new_policies[key]
    )
    return ViewDiff(
        added_keys=added,
        removed_keys=removed,
        changed_keys=changed,
        version_bumped=bool(added or removed or changed),
    )


def diff_conclusions(old_digest_by_key: dict[str, str],
                     new_digest_by_key: dict[str, str]) -> ViewDiff:
    """Compare per-key conclusion digests after whole-view replacement."""
    old_set, new_set = set(old_digest_by_key), set(new_digest_by_key)
    added = sorted(new_set - old_set)
    removed = sorted(old_set - new_set)
    changed = sorted(
        key for key in old_set & new_set
        if old_digest_by_key[key] != new_digest_by_key[key]
    )
    return ViewDiff(added, removed, changed, bool(added or removed or changed))
