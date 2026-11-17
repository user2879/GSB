from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any, Iterable

from .errors import DiagError, E_VIEW
from .latency import aggregate_latency_events
from .ledger import LatencyEvent, LedgerNode, TreeSnapshot


@dataclass(frozen=True)
class ViewItem:
    root_revision: int | None
    cluster_id: str
    node_revision: int
    task_id: str
    parent_revision: int | None
    aggregate_summary_canonical: str
    content_digest: str
    completed_at_ns: int

    @property
    def aggregate_summary(self) -> dict[str, Any]:
        return json.loads(self.aggregate_summary_canonical)

    def as_dict(self) -> dict[str, Any]:
        return {
            "root_revision": self.root_revision,
            "cluster_id": self.cluster_id,
            "node_revision": self.node_revision,
            "task_id": self.task_id,
            "parent_revision": self.parent_revision,
            "aggregate_summary": self.aggregate_summary,
            "content_digest": self.content_digest,
            "completed_at_ns": self.completed_at_ns,
        }


@dataclass(frozen=True)
class RootSwitchReport:
    old_root_revision: int | None
    new_root_revision: int
    affected_clusters: tuple[str, ...]
    changed_clusters: tuple[dict[str, Any], ...]
    stale_nodes: tuple[dict[str, Any], ...]
    reused_unchanged_node_clusters: tuple[str, ...]
    incremental_rows_touched: tuple[str, ...]

    @property
    def changed_cluster_ids(self) -> tuple[str, ...]:
        return tuple(item["cluster_id"] for item in self.changed_clusters)

    @property
    def stale_node_keys(self) -> tuple[tuple[str, int], ...]:
        return tuple((item["cluster_id"], item["revision"]) for item in self.stale_nodes)

    def as_dict(self) -> dict[str, Any]:
        return {
            "old_root_revision": self.old_root_revision,
            "new_root_revision": self.new_root_revision,
            "affected_clusters": list(self.affected_clusters),
            "changed_clusters": list(self.changed_clusters),
            "stale_nodes": list(self.stale_nodes),
            "reused_unchanged_node_clusters": list(self.reused_unchanged_node_clusters),
            "incremental_rows_touched": list(self.incremental_rows_touched),
            "policy": "whole-root-replacement; no old/new node merge",
        }


def _view_item(node: LedgerNode, root_revision: int) -> ViewItem:
    return ViewItem(
        root_revision=root_revision,
        cluster_id=node.cluster_id,
        node_revision=node.revision,
        task_id=node.task_id,
        parent_revision=node.parent_revision,
        aggregate_summary_canonical=node.aggregate_summary_canonical,
        content_digest=node.content_digest,
        completed_at_ns=node.completed_at_ns,
    )


class IncrementalViewModel:
    """UI-side model with incremental rows and immutable whole-root snapshots.

    Node payloads are copied only when a cluster's effective node changes. Root
    bindings for all rows are replaced as one operation. Old root snapshots are
    retained read-only; all old rows immediately become stale and are excluded
    from statistics bound to the new root.
    """

    def __init__(self) -> None:
        self._active_snapshot = TreeSnapshot(None, {}, {}, None, None)
        self._items: dict[str, ViewItem] = {}
        self._root_history: dict[int, TreeSnapshot] = {}
        self._stale_nodes: tuple[dict[str, Any], ...] = ()
        self.switch_generation = 0
        self.last_switch: RootSwitchReport | None = None

    @property
    def root_revision(self) -> int | None:
        return self._active_snapshot.root_revision

    @property
    def active_snapshot(self) -> TreeSnapshot:
        return self._active_snapshot

    @property
    def items(self) -> tuple[ViewItem, ...]:
        return tuple(self._items[cluster_id] for cluster_id in sorted(self._items))

    @property
    def stale_nodes(self) -> tuple[dict[str, Any], ...]:
        return self._stale_nodes

    @property
    def root_history(self) -> dict[int, TreeSnapshot]:
        return dict(self._root_history)

    def initialize(self, snapshot: TreeSnapshot) -> None:
        self._active_snapshot = snapshot
        if snapshot.root_revision is not None:
            self._root_history[snapshot.root_revision] = snapshot
        self._items = {
            cluster_id: _view_item(node, snapshot.root_revision)
            for cluster_id, node in sorted(snapshot.effective_nodes.items())
        }

    def _stale_record(self, item: ViewItem) -> dict[str, Any]:
        return {
            "cluster_id": item.cluster_id,
            "revision": item.node_revision,
            "bound_root_revision": item.root_revision,
            "content_digest": item.content_digest,
        }

    def apply_whole_root_replacement(
        self,
        snapshot: TreeSnapshot,
        *,
        affected_clusters: Iterable[str],
    ) -> RootSwitchReport:
        if snapshot.root_revision is None:
            raise DiagError(E_VIEW, "cannot switch to an empty root", field_path="root_revision")
        old_root = self._active_snapshot.root_revision
        if old_root is not None and snapshot.root_revision <= old_root:
            raise DiagError(
                E_VIEW,
                "new version-tree root must be newer than the current root",
                field_path="root_revision",
                details={"old_root": old_root, "new_root": snapshot.root_revision},
            )
        affected = tuple(sorted(set(affected_clusters)))
        old_items = dict(self._items)
        changed: list[dict[str, Any]] = []
        reused: list[str] = []
        touched: set[str] = set(affected)

        all_clusters = sorted(set(old_items) | set(snapshot.effective_nodes))
        for cluster_id in all_clusters:
            old_item = old_items.get(cluster_id)
            new_node = snapshot.effective_nodes.get(cluster_id)
            old_node_revision = None if old_item is None else old_item.node_revision
            old_digest = None if old_item is None else old_item.content_digest
            old_summary = None if old_item is None else old_item.aggregate_summary
            new_node_revision = None if new_node is None else new_node.revision
            new_digest = None if new_node is None else new_node.content_digest
            new_summary = None if new_node is None else new_node.aggregate_summary
            if (
                old_item is None
                or new_node is None
                or old_node_revision != new_node_revision
                or old_digest != new_digest
            ):
                changed.append(
                    {
                        "cluster_id": cluster_id,
                        "old_revision": old_node_revision,
                        "old_summary": old_summary,
                        "old_summary_digest": old_digest,
                        "new_revision": new_node_revision,
                        "new_summary": new_summary,
                        "new_summary_digest": new_digest,
                    }
                )
                touched.add(cluster_id)
            else:
                reused.append(cluster_id)

        stale = tuple(self._stale_record(item) for item in sorted(old_items.values(), key=lambda item: item.cluster_id))
        if old_root is not None:
            self._root_history[old_root] = self._active_snapshot

        new_items: dict[str, ViewItem] = {}
        for cluster_id, node in sorted(snapshot.effective_nodes.items()):
            new_items[cluster_id] = _view_item(node, snapshot.root_revision)

        # Root binding changes atomically; unchanged node payloads are reused by identity
        # rather than re-merged from the old tree.
        self._active_snapshot = snapshot
        self._items = new_items
        self._root_history[snapshot.root_revision] = snapshot
        self._stale_nodes = stale
        self.switch_generation += 1
        report = RootSwitchReport(
            old_root_revision=old_root,
            new_root_revision=snapshot.root_revision,
            affected_clusters=affected,
            changed_clusters=tuple(changed),
            stale_nodes=stale,
            reused_unchanged_node_clusters=tuple(sorted(reused)),
            incremental_rows_touched=tuple(sorted(touched)),
        )
        self.last_switch = report
        return report

    def item(self, cluster_id: str) -> ViewItem | None:
        return self._items.get(cluster_id)

    def is_stale(self, item: ViewItem) -> bool:
        current = self._items.get(item.cluster_id)
        return (
            item.root_revision != self.root_revision
            or current is None
            or current.node_revision != item.node_revision
            or current.content_digest != item.content_digest
        )

    def current_latency_stats(self, events: Iterable[LatencyEvent]) -> dict[str, Any]:
        current_events: list[LatencyEvent] = []
        for event in events:
            if event.bound_root_revision != self.root_revision:
                continue
            if event.details.get("stale") is True:
                continue
            if event.kind == "ui_query":
                item = self._items.get(event.cluster_id or "")
                if item is None or event.revision != item.node_revision:
                    continue
            elif event.cluster_id is not None and not event.cluster_id.startswith("_") and event.cluster_id not in self._items:
                continue
            current_events.append(event)
        return aggregate_latency_events(current_events)

    def rebind_item_for_current_root(self, item: ViewItem) -> ViewItem:
        if not self.is_stale(item):
            return item
        current = self._items.get(item.cluster_id)
        if current is None:
            raise DiagError(
                E_VIEW,
                "cluster is not present in the active root",
                details={"cluster_id": item.cluster_id, "root_revision": self.root_revision},
            )
        return replace(
            current,
            root_revision=self.root_revision,
        )
