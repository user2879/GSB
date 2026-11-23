"""Core diagnostics engine: incremental dispatch, invalidation, rebuild,
rollback, crash recovery and deterministic replay."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field

from . import aggregate as agg
from .errors import DiagError, canonical_json, raise_ref, raise_snapshot
from .ledger import Ledger
from .model import Revision, Spec, Task


@dataclass(frozen=True)
class NodeKey:
    task_id: str
    cluster_id: str
    version: int

    def as_tuple(self):
        return (self.version, self.task_id, self.cluster_id)

    def as_dict(self):
        return {"task": self.task_id, "cluster": self.cluster_id,
                "revision": self.version}


@dataclass
class InvalidationReport:
    changed_revisions: list[int] = field(default_factory=list)
    seeds: list[dict] = field(default_factory=list)
    steps: list[dict] = field(default_factory=list)
    invalid: list[NodeKey] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "changed_revisions": self.changed_revisions,
            "seeds": self.seeds,
            "propagation": self.steps,
            "invalid_set": [key.as_dict() for key in self.invalid],
        }


class Engine:
    def __init__(self, spec: Spec, ledger: Ledger, orchestrator=None):
        self.spec = spec
        self.ledger = ledger
        self.orchestrator = orchestrator

    # ------------------------------------------------------------------
    # Revision graph helpers
    # ------------------------------------------------------------------
    def forward_edges(self) -> dict[int, list[dict]]:
        """Parent and derivation edges, both pointing at later revisions."""
        edges: dict[int, list[dict]] = {r.version: [] for r in self.spec.revisions}
        for rev in self.spec.revisions:
            if rev.parent is not None:
                edges.setdefault(rev.parent, []).append(
                    {"to": rev.version, "edge": "parent",
                     "from": rev.parent})
            for derived in rev.derives:
                edges.setdefault(derived, []).append(
                    {"to": rev.version, "edge": "derives", "from": derived})
        return edges

    def chain_path(self, version: int) -> list[Revision]:
        return [self.spec.revision(v) for v in self.spec.ancestors_including(version)]

    def nodes_for_version(self, version: int) -> list[NodeKey]:
        """Stable (task, cluster) node ordering for one revision."""
        keys: list[NodeKey] = []
        for task in self.spec.tasks:
            for cluster_id in task.targets:
                keys.append(NodeKey(task.task_id, cluster_id, version))
        return keys

    def root_chain_nodes(self, root_version: int) -> list[NodeKey]:
        nodes: list[NodeKey] = []
        for version in self.spec.ancestors_including(root_version):
            nodes.extend(self.nodes_for_version(version))
        return nodes

    def _task_of(self, task_id: str) -> Task:
        return self.spec.task_map[task_id]

    # ------------------------------------------------------------------
    # Requirement 5: invalidation propagation
    # ------------------------------------------------------------------
    def diff_revisions(self, previous: list[dict] | None,
                       current: list[dict]) -> dict[str, list[int]]:
        """Classify revisions into value/strategy/derives change buckets."""
        buckets = {"values": [], "strategy": [], "derives": []}
        if not previous:
            return buckets
        old = {item["version"]: item for item in previous}
        for item in current:
            before = old.get(item["version"])
            if before is None:
                buckets["values"].append(item["version"])
                buckets["strategy"].append(item["version"])
                buckets["derives"].append(item["version"])
                continue
            if before.get("values") != item.get("values"):
                buckets["values"].append(item["version"])
            if before.get("strategy") != item.get("strategy"):
                buckets["strategy"].append(item["version"])
            if before.get("derives") != item.get("derives"):
                buckets["derives"].append(item["version"])
        for key in buckets:
            buckets[key].sort()
        return buckets

    def analyze_invalidation(self, changed: dict[str, list[int]]) -> InvalidationReport:
        report = InvalidationReport(
            changed_revisions=sorted({v for vals in changed.values() for v in vals}))
        edges = self.forward_edges()
        invalid: dict[NodeKey, str] = {}
        queue: deque[tuple[int, str, NodeKey | None, str | None]] = deque()

        def seed_nodes(version: int, reason: str, basis: str):
            for key in self.nodes_for_version(version):
                if key not in invalid:
                    invalid[key] = reason
                    report.seeds.append({"node": key.as_dict(), "reason": reason,
                                         "basis": basis})
                    queue.append((version, reason, key, None))

        for version in changed["values"]:
            rev = self.spec.revision(version)
            touched = sorted(rev.values.keys())
            seed_nodes(version, "config_values_changed",
                       f"revision {version} changed config keys {touched}")
        for version in changed["strategy"]:
            seed_nodes(version, "strategy_changed",
                       f"revision {version} changed dispatch strategy to "
                       f"{self.spec.revision(version).strategy}")
        for version in changed["derives"]:
            seed_nodes(version, "derives_changed",
                       f"revision {version} changed derived references "
                       f"{list(self.spec.revision(version).derives)}")

        while queue:
            src_version, reason, src_key, _ = queue.popleft()
            for edge in edges.get(src_version, []):
                dst_version = edge["to"]
                candidates = self.nodes_for_version(dst_version)
                if edge["edge"] == "derives":
                    propagated = candidates
                else:
                    propagated = [
                        key for key in candidates
                        if self._node_depends_on(key, src_key, src_version)]
                for dst_key in propagated:
                    if dst_key in invalid:
                        continue
                    basis = (f"{edge['edge']} edge {src_version}->{dst_version} "
                             f"from {reason}")
                    invalid[dst_key] = f"propagated:{reason}"
                    report.steps.append({
                        "from": src_key.as_dict() if src_key else
                        {"revision": src_version},
                        "to": dst_key.as_dict(),
                        "edge": edge["edge"],
                        "reason": reason,
                        "basis": basis,
                    })
                    queue.append((dst_version, reason, dst_key, basis))
        report.invalid = sorted(invalid.keys(), key=lambda k: k.as_tuple())
        return report

    def _node_depends_on(self, dst_key: NodeKey, src_key: NodeKey | None,
                         src_version: int) -> bool:
        """Parent-edge propagation stays within the same task/cluster lane."""
        if src_key is None:
            return True
        return dst_key.task_id == src_key.task_id and \
            dst_key.cluster_id == src_key.cluster_id

    # ------------------------------------------------------------------
    # Deterministic node computation
    # ------------------------------------------------------------------
    def compute_node_locally(self, key: NodeKey) -> dict:
        task = self.spec.task_map[key.task_id]
        cluster = self.spec.cluster_map[key.cluster_id]
        path = self.chain_path(key.version)
        rev = self.spec.revision(key.version)
        values = agg.effective_values(path, task.keys)
        summary = agg.aggregate(
            cluster.seed, cluster.sample_rows, values, rev.strategy,
            task.caliber, rev.derives)
        content = agg.node_content(task.task_id, cluster.cluster_id,
                                   key.version, rev.parent, summary)
        digest = agg.content_digest(content)
        return {"task_id": task.task_id, "cluster_id": cluster.cluster_id,
                "version": key.version, "parent": rev.parent,
                "aggregate": summary, "content_digest": digest,
                "monotonic_ms": self.ledger.now_ms(),
                "status": "committed"}

    def dispatch_node(self, key: NodeKey) -> dict:
        task = self.spec.task_map[key.task_id]
        cluster = self.spec.cluster_map[key.cluster_id]
        path = self.chain_path(key.version)
        rev = self.spec.revision(key.version)
        values = agg.effective_values(path, task.keys)
        request = agg.build_request(cluster, task, key.version, values,
                                    rev.strategy, rev.derives)
        if self.orchestrator is None:
            summary = agg.summarize_request(request)
        else:
            reply = self.orchestrator.dispatch(request)
            summary = reply["aggregate"]
            expected_reply_digest = agg.content_digest({
                "cluster_id": cluster.cluster_id,
                "task_id": task.task_id,
                "version": key.version,
                "aggregate": summary})
            if reply["digest"] != expected_reply_digest:
                raise DiagError(
                    "E_REPLY",
                    f"server reply digest mismatch for cluster {key.cluster_id!r}",
                    key.as_dict())
        content = agg.node_content(task.task_id, cluster.cluster_id,
                                   key.version, rev.parent, summary)
        return {"task_id": task.task_id, "cluster_id": cluster.cluster_id,
                "version": key.version, "parent": rev.parent,
                "aggregate": summary,
                "content_digest": agg.content_digest(content),
                "monotonic_ms": self.ledger.now_ms(),
                "status": "committed"}

    def _persist(self, nodes: list[dict], batch_id: str,
                 root_version: int, replace_keys=None) -> dict:
        inserted = identical = replaced = 0
        replace_keys = replace_keys or set()
        self.ledger.begin_batch(batch_id)
        try:
            with self.ledger.transaction():
                for node in nodes:
                    self.ledger.assert_parent_committed(node)
                    key = (node["task_id"], node["cluster_id"], node["version"])
                    outcome = self.ledger.upsert_node(
                        node, allow_replace=key in replace_keys)
                    inserted += outcome == "inserted"
                    identical += outcome == "identical"
                    replaced += outcome == "replaced"
                self.ledger.set_root(root_version)
        finally:
            self.ledger.finish_batch(batch_id)
        return {"inserted": inserted, "identical": identical,
                "replaced": replaced}

    # ------------------------------------------------------------------
    # Requirement 4/5: incremental apply with invalid-only redistribution
    # ------------------------------------------------------------------
    def apply_chain(self, previous_revisions: list[dict] | None,
                    batch_id: str, progress=None) -> dict:
        current_records = [
            {"version": r.version, "values": r.values,
             "strategy": r.strategy, "derives": list(r.derives)}
            for r in self.spec.revisions]
        if previous_revisions is None:
            changed = {"values": [r["version"] for r in current_records],
                       "strategy": [r["version"] for r in current_records],
                       "derives": [r["version"] for r in current_records]}
        else:
            changed = self.diff_revisions(previous_revisions, current_records)
        report = self.analyze_invalidation(changed)
        invalid_set = set((k.task_id, k.cluster_id, k.version)
                          for k in report.invalid)
        head = self.spec.revisions[-1].version
        all_nodes = self.root_chain_nodes(head)
        recomputed_nodes: list[dict] = []
        all_persisted: list[dict] = []
        verified: list[dict] = []
        total = len(all_nodes)
        for index, key in enumerate(all_nodes, start=1):
            if progress is not None:
                progress(index, total, key)
            if (key.task_id, key.cluster_id, key.version) in invalid_set:
                node = self.dispatch_node(key)
                recomputed_nodes.append(node)
            else:
                node = self.compute_node_locally(key)
                stored = self.ledger.get_node(key.task_id, key.cluster_id,
                                              key.version)
                if stored is not None:
                    if stored["content_digest"] != node["content_digest"]:
                        raise DiagError(
                            "E_PREV",
                            "unaffected node digest changed during incremental apply",
                            key.as_dict())
                    verified.append(key.as_dict())
            all_persisted.append(node)
        persist = self._persist(all_persisted, batch_id, head,
                                replace_keys=invalid_set)
        unchanged = [key.as_dict() for key in all_nodes
                     if (key.task_id, key.cluster_id, key.version)
                     not in invalid_set]
        byte_check = self._byte_identical_check(unchanged)
        return {
            "head": head,
            "invalidation": report.as_dict(),
            "recomputed": [key.as_dict() for key in report.invalid],
            "recompute_count": len(recomputed_nodes),
            "unchanged_verified": byte_check,
            "persist": persist,
        }

    def _byte_identical_check(self, node_refs: list[dict]) -> dict:
        checked = []
        for ref in node_refs:
            stored = self.ledger.get_node(ref["task"], ref["cluster"],
                                          ref["revision"])
            key = NodeKey(ref["task"], ref["cluster"], ref["revision"])
            expected = self.compute_node_locally(key)
            if stored is None:
                continue
            same_aggregate = stored["aggregate_json"] == canonical_json(
                expected["aggregate"])
            same_digest = stored["content_digest"] == expected["content_digest"]
            checked.append({
                "node": ref,
                "aggregate_byte_identical": same_aggregate,
                "digest_byte_identical": same_digest,
            })
        return {"count": len(checked),
                "all_byte_identical": all(
                    item["aggregate_byte_identical"] and
                    item["digest_byte_identical"] for item in checked),
                "details": checked}

    # ------------------------------------------------------------------
    # Requirement 4: authority rebuild from a root revision
    # ------------------------------------------------------------------
    def rebuild(self, root_version: int, use_services: bool = True,
                progress=None) -> dict:
        keys = self.root_chain_nodes(root_version)
        mismatches: list[dict] = []
        rebuilt: list[dict] = []
        total = len(keys)
        for index, key in enumerate(keys, start=1):
            if progress is not None:
                progress(index, total, key)
            node = self.dispatch_node(key) if use_services \
                else self.compute_node_locally(key)
            stored = self.ledger.get_node(key.task_id, key.cluster_id,
                                          key.version)
            if stored is not None:
                field_mismatch = self._compare_fields(stored, node, key)
                if field_mismatch is not None:
                    raise DiagError("E_PREV",
                                    "first inconsistent node during authority rebuild",
                                    field_mismatch)
            rebuilt.append(node)
        self._persist(rebuilt, f"rebuild-{root_version}", root_version)
        return {"root": root_version, "nodes_compared": len(keys),
                "consistent": True, "mismatches": mismatches}

    @staticmethod
    def _compare_fields(stored: dict, expected: dict, key: NodeKey):
        if stored["aggregate_json"] != canonical_json(expected["aggregate"]):
            return {**key.as_dict(), "field": "aggregate",
                    "stored": json.loads(stored["aggregate_json"]),
                    "expected": expected["aggregate"]}
        if stored["content_digest"] != expected["content_digest"]:
            return {**key.as_dict(), "field": "content_digest",
                    "stored": stored["content_digest"],
                    "expected": expected["content_digest"]}
        if stored["parent"] != expected["parent"]:
            return {**key.as_dict(), "field": "parent",
                    "stored": stored["parent"], "expected": expected["parent"]}
        return None

    # ------------------------------------------------------------------
    # Requirement 6: rollback with stale-node isolation
    # ------------------------------------------------------------------
    def rollback(self, target_version: int, progress=None) -> dict:
        versions_on_chain = set(self.spec.ancestors_including(target_version))
        all_committed = [n for n in self.ledger.list_nodes("committed")]
        isolated = []
        before_digests = {}
        for node in all_committed:
            before_digests[(node["task_id"], node["cluster_id"],
                            node["version"])] = node["content_digest"]
            if node["version"] not in versions_on_chain:
                isolated.append({"task": node["task_id"],
                                 "cluster": node["cluster_id"],
                                 "revision": node["version"],
                                 "digest": node["content_digest"]})
        keys = self.root_chain_nodes(target_version)
        to_recompute = []
        digest_comparison = []
        total = len(keys)
        rebuilt: list[dict] = []
        for index, key in enumerate(keys, start=1):
            if progress is not None:
                progress(index, total, key)
            node = self.dispatch_node(key)
            rebuilt.append(node)
            stored = self.ledger.get_node(key.task_id, key.cluster_id,
                                          key.version)
            ref = key.as_tuple()
            if stored is None or stored["status"] != "committed":
                to_recompute.append(key.as_dict())
            before = before_digests.get(ref)
            digest_comparison.append({
                "node": key.as_dict(),
                "before_digest": before,
                "after_digest": node["content_digest"],
                "identical": before == node["content_digest"],
            })
        replace_keys = {(n["task_id"], n["cluster_id"], n["version"])
                        for n in rebuilt}
        self._persist(rebuilt, f"rollback-{target_version}", target_version,
                      replace_keys=replace_keys)
        return {
            "new_root": target_version,
            "chain_versions": sorted(versions_on_chain),
            "isolated_nodes": sorted(isolated,
                                     key=lambda n: (n["revision"], n["task"],
                                                    n["cluster"])),
            "recomputed_nodes": to_recompute or [k.as_dict() for k in keys],
            "digest_comparison": digest_comparison,
        }

    def active_view(self, root_version: int | None = None) -> dict:
        root = self.ledger.get_root() if root_version is None else root_version
        if root is None:
            return {"root": None, "active": [], "isolated": []}
        chain = set(self.spec.ancestors_including(root))
        active, isolated = [], []
        for node in self.ledger.list_nodes("committed"):
            entry = {"task": node["task_id"], "cluster": node["cluster_id"],
                     "revision": node["version"],
                     "digest": node["content_digest"],
                     "aggregate": json.loads(node["aggregate_json"])}
            (active if node["version"] in chain else isolated).append(entry)
        return {"root": root, "active": active, "isolated": isolated}

    # ------------------------------------------------------------------
    # Requirement 7: crash recovery
    # ------------------------------------------------------------------
    def recover(self) -> dict:
        staging = self.ledger.staging_batches()
        unknown = [{"task": n["task_id"], "cluster": n["cluster_id"],
                    "revision": n["version"]}
                   for n in self.ledger.list_unknown_nodes()]
        root = self.ledger.get_root()
        view = self.active_view(root) if root is not None else \
            {"root": None, "active": [], "isolated": []}
        return {
            "staging_batches": staging,
            "unknown_nodes": unknown,
            "root": root,
            "active_view": view,
            "requires_rerun": bool(unknown or staging),
            "exit_report_available": False,
        }

    def rerun_unknown(self) -> dict:
        unknown = self.ledger.list_unknown_nodes()
        rebuilt = []
        for row in unknown:
            key = NodeKey(row["task_id"], row["cluster_id"], row["version"])
            self.ledger.delete_node(key.task_id, key.cluster_id, key.version)
            node = self.dispatch_node(key)
            rebuilt.append(node)
        if rebuilt:
            self._persist(rebuilt, "recovery-rerun", self.ledger.get_root(),
                          replace_keys={(n["task_id"], n["cluster_id"],
                                         n["version"]) for n in rebuilt})
        return {"rerun": [{"task": n["task_id"], "cluster": n["cluster_id"],
                           "revision": n["version"]} for n in unknown]}

    # ------------------------------------------------------------------
    # Requirement 8: replay verification and reference checks
    # ------------------------------------------------------------------
    def validate_references(self) -> None:
        versions = {r.version for r in self.spec.revisions}
        for rev in self.spec.revisions:
            if rev.parent is not None and rev.parent not in versions:
                raise_ref(
                    f"dangling parent reference {rev.parent} on revision "
                    f"{rev.version}",
                    {"field": "revisions.parent", "revision": rev.version,
                     "reference": rev.parent})
            for derived in rev.derives:
                if derived not in versions:
                    raise_ref(
                        f"dangling derived reference {derived} on revision "
                        f"{rev.version}",
                        {"field": "revisions.derives", "revision": rev.version,
                         "reference": derived})

    def replay(self, snapshot: dict, use_services: bool = True,
               progress=None) -> dict:
        self.validate_references()
        root = snapshot["root_version"]
        expected_nodes = snapshot["nodes"]
        total = len(expected_nodes)
        for index, snap_node in enumerate(expected_nodes, start=1):
            key = NodeKey(snap_node["task_id"], snap_node["cluster_id"],
                          snap_node["version"])
            if progress is not None:
                progress(index, total, key)
            actual = self.dispatch_node(key) if use_services \
                else self.compute_node_locally(key)
            mismatch = self._replay_field_mismatch(snap_node, actual, key)
            if mismatch is not None:
                raise DiagError("E_REPLAY",
                                "first non-reproducible node during replay",
                                mismatch)
        return {"root": root, "nodes_replayed": total, "reproducible": True}

    @staticmethod
    def _replay_field_mismatch(snap_node: dict, actual: dict, key: NodeKey):
        if canonical_json(snap_node["aggregate"]) != \
                canonical_json(actual["aggregate"]):
            return {**key.as_dict(), "field": "aggregate",
                    "snapshot": snap_node["aggregate"],
                    "recomputed": actual["aggregate"]}
        if snap_node["content_digest"] != actual["content_digest"]:
            return {**key.as_dict(), "field": "content_digest",
                    "snapshot": snap_node["content_digest"],
                    "recomputed": actual["content_digest"]}
        if snap_node.get("parent") != actual["parent"]:
            return {**key.as_dict(), "field": "parent",
                    "snapshot": snap_node.get("parent"),
                    "recomputed": actual["parent"]}
        return None
