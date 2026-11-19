"""Persistent directed dependency graph and deterministic invalidation."""

from __future__ import annotations

import heapq
import json
import sqlite3
import uuid
from collections import defaultdict, deque
from typing import Any, Iterable

from .errors import E_INTERNAL, BusinessError
from .util import digest_text, mono_ms


KIND_ORDER = {
    "chunk": 10,
    "manifest": 20,
    "metadata_stage": 30,
    "metadata": 30,
    "object_key": 40,
    "prefix_view": 50,
    "session_state": 5,
    "session_summary": 60,
}


class DependencyGraph:
    """A small persistent DAG whose edges point dependency -> dependent."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def graph_version(self) -> int:
        row = self.conn.execute("SELECT graph_version FROM graph_meta WHERE id=1").fetchone()
        return int(row["graph_version"])

    def _bump_graph_version(self, amount: int = 1) -> int:
        self.conn.execute(
            "UPDATE graph_meta SET graph_version=graph_version+? WHERE id=1",
            (amount,),
        )
        return self.graph_version()

    def node(self, node_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM graph_nodes WHERE node_id=?", (node_id,)
        ).fetchone()
        return self._node_dict(row) if row else None

    def ensure_node(
        self,
        node_id: str,
        kind: str,
        ref_id: str,
        digest: str | None,
        payload: dict[str, Any],
        *,
        valid: bool = True,
    ) -> dict[str, Any]:
        existing = self.node(node_id)
        if existing is not None:
            return existing
        self.conn.execute(
            """
            INSERT INTO graph_nodes(
                node_id, kind, ref_id, entry_version, cached_digest,
                valid, updated_mono_ms, payload_json
            ) VALUES(?,?,?,?,?,?,?,?)
            """,
            (
                node_id,
                kind,
                ref_id,
                1,
                digest,
                1 if valid and digest is not None else 0,
                mono_ms(),
                json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            ),
        )
        self._bump_graph_version()
        result = self.node(node_id)
        assert result is not None
        return result

    def refresh_node(
        self,
        node_id: str,
        digest: str,
        *,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        existing = self.node(node_id)
        if existing is None:
            raise BusinessError(E_INTERNAL, "cannot refresh an unknown graph node", {"node_id": node_id})
        payload_json = existing["payload_json"]
        if payload is not None:
            payload_json = json.dumps(
                payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
        if (
            existing["valid"] == 1
            and existing["cached_digest"] == digest
            and existing["payload_json"] == payload_json
        ):
            return {**existing, "changed": False}
        self.conn.execute(
            """
            UPDATE graph_nodes
               SET entry_version=entry_version+1,
                   cached_digest=?, valid=1, updated_mono_ms=?, payload_json=?
             WHERE node_id=?
            """,
            (digest, mono_ms(), payload_json, node_id),
        )
        self._bump_graph_version()
        updated = self.node(node_id)
        assert updated is not None
        return {**updated, "changed": True}

    def add_edge(self, dependency_id: str, dependent_id: str) -> bool:
        if dependency_id == dependent_id:
            raise BusinessError(
                E_INTERNAL,
                "dependency graph self-edge rejected",
                {"node_id": dependency_id},
            )
        missing = [
            node_id
            for node_id in (dependency_id, dependent_id)
            if self.node(node_id) is None
        ]
        if missing:
            raise BusinessError(
                E_INTERNAL,
                "dependency graph edge references a missing node",
                {"missing": missing, "dependency": dependency_id, "dependent": dependent_id},
            )
        if self._exists_edge(dependency_id, dependent_id):
            return False
        if self._reachable(dependent_id, dependency_id):
            raise BusinessError(
                E_INTERNAL,
                "dependency graph cycle rejected",
                {"dependency": dependency_id, "dependent": dependent_id},
            )
        self.conn.execute(
            "INSERT INTO graph_edges(dependency_id, dependent_id) VALUES(?,?)",
            (dependency_id, dependent_id),
        )
        self._bump_graph_version()
        return True

    def remove_edge(self, dependency_id: str, dependent_id: str) -> bool:
        cursor = self.conn.execute(
            "DELETE FROM graph_edges WHERE dependency_id=? AND dependent_id=?",
            (dependency_id, dependent_id),
        )
        if cursor.rowcount:
            self._bump_graph_version()
            return True
        return False

    def _exists_edge(self, dependency_id: str, dependent_id: str) -> bool:
        return (
            self.conn.execute(
                "SELECT 1 FROM graph_edges WHERE dependency_id=? AND dependent_id=?",
                (dependency_id, dependent_id),
            ).fetchone()
            is not None
        )

    def _outgoing(self) -> dict[str, list[str]]:
        outgoing: dict[str, list[str]] = defaultdict(list)
        for row in self.conn.execute("SELECT dependency_id, dependent_id FROM graph_edges"):
            outgoing[row["dependency_id"]].append(row["dependent_id"])
        return outgoing

    def _reachable(self, start: str, target: str) -> bool:
        outgoing = self._outgoing()
        queue = deque([start])
        seen = {start}
        while queue:
            current = queue.popleft()
            if current == target:
                return True
            for child in outgoing.get(current, []):
                if child not in seen:
                    seen.add(child)
                    queue.append(child)
        return False

    def invalidate(
        self,
        trigger_node_ids: Iterable[str],
        *,
        trigger_type: str,
        reason: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        roots = sorted(set(trigger_node_ids))
        for node_id in roots:
            if self.node(node_id) is None:
                raise BusinessError(
                    E_INTERNAL,
                    "invalidation trigger references a missing node",
                    {"node_id": node_id},
                )

        outgoing = self._outgoing()
        depth: dict[str, int] = {}
        queue: deque[tuple[str, int]] = deque((node_id, 0) for node_id in roots)
        for node_id in roots:
            depth[node_id] = 0
        while queue:
            current, current_depth = queue.popleft()
            for child in outgoing.get(current, []):
                candidate_depth = current_depth + 1
                if child not in depth or candidate_depth < depth[child]:
                    depth[child] = candidate_depth
                    queue.append((child, candidate_depth))

        affected = set(depth)
        indegree = {node_id: 0 for node_id in affected}
        for dependency_id in affected:
            for dependent_id in outgoing.get(dependency_id, []):
                if dependent_id in affected:
                    indegree[dependent_id] += 1

        def sort_key(node_id: str) -> tuple[int, int, str]:
            row = self.node(node_id)
            assert row is not None
            return (depth[node_id], KIND_ORDER.get(row["kind"], 999), node_id)

        heap: list[tuple[tuple[int, int, str], str]] = []
        for node_id, degree in indegree.items():
            if degree == 0:
                heapq.heappush(heap, (sort_key(node_id), node_id))
        ordered: list[dict[str, Any]] = []
        while heap:
            _, node_id = heapq.heappop(heap)
            before = self.node(node_id)
            assert before is not None
            ordered.append(
                {
                    "node_id": node_id,
                    "kind": before["kind"],
                    "ref_id": before["ref_id"],
                    "depth": depth[node_id],
                    "direct": depth[node_id] == 0,
                    "entry_version_before": before["entry_version"],
                    "entry_version_after": before["entry_version"] + 1,
                    "digest_before": before["cached_digest"],
                    "digest_after": None,
                    "valid_after": False,
                }
            )
            for dependent_id in outgoing.get(node_id, []):
                if dependent_id in affected:
                    indegree[dependent_id] -= 1
                    if indegree[dependent_id] == 0:
                        heapq.heappush(heap, (sort_key(dependent_id), dependent_id))

        if len(ordered) != len(affected):
            raise BusinessError(E_INTERNAL, "dependency graph contains a cycle during invalidation")

        now = mono_ms()
        for entry in ordered:
            self.conn.execute(
                """
                UPDATE graph_nodes
                   SET entry_version=entry_version+1,
                       cached_digest=NULL, valid=0, updated_mono_ms=?
                 WHERE node_id=?
                """,
                (now, entry["node_id"]),
            )
        graph_version = self._bump_graph_version()
        event_id = "inv-" + uuid.uuid4().hex[:20]
        event = {
            "event_id": event_id,
            "mono_ms": now,
            "trigger_type": trigger_type,
            "trigger_sources": roots,
            "reason": reason,
            "context": context or {},
            "affected": ordered,
            "affected_count": len(ordered),
            "direct_count": sum(1 for entry in ordered if entry["direct"]),
            "indirect_count": sum(1 for entry in ordered if not entry["direct"]),
            "graph_version": graph_version,
        }
        self.conn.execute(
            """
            INSERT INTO graph_invalidation_events(
                event_id, mono_ms, trigger_type, trigger_sources_json,
                reason, context_json, affected_json, graph_version
            ) VALUES(?,?,?,?,?,?,?,?)
            """,
            (
                event_id,
                now,
                trigger_type,
                json.dumps(roots, ensure_ascii=False, separators=(",", ":")),
                reason,
                json.dumps(context or {}, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                json.dumps(ordered, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                graph_version,
            ),
        )
        return event

    def validate(self) -> dict[str, Any]:
        nodes = {
            row["node_id"]: row
            for row in self.conn.execute("SELECT * FROM graph_nodes")
        }
        outgoing: dict[str, list[str]] = defaultdict(list)
        indegree = {node_id: 0 for node_id in nodes}
        edge_count = 0
        for row in self.conn.execute("SELECT dependency_id, dependent_id FROM graph_edges"):
            dependency_id = row["dependency_id"]
            dependent_id = row["dependent_id"]
            missing = [
                node_id
                for node_id in (dependency_id, dependent_id)
                if node_id not in nodes
            ]
            if missing:
                raise BusinessError(
                    E_INTERNAL,
                    "dependency graph has a dangling reference",
                    {"edge": [dependency_id, dependent_id], "missing": missing},
                )
            if dependency_id == dependent_id:
                raise BusinessError(
                    E_INTERNAL,
                    "dependency graph has a self reference",
                    {"node_id": dependency_id},
                )
            outgoing[dependency_id].append(dependent_id)
            indegree[dependent_id] += 1
            edge_count += 1

        queue = deque(sorted(node_id for node_id, degree in indegree.items() if degree == 0))
        visited = 0
        while queue:
            node_id = queue.popleft()
            visited += 1
            for child in outgoing.get(node_id, []):
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)
        if visited != len(nodes):
            cycle_nodes = sorted(node_id for node_id, degree in indegree.items() if degree > 0)
            raise BusinessError(
                E_INTERNAL,
                "dependency graph contains a cycle",
                {"cycle_nodes": cycle_nodes},
            )
        return {
            "valid": True,
            "graph_version": self.graph_version(),
            "node_count": len(nodes),
            "edge_count": edge_count,
            "nodes": len(nodes),
            "edges": edge_count,
        }

    def snapshot(self) -> dict[str, Any]:
        nodes = [self._node_dict(row) for row in self.conn.execute("SELECT * FROM graph_nodes ORDER BY node_id")]
        edges = [
            {"dependency_id": row["dependency_id"], "dependent_id": row["dependent_id"]}
            for row in self.conn.execute(
                "SELECT dependency_id, dependent_id FROM graph_edges ORDER BY dependency_id, dependent_id"
            )
        ]
        return {"graph_version": self.graph_version(), "nodes": nodes, "edges": edges}

    def graph_payload(self, *, include_events: bool = False) -> dict[str, Any]:
        nodes = [self._node_dict(row) for row in self.conn.execute("SELECT * FROM graph_nodes ORDER BY node_id")]
        edges = [
            {"dependency_id": row["dependency_id"], "dependent_id": row["dependent_id"]}
            for row in self.conn.execute(
                "SELECT dependency_id, dependent_id FROM graph_edges ORDER BY dependency_id, dependent_id"
            )
        ]
        result: dict[str, Any] = {
            "graph_version": self.graph_version(),
            "nodes": nodes,
            "edges": edges,
            "validation": self.validate(),
        }
        if include_events:
            result["events"] = [
                self._event_dict(row)
                for row in self.conn.execute(
                    "SELECT * FROM graph_invalidation_events ORDER BY id"
                )
            ]
        return result

    def latest_events(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM graph_invalidation_events ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [self._event_dict(row) for row in reversed(rows)]

    def node_digest_for_key(self, object_key: str, active_version: int, metadata_digest: str | None, status: str) -> str:
        return digest_text(
            json.dumps(
                {
                    "object_key": object_key,
                    "active_version": active_version,
                    "metadata_digest": metadata_digest,
                    "status": status,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )

    def _node_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        payload = json.loads(row["payload_json"])
        return {
            "node_id": row["node_id"],
            "kind": row["kind"],
            "ref_id": row["ref_id"],
            "entry_version": row["entry_version"],
            "cached_digest": row["cached_digest"],
            "valid": bool(row["valid"]),
            "updated_mono_ms": row["updated_mono_ms"],
            "payload": payload,
            "payload_json": row["payload_json"],
        }

    def _event_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "event_id": row["event_id"],
            "mono_ms": row["mono_ms"],
            "trigger_type": row["trigger_type"],
            "trigger_sources": json.loads(row["trigger_sources_json"]),
            "reason": row["reason"],
            "context": json.loads(row["context_json"]),
            "affected": json.loads(row["affected_json"]),
            "graph_version": row["graph_version"],
        }


SCHEMA_GRAPH = """
CREATE TABLE IF NOT EXISTS graph_meta(
    id INTEGER PRIMARY KEY CHECK(id=1),
    graph_version INTEGER NOT NULL
);
INSERT OR IGNORE INTO graph_meta(id, graph_version) VALUES(1, 1);

CREATE TABLE IF NOT EXISTS graph_nodes(
    node_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    ref_id TEXT NOT NULL,
    entry_version INTEGER NOT NULL,
    cached_digest TEXT,
    valid INTEGER NOT NULL,
    updated_mono_ms INTEGER NOT NULL,
    payload_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS graph_edges(
    dependency_id TEXT NOT NULL,
    dependent_id TEXT NOT NULL,
    PRIMARY KEY(dependency_id, dependent_id),
    FOREIGN KEY(dependency_id) REFERENCES graph_nodes(node_id),
    FOREIGN KEY(dependent_id) REFERENCES graph_nodes(node_id)
);
CREATE INDEX IF NOT EXISTS idx_graph_edges_dependent ON graph_edges(dependent_id);

CREATE TABLE IF NOT EXISTS graph_invalidation_events(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL UNIQUE,
    mono_ms INTEGER NOT NULL,
    trigger_type TEXT NOT NULL,
    trigger_sources_json TEXT NOT NULL,
    reason TEXT NOT NULL,
    context_json TEXT NOT NULL,
    affected_json TEXT NOT NULL,
    graph_version INTEGER NOT NULL
);
"""

