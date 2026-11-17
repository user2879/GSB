from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .errors import DiagError, E_LEDGER
from .util import canonical_json, node_content_digest, tree_snapshot_digest


@dataclass(frozen=True)
class LedgerNode:
    cluster_id: str
    revision: int
    task_id: str
    parent_revision: int | None
    aggregate_summary_canonical: str
    content_digest: str
    completed_at_ns: int

    @property
    def key(self) -> str:
        return f"{self.cluster_id}|{self.revision}"

    @property
    def aggregate_summary(self) -> dict[str, Any]:
        return json.loads(self.aggregate_summary_canonical)

    def as_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "revision": self.revision,
            "task_id": self.task_id,
            "parent_revision": self.parent_revision,
            "aggregate_summary": self.aggregate_summary,
            "aggregate_summary_canonical": self.aggregate_summary_canonical,
            "content_digest": self.content_digest,
            "completed_at_ns": self.completed_at_ns,
        }


@dataclass(frozen=True)
class CommitInfo:
    revision: int
    parent_revision: int | None
    committed_at_ns: int
    snapshot_digest: str


@dataclass(frozen=True)
class TreeSnapshot:
    root_revision: int | None
    effective_nodes: dict[str, LedgerNode]
    history: dict[str, tuple[LedgerNode, ...]]
    committed_at_ns: int | None
    snapshot_digest: str | None

    @property
    def node_count(self) -> int:
        return sum(len(nodes) for nodes in self.history.values())

    @property
    def all_nodes(self) -> tuple[LedgerNode, ...]:
        nodes = [node for chain in self.history.values() for node in chain]
        return tuple(sorted(nodes, key=lambda node: (node.cluster_id, node.revision)))

    def node_digest_map(self) -> dict[str, str]:
        return {node.key: node.content_digest for node in self.all_nodes}

    def effective_digest_map(self) -> dict[str, str]:
        return {cluster_id: node.content_digest for cluster_id, node in sorted(self.effective_nodes.items())}


@dataclass(frozen=True)
class NodeCandidate:
    cluster_id: str
    task_id: str
    aggregate_summary_canonical: str


@dataclass(frozen=True)
class LatencyEvent:
    id: int
    session_id: str
    kind: str
    cluster_id: str | None
    revision: int | None
    bound_root_revision: int | None
    started_ns: int
    ended_ns: int
    elapsed_us: int
    elapsed_ms_x1000: int
    budget_ms_x1000: int | None
    exceeded: bool
    clock_basis: str
    truth_source: str
    details: dict[str, Any]
    error_code: str | None


_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=FULL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS commits (
    revision INTEGER PRIMARY KEY CHECK (revision > 0),
    parent_revision INTEGER,
    committed_at_ns INTEGER NOT NULL,
    snapshot_digest TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS roots (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    root_revision INTEGER NOT NULL,
    updated_at_ns INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS nodes (
    cluster_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    task_id TEXT NOT NULL,
    parent_revision INTEGER,
    aggregate_summary TEXT NOT NULL,
    content_digest TEXT NOT NULL,
    completed_at_ns INTEGER NOT NULL,
    PRIMARY KEY (cluster_id, revision)
);

CREATE INDEX IF NOT EXISTS idx_nodes_revision ON nodes(revision);
CREATE INDEX IF NOT EXISTS idx_nodes_cluster_revision ON nodes(cluster_id, revision);

CREATE TABLE IF NOT EXISTS latency_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('ui_query', 'authoritative')),
    cluster_id TEXT,
    revision INTEGER,
    bound_root_revision INTEGER,
    started_ns INTEGER NOT NULL,
    ended_ns INTEGER NOT NULL,
    elapsed_us INTEGER NOT NULL,
    elapsed_ms_x1000 INTEGER NOT NULL,
    budget_ms_x1000 INTEGER,
    exceeded INTEGER NOT NULL CHECK (exceeded IN (0, 1)),
    clock_basis TEXT NOT NULL,
    truth_source TEXT NOT NULL,
    details_json TEXT NOT NULL,
    error_code TEXT
);

CREATE INDEX IF NOT EXISTS idx_latency_root ON latency_events(bound_root_revision, kind);
"""


class Ledger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path,
            timeout=15.0,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=15000")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        connection = self._connect()
        try:
            connection.executescript(_SCHEMA)
            connection.execute("PRAGMA user_version=1")
        finally:
            connection.close()

    @staticmethod
    def _ledger_error(
        message: str,
        *,
        cluster_id: str | None = None,
        revision: int | None = None,
        field: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> DiagError:
        merged = dict(details or {})
        if cluster_id is not None:
            merged["cluster_id"] = cluster_id
        if revision is not None:
            merged["revision"] = revision
        if field is not None:
            merged["field"] = field
        return DiagError(E_LEDGER, message, details=merged)

    @staticmethod
    def _row_to_node(row: sqlite3.Row) -> LedgerNode:
        return LedgerNode(
            cluster_id=row["cluster_id"],
            revision=row["revision"],
            task_id=row["task_id"],
            parent_revision=row["parent_revision"],
            aggregate_summary_canonical=row["aggregate_summary"],
            content_digest=row["content_digest"],
            completed_at_ns=row["completed_at_ns"],
        )

    @staticmethod
    def _validate_node(node: LedgerNode) -> None:
        try:
            decoded = json.loads(node.aggregate_summary_canonical)
            recanonical = canonical_json(decoded)
        except Exception as exc:
            raise Ledger._ledger_error(
                "node aggregate summary is not valid canonical JSON",
                cluster_id=node.cluster_id,
                revision=node.revision,
                field="aggregate_summary",
            ) from exc
        if recanonical != node.aggregate_summary_canonical:
            raise Ledger._ledger_error(
                "node aggregate summary is not canonical JSON",
                cluster_id=node.cluster_id,
                revision=node.revision,
                field="aggregate_summary",
            )
        expected = node_content_digest(
            task_id=node.task_id,
            cluster_id=node.cluster_id,
            revision=node.revision,
            parent_revision=node.parent_revision,
            aggregate_summary_canonical=node.aggregate_summary_canonical,
        )
        if expected != node.content_digest:
            raise Ledger._ledger_error(
                "node content digest does not match its committed content",
                cluster_id=node.cluster_id,
                revision=node.revision,
                field="content_digest",
                details={"stored": node.content_digest, "computed": expected},
            )

    def root_revision(self) -> int | None:
        connection = self._connect()
        try:
            row = connection.execute("SELECT root_revision FROM roots WHERE singleton=1").fetchone()
            return None if row is None else int(row["root_revision"])
        finally:
            connection.close()

    def latest_committed_revision(self) -> int | None:
        connection = self._connect()
        try:
            row = connection.execute("SELECT MAX(revision) AS revision FROM commits").fetchone()
            return None if row is None or row["revision"] is None else int(row["revision"])
        finally:
            connection.close()

    def list_commits(self) -> list[CommitInfo]:
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT revision, parent_revision, committed_at_ns, snapshot_digest FROM commits ORDER BY revision"
            ).fetchall()
            return [
                CommitInfo(
                    revision=int(row["revision"]),
                    parent_revision=row["parent_revision"],
                    committed_at_ns=int(row["committed_at_ns"]),
                    snapshot_digest=str(row["snapshot_digest"]),
                )
                for row in rows
            ]
        finally:
            connection.close()

    def latest_node_revision(self, cluster_id: str, *, before_revision: int | None = None) -> int | None:
        connection = self._connect()
        try:
            if before_revision is None:
                row = connection.execute(
                    "SELECT MAX(revision) AS revision FROM nodes WHERE cluster_id=?",
                    (cluster_id,),
                ).fetchone()
            else:
                row = connection.execute(
                    "SELECT MAX(revision) AS revision FROM nodes WHERE cluster_id=? AND revision<?",
                    (cluster_id, before_revision),
                ).fetchone()
            return None if row is None or row["revision"] is None else int(row["revision"])
        finally:
            connection.close()

    def get_node(self, cluster_id: str, revision: int) -> LedgerNode | None:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT * FROM nodes WHERE cluster_id=? AND revision=?",
                (cluster_id, revision),
            ).fetchone()
            if row is None:
                return None
            node = self._row_to_node(row)
            self._validate_node(node)
            return node
        finally:
            connection.close()

    def commit_revision(
        self,
        *,
        revision: int,
        parent_revision: int | None,
        candidates: Iterable[NodeCandidate],
        required_clusters: set[str],
        completed_at_ns: int | None = None,
    ) -> TreeSnapshot:
        candidate_list = list(candidates)
        if not candidate_list:
            raise DiagError(E_LEDGER, "a revision commit must contain at least one node")
        candidate_clusters = [candidate.cluster_id for candidate in candidate_list]
        if len(candidate_clusters) != len(set(candidate_clusters)):
            raise DiagError(E_LEDGER, "duplicate cluster candidate in one revision commit")

        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            root_row = connection.execute("SELECT root_revision FROM roots WHERE singleton=1").fetchone()
            current_root = None if root_row is None else int(root_row["root_revision"])
            if current_root != parent_revision:
                raise self._ledger_error(
                    "root pointer changed or does not match the expected parent revision",
                    revision=revision,
                    field="root_revision",
                    details={"current_root": current_root, "expected_parent": parent_revision},
                )
            existing_commit = connection.execute(
                "SELECT revision FROM commits WHERE revision=?", (revision,)
            ).fetchone()
            if existing_commit is not None:
                raise self._ledger_error(
                    "revision is already committed",
                    revision=revision,
                    field="revision",
                )

            committed_at = completed_at_ns if completed_at_ns is not None else time.time_ns()
            node_records: list[dict[str, Any]] = []
            for candidate in candidate_list:
                try:
                    decoded = json.loads(candidate.aggregate_summary_canonical)
                    recanonical = canonical_json(decoded)
                except Exception as exc:
                    raise self._ledger_error(
                        "candidate aggregate summary is invalid",
                        cluster_id=candidate.cluster_id,
                        revision=revision,
                        field="aggregate_summary",
                    ) from exc
                if recanonical != candidate.aggregate_summary_canonical:
                    raise self._ledger_error(
                        "candidate aggregate summary is not canonical",
                        cluster_id=candidate.cluster_id,
                        revision=revision,
                        field="aggregate_summary",
                    )
                parent_row = connection.execute(
                    "SELECT MAX(revision) AS revision FROM nodes WHERE cluster_id=? AND revision<?",
                    (candidate.cluster_id, revision),
                ).fetchone()
                node_parent = (
                    None
                    if parent_row is None or parent_row["revision"] is None
                    else int(parent_row["revision"])
                )
                if parent_revision is not None and node_parent is None:
                    raise self._ledger_error(
                        "cluster has no previous node for a non-initial revision",
                        cluster_id=candidate.cluster_id,
                        revision=revision,
                        field="parent_revision",
                    )
                digest = node_content_digest(
                    task_id=candidate.task_id,
                    cluster_id=candidate.cluster_id,
                    revision=revision,
                    parent_revision=node_parent,
                    aggregate_summary_canonical=candidate.aggregate_summary_canonical,
                )
                node_records.append(
                    {
                        "cluster_id": candidate.cluster_id,
                        "revision": revision,
                        "task_id": candidate.task_id,
                        "parent_revision": node_parent,
                        "aggregate_summary": candidate.aggregate_summary_canonical,
                        "content_digest": digest,
                        "completed_at_ns": committed_at,
                    }
                )
                connection.execute(
                    """
                    INSERT INTO nodes(
                        cluster_id, revision, task_id, parent_revision,
                        aggregate_summary, content_digest, completed_at_ns
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        candidate.cluster_id,
                        revision,
                        candidate.task_id,
                        node_parent,
                        candidate.aggregate_summary_canonical,
                        digest,
                        committed_at,
                    ),
                )

            effective_rows = connection.execute(
                """
                SELECT n.* FROM nodes n
                JOIN (
                    SELECT cluster_id, MAX(revision) AS revision
                    FROM nodes WHERE revision<=?
                    GROUP BY cluster_id
                ) latest
                ON n.cluster_id=latest.cluster_id AND n.revision=latest.revision
                """,
                (revision,),
            ).fetchall()
            effective = [self._row_to_node(row) for row in effective_rows]
            effective_map = {node.cluster_id: node for node in effective}
            missing = sorted(required_clusters - set(effective_map))
            if missing:
                raise self._ledger_error(
                    "version tree would be incomplete after committing this revision",
                    revision=revision,
                    field="root_revision",
                    details={"missing_clusters": missing},
                )
            snapshot_digest = tree_snapshot_digest(
                revision,
                [node.as_dict() for node in effective],
            )
            connection.execute(
                "INSERT INTO commits(revision, parent_revision, committed_at_ns, snapshot_digest) VALUES(?,?,?,?)",
                (revision, parent_revision, committed_at, snapshot_digest),
            )
            connection.execute(
                """
                INSERT INTO roots(singleton, root_revision, updated_at_ns) VALUES(1, ?, ?)
                ON CONFLICT(singleton) DO UPDATE SET
                    root_revision=excluded.root_revision,
                    updated_at_ns=excluded.updated_at_ns
                """,
                (revision, committed_at),
            )
            connection.execute("COMMIT")
        except sqlite3.IntegrityError as exc:
            connection.execute("ROLLBACK")
            raise self._ledger_error(
                "ledger uniqueness constraint rejected the revision commit",
                revision=revision,
                details={"sqlite_error": str(exc)},
            ) from exc
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

        return self.load_snapshot(revision, verify_parents=True, verify_snapshot_digest=True)

    def load_snapshot(
        self,
        root_revision: int,
        *,
        verify_parents: bool = True,
        verify_snapshot_digest: bool = True,
    ) -> TreeSnapshot:
        connection = self._connect()
        try:
            commit = connection.execute(
                "SELECT revision, parent_revision, committed_at_ns, snapshot_digest FROM commits WHERE revision=?",
                (root_revision,),
            ).fetchone()
            if commit is None:
                raise self._ledger_error(
                    "root revision has no committed tree",
                    revision=root_revision,
                    field="root_revision",
                )
            rows = connection.execute(
                "SELECT * FROM nodes WHERE revision<=? ORDER BY cluster_id, revision",
                (root_revision,),
            ).fetchall()
        finally:
            connection.close()

        grouped: dict[str, list[LedgerNode]] = {}
        for row in rows:
            node = self._row_to_node(row)
            self._validate_node(node)
            grouped.setdefault(node.cluster_id, []).append(node)

        history: dict[str, tuple[LedgerNode, ...]] = {}
        effective: dict[str, LedgerNode] = {}
        for cluster_id, nodes in sorted(grouped.items()):
            ordered = tuple(sorted(nodes, key=lambda node: node.revision))
            if verify_parents:
                for index, node in enumerate(ordered):
                    expected_parent = None if index == 0 else ordered[index - 1].revision
                    if node.parent_revision != expected_parent:
                        raise self._ledger_error(
                            "node parent reference does not match the cluster revision chain",
                            cluster_id=cluster_id,
                            revision=node.revision,
                            field="parent_revision",
                            details={"expected": expected_parent, "stored": node.parent_revision},
                        )
            history[cluster_id] = ordered
            effective[cluster_id] = ordered[-1]

        if verify_parents:
            commits = self.list_commits()
            visible_commits = [item for item in commits if item.revision <= root_revision]
            for index, item in enumerate(visible_commits):
                expected_parent = None if index == 0 else visible_commits[index - 1].revision
                if item.parent_revision != expected_parent:
                    raise self._ledger_error(
                        "commit parent reference does not match the revision chain",
                        revision=item.revision,
                        field="parent_revision",
                        details={"expected": expected_parent, "stored": item.parent_revision},
                    )

        digest = tree_snapshot_digest(
            root_revision,
            [node.as_dict() for node in effective.values()],
        )
        if verify_snapshot_digest and digest != commit["snapshot_digest"]:
            raise self._ledger_error(
                "stored root snapshot digest does not match the reconstructed version tree",
                revision=root_revision,
                field="snapshot_digest",
                details={"stored": commit["snapshot_digest"], "computed": digest},
            )

        return TreeSnapshot(
            root_revision=root_revision,
            effective_nodes=effective,
            history=history,
            committed_at_ns=int(commit["committed_at_ns"]),
            snapshot_digest=digest,
        )

    def load_current_snapshot(
        self,
        *,
        verify_parents: bool = True,
        verify_snapshot_digest: bool = True,
        verify_root_pointer: bool = True,
    ) -> TreeSnapshot:
        root = self.root_revision()
        if root is None:
            return TreeSnapshot(None, {}, {}, None, None)
        snapshot = self.load_snapshot(
            root,
            verify_parents=verify_parents,
            verify_snapshot_digest=verify_snapshot_digest,
        )
        if verify_root_pointer:
            latest = self.latest_committed_revision()
            if latest != root:
                raise self._ledger_error(
                    "root pointer does not point at the latest committed revision",
                    revision=root,
                    field="root_revision",
                    details={"root_revision": root, "latest_commit": latest},
                )
        return snapshot

    def record_latency_event(
        self,
        *,
        session_id: str,
        kind: str,
        cluster_id: str | None,
        revision: int | None,
        bound_root_revision: int | None,
        started_ns: int,
        ended_ns: int,
        elapsed_us: int,
        budget_ms_x1000: int | None,
        clock_basis: str,
        truth_source: str,
        details: dict[str, Any] | None = None,
        error_code: str | None = None,
    ) -> LatencyEvent:
        elapsed_us = max(0, int(elapsed_us))
        exceeded = budget_ms_x1000 is not None and elapsed_us > budget_ms_x1000
        details_json = canonical_json(details or {})
        connection = self._connect()
        try:
            cursor = connection.execute(
                """
                INSERT INTO latency_events(
                    session_id, kind, cluster_id, revision, bound_root_revision,
                    started_ns, ended_ns, elapsed_us, elapsed_ms_x1000,
                    budget_ms_x1000, exceeded, clock_basis, truth_source,
                    details_json, error_code
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    session_id,
                    kind,
                    cluster_id,
                    revision,
                    bound_root_revision,
                    int(started_ns),
                    int(ended_ns),
                    elapsed_us,
                    elapsed_us,
                    budget_ms_x1000,
                    1 if exceeded else 0,
                    clock_basis,
                    truth_source,
                    details_json,
                    error_code,
                ),
            )
            event_id = int(cursor.lastrowid)
        finally:
            connection.close()
        return LatencyEvent(
            id=event_id,
            session_id=session_id,
            kind=kind,
            cluster_id=cluster_id,
            revision=revision,
            bound_root_revision=bound_root_revision,
            started_ns=int(started_ns),
            ended_ns=int(ended_ns),
            elapsed_us=elapsed_us,
            elapsed_ms_x1000=elapsed_us,
            budget_ms_x1000=budget_ms_x1000,
            exceeded=exceeded,
            clock_basis=clock_basis,
            truth_source=truth_source,
            details=json.loads(details_json),
            error_code=error_code,
        )

    def latency_events(
        self,
        *,
        bound_root_revision: int | None,
        kind: str | None = None,
    ) -> list[LatencyEvent]:
        connection = self._connect()
        try:
            if bound_root_revision is None:
                query = "SELECT * FROM latency_events WHERE bound_root_revision IS NULL"
                parameters: tuple[Any, ...] = ()
            else:
                query = "SELECT * FROM latency_events WHERE bound_root_revision=?"
                parameters = (bound_root_revision,)
            if kind is not None:
                query += " AND kind=?"
                parameters += (kind,)
            query += " ORDER BY id"
            rows = connection.execute(query, parameters).fetchall()
        finally:
            connection.close()
        return [
            LatencyEvent(
                id=int(row["id"]),
                session_id=str(row["session_id"]),
                kind=str(row["kind"]),
                cluster_id=row["cluster_id"],
                revision=row["revision"],
                bound_root_revision=row["bound_root_revision"],
                started_ns=int(row["started_ns"]),
                ended_ns=int(row["ended_ns"]),
                elapsed_us=int(row["elapsed_us"]),
                elapsed_ms_x1000=int(row["elapsed_ms_x1000"]),
                budget_ms_x1000=row["budget_ms_x1000"],
                exceeded=bool(row["exceeded"]),
                clock_basis=str(row["clock_basis"]),
                truth_source=str(row["truth_source"]),
                details=json.loads(row["details_json"]),
                error_code=row["error_code"],
            )
            for row in rows
        ]

    def corrupt_first_node_digest_for_injection(self) -> dict[str, Any]:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT cluster_id, revision FROM nodes ORDER BY cluster_id, revision LIMIT 1"
            ).fetchone()
            if row is None:
                raise DiagError(E_LEDGER, "cannot inject ledger corruption: ledger has no nodes")
            original = connection.execute(
                "SELECT content_digest FROM nodes WHERE cluster_id=? AND revision=?",
                (row["cluster_id"], row["revision"]),
            ).fetchone()["content_digest"]
            corrupted = "sha256:" + ("0" * 64)
            connection.execute(
                "UPDATE nodes SET content_digest=? WHERE cluster_id=? AND revision=?",
                (corrupted, row["cluster_id"], row["revision"]),
            )
            return {
                "cluster_id": row["cluster_id"],
                "revision": row["revision"],
                "field": "content_digest",
                "original": original,
                "corrupted": corrupted,
            }
        finally:
            connection.close()
