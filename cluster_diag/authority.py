from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterable

from .errors import DiagError, E_PORT, E_REPLAY, E_VIEW, StructuredLogger
from .latency import CLOCK_BASIS, TRUTH_SOURCE, aggregate_latency_events, measure_elapsed_us
from .ledger import Ledger, LedgerNode, TreeSnapshot
from .snapshot import SESSION_SCHEMA
from .spec import SystemSpec
from .transport import ServiceManager
from .util import canonical_json, node_content_digest, sha256_text


def first_json_difference(expected: Any, actual: Any, path: str = "$") -> dict[str, Any] | None:
    if type(expected) is not type(actual):
        return {"field_path": path, "expected": expected, "actual": actual}
    if isinstance(expected, dict):
        for key in sorted(set(expected) | set(actual)):
            child = f"{path}.{key}"
            if key not in expected:
                return {"field_path": child, "expected": "<missing>", "actual": actual[key]}
            if key not in actual:
                return {"field_path": child, "expected": expected[key], "actual": "<missing>"}
            difference = first_json_difference(expected[key], actual[key], child)
            if difference is not None:
                return difference
        return None
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return {"field_path": path, "expected_length": len(expected), "actual_length": len(actual)}
        for index, (left, right) in enumerate(zip(expected, actual)):
            difference = first_json_difference(left, right, f"{path}[{index}]")
            if difference is not None:
                return difference
        return None
    if expected != actual:
        return {"field_path": path, "expected": expected, "actual": actual}
    return None


def _has_first_difference(expected: Any, actual: Any, path: str = "$") -> bool:
    return first_json_difference(expected, actual, path) is not None


def _view_error(
    message: str,
    *,
    cluster_id: str | None,
    revision: int | None,
    field: str,
    details: dict[str, Any] | None = None,
) -> DiagError:
    merged = dict(details or {})
    merged["cluster_id"] = cluster_id
    merged["revision"] = revision
    merged["field"] = field
    return DiagError(
        E_VIEW,
        message,
        field_path=f"nodes[{cluster_id}].{field}" if cluster_id is not None else f"root.{field}",
        details=merged,
    )


class AuthoritativeVerifier:
    def __init__(
        self,
        *,
        spec: SystemSpec,
        ledger: Ledger,
        services: ServiceManager,
        logger: StructuredLogger,
        session_id: str,
    ) -> None:
        self.spec = spec
        self.ledger = ledger
        self.services = services
        self.logger = logger
        self.session_id = session_id

    def _recompute_node(self, node: LedgerNode) -> tuple[dict[str, Any], str]:
        try:
            response = self.services.client(node.cluster_id).distribute(
                node.revision,
                node.task_id,
                request_id=f"authority-r{node.revision}-{node.cluster_id}",
            )
        except DiagError as exc:
            if exc.code == E_PORT:
                raise
            raise _view_error(
                "authoritative simulator recomputation failed",
                cluster_id=node.cluster_id,
                revision=node.revision,
                field="service_recompute",
                details={"underlying_code": exc.code, "underlying_message": exc.message},
            ) from exc
        summary = response.get("aggregate_summary")
        summary_canonical = response.get("aggregate_summary_canonical")
        if not isinstance(summary, dict) or not isinstance(summary_canonical, str):
            raise _view_error(
                "simulator response omitted the aggregate summary",
                cluster_id=node.cluster_id,
                revision=node.revision,
                field="aggregate_summary",
            )
        if canonical_json(summary) != summary_canonical:
            raise _view_error(
                "simulator aggregate summary is not canonical",
                cluster_id=node.cluster_id,
                revision=node.revision,
                field="aggregate_summary_canonical",
            )
        return summary, summary_canonical

    def _semantic_checks(self, snapshot: TreeSnapshot) -> tuple[list[LedgerNode], dict[int, Any]]:
        root = snapshot.root_revision
        if root is None:
            raise _view_error(
                "authoritative verification requires a committed root",
                cluster_id=None,
                revision=None,
                field="root_revision",
            )
        commits = self.ledger.list_commits()
        commit_by_revision = {commit.revision: commit for commit in commits}
        if commits and max(commit.revision for commit in commits) != root:
            raise _view_error(
                "root pointer is behind the latest committed revision",
                cluster_id=None,
                revision=root,
                field="root_revision",
                details={"latest_commit": max(commit.revision for commit in commits)},
            )
        visible_commits = [commit for commit in commits if commit.revision <= root]
        if not visible_commits:
            raise _view_error(
                "root revision has no commit record",
                cluster_id=None,
                revision=root,
                field="commit",
            )
        if visible_commits[-1].revision != root:
            raise _view_error(
                "root pointer does not point at the latest committed revision",
                cluster_id=None,
                revision=root,
                field="root_revision",
                details={"latest_commit": visible_commits[-1].revision},
            )
        for index, commit in enumerate(visible_commits):
            expected_parent = None if index == 0 else visible_commits[index - 1].revision
            if commit.parent_revision != expected_parent:
                raise _view_error(
                    "commit parent reference is inconsistent",
                    cluster_id="__root__",
                    revision=commit.revision,
                    field="parent_revision",
                    details={"expected": expected_parent, "actual": commit.parent_revision},
                )

        if root not in self.spec.revisions_by_version:
            raise _view_error(
                "ledger root revision is absent from the specification",
                cluster_id=None,
                revision=root,
                field="revision",
            )

        for cluster_id in self.spec.cluster_ids:
            if cluster_id not in snapshot.effective_nodes:
                raise _view_error(
                    "reconstructed version tree omits a declared cluster",
                    cluster_id=cluster_id,
                    revision=root,
                    field="effective_node",
                )

        nodes = list(snapshot.all_nodes)
        for cluster_id, chain in sorted(snapshot.history.items()):
            for index, node in enumerate(chain):
                expected_parent = None if index == 0 else chain[index - 1].revision
                if node.parent_revision != expected_parent:
                    raise _view_error(
                        "node parent reference is inconsistent",
                        cluster_id=cluster_id,
                        revision=node.revision,
                        field="parent_revision",
                        details={"expected": expected_parent, "actual": node.parent_revision},
                    )

        for node in nodes:
            revision = self.spec.revisions_by_version.get(node.revision)
            if revision is None:
                raise _view_error(
                    "ledger node revision is absent from the specification",
                    cluster_id=node.cluster_id,
                    revision=node.revision,
                    field="revision",
                )
            try:
                task = self.spec.task_for_cluster(revision, node.cluster_id)
            except DiagError:
                raise _view_error(
                    "ledger node task is not assigned to its cluster in that revision",
                    cluster_id=node.cluster_id,
                    revision=node.revision,
                    field="task_id",
                )
            if task.id != node.task_id:
                raise _view_error(
                    "ledger node task identifier does not match the specification",
                    cluster_id=node.cluster_id,
                    revision=node.revision,
                    field="task_id",
                    details={"expected": task.id, "actual": node.task_id},
                )

        stored_commit = commit_by_revision[root]
        if snapshot.snapshot_digest != stored_commit.snapshot_digest:
            raise _view_error(
                "root snapshot digest does not match the reconstructed tree",
                cluster_id="__root__",
                revision=root,
                field="snapshot_digest",
                details={"expected": stored_commit.snapshot_digest, "actual": snapshot.snapshot_digest},
            )
        return nodes, commit_by_revision

    def _run(self) -> dict[str, Any]:
        snapshot = self.ledger.load_current_snapshot(
            verify_parents=False,
            verify_snapshot_digest=False,
            verify_root_pointer=False,
        )
        nodes, _commits = self._semantic_checks(snapshot)
        recomputed: list[dict[str, Any]] = []
        for node in nodes:
            summary, summary_canonical = self._recompute_node(node)
            if summary_canonical != node.aggregate_summary_canonical:
                expected_summary = node.aggregate_summary
                difference = first_json_difference(expected_summary, summary) or {
                    "field_path": "$",
                    "expected": "<different>",
                    "actual": "<different>",
                }
                raise _view_error(
                    "authoritative aggregate summary differs from the ledger node",
                    cluster_id=node.cluster_id,
                    revision=node.revision,
                    field="aggregate_summary",
                    details={
                        "first_difference": difference,
                        "ledger_summary_digest": node.content_digest,
                        "recomputed_summary_digest": node_content_digest(
                            task_id=node.task_id,
                            cluster_id=node.cluster_id,
                            revision=node.revision,
                            parent_revision=node.parent_revision,
                            aggregate_summary_canonical=summary_canonical,
                        ),
                    },
                )
            recomputed_content_digest = node_content_digest(
                task_id=node.task_id,
                cluster_id=node.cluster_id,
                revision=node.revision,
                parent_revision=node.parent_revision,
                aggregate_summary_canonical=summary_canonical,
            )
            if recomputed_content_digest != node.content_digest:
                raise _view_error(
                    "authoritative content digest differs from the ledger node",
                    cluster_id=node.cluster_id,
                    revision=node.revision,
                    field="content_digest",
                    details={"expected": node.content_digest, "actual": recomputed_content_digest},
                )
            recomputed.append(
                {
                    "cluster_id": node.cluster_id,
                    "revision": node.revision,
                    "task_id": node.task_id,
                    "parent_revision": node.parent_revision,
                    "content_digest": recomputed_content_digest,
                    "aggregate_summary_digest": sha256_text(summary_canonical),
                }
            )
        return {
            "ok": True,
            "root_revision": snapshot.root_revision,
            "reconstructed_node_count": len(nodes),
            "recomputed_node_count": len(recomputed),
            "clusters": sorted(snapshot.effective_nodes),
            "root_pointer_consistent": True,
            "parent_references_consistent": True,
            "nodes": recomputed,
        }

    def run(self) -> dict[str, Any]:
        started_ns = time.monotonic_ns()
        root = self.ledger.root_revision()
        error: DiagError | None = None
        try:
            result = self._run()
            return result
        except DiagError as exc:
            error = exc
            self.logger.error(exc)
            raise
        finally:
            ended_ns = time.monotonic_ns()
            elapsed_us = measure_elapsed_us(started_ns, ended_ns)
            try:
                event = self.ledger.record_latency_event(
                    session_id=self.session_id,
                    kind="authoritative",
                    cluster_id=None,
                    revision=root,
                    bound_root_revision=root,
                    started_ns=started_ns,
                    ended_ns=ended_ns,
                    elapsed_us=elapsed_us,
                    budget_ms_x1000=None,
                    clock_basis=CLOCK_BASIS,
                    truth_source=TRUTH_SOURCE,
                    details={"error_code": None if error is None else error.code},
                    error_code=None if error is None else error.code,
                )
                self.logger.info(
                    "AUTHORITATIVE_RECOMPUTE",
                    "authoritative recomputation completed",
                    details={
                        "root_revision": root,
                        "elapsed_ms_x1000": event.elapsed_ms_x1000,
                        "error_code": None if error is None else error.code,
                    },
                )
            except Exception:
                # Preserve the primary verification failure if latency evidence cannot be written.
                pass


class ReplayVerifier:
    def __init__(
        self,
        *,
        spec: SystemSpec,
        ledger: Ledger,
        services: ServiceManager,
        logger: StructuredLogger,
        snapshot_path: str | Path,
    ) -> None:
        self.spec = spec
        self.ledger = ledger
        self.services = services
        self.logger = logger
        self.snapshot_path = Path(snapshot_path)

    @staticmethod
    def _replay_error(message: str, *, basis: str, **details: Any) -> DiagError:
        merged = dict(details)
        merged["basis"] = basis
        return DiagError(E_REPLAY, message, details=merged)

    def run(self, expected: dict[str, Any], known_unknown: Iterable[dict[str, Any]]) -> dict[str, Any]:
        try:
            snapshot = self.ledger.load_current_snapshot(verify_parents=True, verify_snapshot_digest=True)
        except DiagError as exc:
            replay_error = self._replay_error(
                "recovered ledger tree cannot be reconstructed",
                basis="ledger_tree",
                underlying_code=exc.code,
                underlying_details=exc.details,
            )
            self.logger.error(replay_error)
            raise replay_error from exc

        if expected.get("schema") != SESSION_SCHEMA:
            raise self._replay_error("session snapshot schema is not reproducible", basis="snapshot_schema")
        if expected.get("root_revision") != snapshot.root_revision:
            raise self._replay_error(
                "snapshot root revision differs from the reconstructed root",
                basis="root_revision",
                expected=expected.get("root_revision"),
                actual=snapshot.root_revision,
            )
        expected_nodes = expected.get("node_digests")
        actual_nodes = snapshot.node_digest_map()
        if expected_nodes != actual_nodes:
            missing = sorted(set(actual_nodes) - set(expected_nodes or {}))
            extra = sorted(set(expected_nodes or {}) - set(actual_nodes))
            changed = sorted(
                key
                for key in set(actual_nodes) & set(expected_nodes or {})
                if actual_nodes[key] != expected_nodes[key]
            )
            raise self._replay_error(
                "snapshot node digest set is not reproducible from the ledger",
                basis="node_digests",
                missing=missing,
                extra=extra,
                changed=changed,
            )
        if expected.get("root_snapshot_digest") != snapshot.snapshot_digest:
            raise self._replay_error(
                "snapshot root digest is not reproducible",
                basis="root_snapshot_digest",
                expected=expected.get("root_snapshot_digest"),
                actual=snapshot.snapshot_digest,
            )

        expected_parent_by_node: dict[str, int | None] = {}
        for cluster_id, chain in snapshot.history.items():
            for index, node in enumerate(chain):
                expected_parent_by_node[node.key] = None if index == 0 else chain[index - 1].revision

        by_key = {node.key: node for node in snapshot.all_nodes}
        replayed_nodes: list[dict[str, Any]] = []
        for key in sorted(by_key):
            node = by_key[key]
            try:
                response = self.services.client(node.cluster_id).distribute(
                    node.revision,
                    node.task_id,
                    request_id=f"replay-r{node.revision}-{node.cluster_id}",
                )
            except DiagError as exc:
                raise self._replay_error(
                    "node cannot be replayed on the local simulator",
                    basis="service_recompute",
                    cluster_id=node.cluster_id,
                    revision=node.revision,
                    underlying_code=exc.code,
                    underlying_message=exc.message,
                ) from exc
            if response.get("aggregate_summary_canonical") != node.aggregate_summary_canonical:
                raise self._replay_error(
                    "node aggregate summary is not reproducible",
                    basis="aggregate_summary",
                    cluster_id=node.cluster_id,
                    revision=node.revision,
                    expected_digest=node.content_digest,
                    actual=response.get("aggregate_summary_digest"),
                )
            recomputed_digest = node_content_digest(
                task_id=node.task_id,
                cluster_id=node.cluster_id,
                revision=node.revision,
                parent_revision=node.parent_revision,
                aggregate_summary_canonical=node.aggregate_summary_canonical,
            )
            if recomputed_digest != node.content_digest:
                raise self._replay_error(
                    "node content digest is not reproducible",
                    basis="content_digest",
                    cluster_id=node.cluster_id,
                    revision=node.revision,
                    expected=node.content_digest,
                    actual=recomputed_digest,
                )
            if node.parent_revision != expected_parent_by_node[key]:
                raise self._replay_error(
                    "node parent reference is not reproducible",
                    basis="parent_revision",
                    cluster_id=node.cluster_id,
                    revision=node.revision,
                    expected=expected_parent_by_node[key],
                    actual=node.parent_revision,
                )
            replayed_nodes.append(
                {
                    "cluster_id": node.cluster_id,
                    "revision": node.revision,
                    "content_digest": node.content_digest,
                }
            )

        root = snapshot.root_revision
        events = self.ledger.latency_events(bound_root_revision=root)
        recomputed_stats = aggregate_latency_events(events)
        expected_stats = expected.get("latency_stats")
        if expected.get("latency_schema_version") != recomputed_stats["schema"]:
            raise self._replay_error(
                "latency statistics schema version is not reproducible",
                basis="latency_schema_version",
                expected=expected.get("latency_schema_version"),
                actual=recomputed_stats["schema"],
            )
        if _has_first_difference(expected_stats, recomputed_stats, "$.latency_stats"):
            difference = first_json_difference(expected_stats, recomputed_stats, "$.latency_stats")
            raise self._replay_error(
                "latency statistics are not reproducible from ledger query records",
                basis="latency_stats",
                first_difference=difference,
            )

        result = {
            "ok": True,
            "root_revision": root,
            "replayed_node_count": len(replayed_nodes),
            "latency_stats": recomputed_stats,
            "unknown_uncommitted_nodes": list(known_unknown),
            "parent_references_consistent": True,
            "root_pointer_consistent": True,
        }
        self.logger.info(
            "REPLAY_OK",
            "snapshot recovery replay matched the ledger and simulator",
            details={"root_revision": root, "node_count": len(replayed_nodes)},
        )
        return result
