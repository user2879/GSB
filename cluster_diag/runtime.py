from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any, Callable

from .authority import AuthoritativeVerifier, ReplayVerifier
from .errors import DiagError, E_LATENCY, E_LEDGER, E_VIEW, StructuredLogger
from .latency import CLOCK_BASIS, TRUTH_SOURCE, measure_elapsed_us
from .ledger import Ledger, NodeCandidate, TreeSnapshot
from .snapshot import SessionOpenResult, build_session_document, open_session, save_session
from .spec import RevisionSpec, SystemSpec
from .transport import ServiceManager
from .util import canonical_json
from .view import IncrementalViewModel


ProgressCallback = Callable[[str, str, int, dict[str, Any]], None]


class DiagnosticRuntime:
    def __init__(
        self,
        *,
        spec: SystemSpec,
        workdir: str | Path,
        latency_budget_ms: int = 50,
        session_id: str | None = None,
        inject_port_conflict: bool = False,
        inject_ledger_corruption: bool = False,
        defer_semantic_ledger_checks: bool = False,
        logger: StructuredLogger | None = None,
    ) -> None:
        if latency_budget_ms < 0:
            raise DiagError(E_VIEW, "latency budget must be non-negative", field_path="latency_budget_ms")
        self.spec = spec
        self.workdir = Path(workdir).resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.logs_dir = self.workdir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or StructuredLogger(self.logs_dir / "events.jsonl")
        self.session_id = session_id or uuid.uuid4().hex
        self.latency_budget_ms = int(latency_budget_ms)
        self.inject_port_conflict = inject_port_conflict
        self.inject_ledger_corruption = inject_ledger_corruption
        self.defer_semantic_ledger_checks = defer_semantic_ledger_checks
        self.ledger_path = self.workdir / "results.sqlite3"
        self.snapshot_path = self.workdir / "session_snapshot.json"
        self.ledger = Ledger(self.ledger_path)
        self.services: ServiceManager | None = None
        self.view = IncrementalViewModel()
        self.current_snapshot: TreeSnapshot
        self.session_open: SessionOpenResult | None = None
        self.session_document: dict[str, Any] | None = None
        self.unknown_uncommitted_nodes: list[dict[str, Any]] = []
        self._entered = False

    def _compute_unknown_nodes(self, root: int | None) -> list[dict[str, Any]]:
        unknown: list[dict[str, Any]] = []
        for revision in self.spec.revisions:
            if root is not None and revision.version <= root:
                continue
            for cluster_id in self.spec.targeted_clusters(revision):
                unknown.append(
                    {
                        "cluster_id": cluster_id,
                        "revision": revision.version,
                        "task_id": self.spec.task_for_cluster(revision, cluster_id).id,
                        "status": "unknown",
                        "reason": "revision node is not committed at the observed root",
                        "required_action": "rerun",
                    }
                )
        return unknown

    def _ledger_latency_stats(self) -> dict[str, Any]:
        root = self.current_snapshot.root_revision
        events = self.ledger.latency_events(bound_root_revision=root)
        return self.view.current_latency_stats(events)

    def _save_session(self) -> None:
        snapshot = self.ledger.load_current_snapshot(verify_parents=True, verify_snapshot_digest=True)
        stats = self._ledger_latency_stats()
        document = build_session_document(snapshot, stats)
        save_session(self.snapshot_path, document)
        self.current_snapshot = snapshot
        self.session_document = document

    def _open_and_compare_session(self) -> None:
        stats = self._ledger_latency_stats()
        opened = open_session(
            self.snapshot_path,
            snapshot=self.current_snapshot,
            current_latency_stats=stats,
        )
        self.session_open = opened
        self.session_document = opened.active
        if opened.replaced:
            self.view.initialize(self.current_snapshot)
            save_session(self.snapshot_path, opened.active)
            self.logger.info(
                "SNAPSHOT_REPLACED",
                "session snapshot did not match the ledger and was replaced as a whole",
                details={"reasons": list(opened.reasons)},
            )

    def __enter__(self) -> "DiagnosticRuntime":
        try:
            if self.inject_ledger_corruption:
                details = self.ledger.corrupt_first_node_digest_for_injection()
                self.logger.info(
                    "INJECT_LEDGER_CORRUPTION",
                    "corrupted one persisted node content digest",
                    details=details,
                )
            self.current_snapshot = self.ledger.load_current_snapshot(
                verify_parents=not self.defer_semantic_ledger_checks,
                verify_snapshot_digest=not self.defer_semantic_ledger_checks,
                verify_root_pointer=not self.defer_semantic_ledger_checks,
            )
            self.view.initialize(self.current_snapshot)
            self.unknown_uncommitted_nodes = self._compute_unknown_nodes(self.current_snapshot.root_revision)
            if not self.defer_semantic_ledger_checks:
                self._open_and_compare_session()
            self.services = ServiceManager(
                self.spec,
                self.workdir,
                self.logger,
                inject_port_conflict=self.inject_port_conflict,
            )
            self.services.start()
            self._entered = True
            self.logger.info(
                "SESSION_READY",
                "diagnostic runtime is ready",
                details={
                    "root_revision": self.current_snapshot.root_revision,
                    "session_id": self.session_id,
                    "snapshot_replaced": None if self.session_open is None else self.session_open.replaced,
                    "snapshot_reasons": [] if self.session_open is None else list(self.session_open.reasons),
                    "unknown_node_count": len(self.unknown_uncommitted_nodes),
                },
            )
            return self
        except Exception:
            if self.services is not None:
                self.services.close()
            raise

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self.services is not None:
            self.services.close()
        self._entered = False

    def _revision_after(self, root: int | None) -> RevisionSpec | None:
        for revision in self.spec.revisions:
            if root is None or revision.version > root:
                return revision
        return None

    def commit_next(self, progress: ProgressCallback | None = None) -> dict[str, Any]:
        self._require_entered()
        root = self.ledger.root_revision()
        revision = self._revision_after(root)
        if revision is None:
            raise DiagError(
                E_VIEW,
                "there is no uncommitted revision left to apply",
                field_path="$.revisions",
                details={"root_revision": root},
            )
        if root is not None and revision.parent != root:
            raise DiagError(
                E_LEDGER,
                "next specification revision does not extend the ledger root",
                field_path=f"$.revisions[version={revision.version}].parent",
                details={"root_revision": root, "revision_parent": revision.parent},
            )
        if root is not None and root not in self.spec.revisions_by_version:
            raise DiagError(
                E_LEDGER,
                "ledger root is absent from the specification revision chain",
                field_path="$.revisions",
                details={"root_revision": root},
            )

        affected_clusters = self.spec.targeted_clusters(revision)
        if progress is not None:
            progress("revision_started", "", revision.version, {"clusters": list(affected_clusters)})
        candidates: list[NodeCandidate] = []
        node_report: list[dict[str, Any]] = []
        assert self.services is not None
        for cluster_id in affected_clusters:
            task = self.spec.task_for_cluster(revision, cluster_id)
            if progress is not None:
                progress("cluster_started", cluster_id, revision.version, {"task_id": task.id})
            response = self.services.client(cluster_id).distribute(
                revision.version,
                task.id,
                request_id=f"commit-r{revision.version}-{cluster_id}",
            )
            summary = response["aggregate_summary"]
            summary_canonical = response["aggregate_summary_canonical"]
            if canonical_json(summary) != summary_canonical:
                raise DiagError(
                    E_VIEW,
                    "simulator aggregate summary was not canonical before ledger commit",
                    details={"cluster_id": cluster_id, "revision": revision.version},
                )
            candidates.append(
                NodeCandidate(
                    cluster_id=cluster_id,
                    task_id=task.id,
                    aggregate_summary_canonical=summary_canonical,
                )
            )
            node_report.append(
                {
                    "cluster_id": cluster_id,
                    "revision": revision.version,
                    "task_id": task.id,
                    "aggregate_summary": summary,
                    "aggregate_summary_canonical": summary_canonical,
                    "compute_ms": response.get("compute_ms"),
                }
            )
            if progress is not None:
                progress("cluster_completed", cluster_id, revision.version, {"task_id": task.id})

        committed = self.ledger.commit_revision(
            revision=revision.version,
            parent_revision=root,
            candidates=candidates,
            required_clusters=set(self.spec.cluster_ids),
        )
        switch_report = self.view.apply_whole_root_replacement(
            committed,
            affected_clusters=affected_clusters,
        )
        self.current_snapshot = committed
        self.unknown_uncommitted_nodes = self._compute_unknown_nodes(committed.root_revision)
        self._save_session()
        if progress is not None:
            progress(
                "revision_completed",
                "",
                revision.version,
                {"root_switch": switch_report.as_dict()},
            )
        self.logger.info(
            "ROOT_REPLACED",
            "version-tree root replaced as a whole",
            details={
                "revision": revision.version,
                "affected_clusters": list(affected_clusters),
                "changed_clusters": list(switch_report.changed_cluster_ids),
                "stale_node_count": len(switch_report.stale_nodes),
            },
        )
        return {
            "ok": True,
            "committed_revision": revision.version,
            "parent_revision": root,
            "nodes": node_report,
            "root_switch": switch_report.as_dict(),
            "root_revision": committed.root_revision,
            "snapshot_digest": committed.snapshot_digest,
            "unknown_uncommitted_nodes": self.unknown_uncommitted_nodes,
        }

    def commit_to(self, target: str | int, progress: ProgressCallback | None = None) -> dict[str, Any]:
        self._require_entered()
        if target == "latest":
            target_version = self.spec.latest_state.version
        else:
            try:
                target_version = int(target)
            except (TypeError, ValueError) as exc:
                raise DiagError(
                    E_VIEW,
                    f"invalid revision target {target!r}",
                    field_path="$.revisions",
                ) from exc
            if target_version not in self.spec.revisions_by_version:
                raise DiagError(
                    E_VIEW,
                    f"target revision {target_version} is absent from the specification",
                    field_path="$.revisions",
                )
        current = self.ledger.root_revision()
        if current is not None and target_version < current:
            raise DiagError(
                E_VIEW,
                "cannot move the root backwards through the commit command",
                details={"root_revision": current, "target_revision": target_version},
            )
        commits: list[dict[str, Any]] = []
        while True:
            root = self.ledger.root_revision()
            if root is not None and root >= target_version:
                break
            commits.append(self.commit_next(progress=progress))
            if len(commits) > len(self.spec.revisions):
                raise DiagError(E_LEDGER, "revision commit loop made no progress")
        return {
            "ok": True,
            "target_revision": target_version,
            "commits": commits,
            "root_revision": self.ledger.root_revision(),
            "root_switch": None if not commits else commits[-1]["root_switch"],
        }

    def query(self, cluster_id: str, revision: int | None = None) -> dict[str, Any]:
        self._require_entered()
        started_ns = time.monotonic_ns()
        root_before = self.ledger.root_revision()
        error: DiagError | None = None
        node = None
        stale = False
        try:
            snapshot = self.ledger.load_current_snapshot(
                verify_parents=True,
                verify_snapshot_digest=True,
                verify_root_pointer=True,
            )
            if cluster_id not in self.spec.cluster_ids:
                raise DiagError(
                    E_VIEW,
                    f"unknown cluster {cluster_id!r}",
                    field_path="$.clusters",
                )
            effective = snapshot.effective_nodes.get(cluster_id)
            if effective is None:
                raise DiagError(
                    E_VIEW,
                    "cluster has no effective node in the current root",
                    details={"cluster_id": cluster_id, "root_revision": snapshot.root_revision},
                )
            node = effective if revision is None else self.ledger.get_node(cluster_id, revision)
            if node is None:
                raise DiagError(
                    E_VIEW,
                    "requested node is not present in the ledger",
                    details={"cluster_id": cluster_id, "revision": revision},
                )
            stale = node.revision != effective.revision or node.content_digest != effective.content_digest
        except DiagError as exc:
            error = exc
        ended_ns = time.monotonic_ns()
        elapsed_us = measure_elapsed_us(started_ns, ended_ns)
        budget_us = self.latency_budget_ms * 1000
        exceeded = elapsed_us > budget_us
        try:
            event = self.ledger.record_latency_event(
                session_id=self.session_id,
                kind="ui_query",
                cluster_id=cluster_id,
                revision=None if node is None else node.revision,
                bound_root_revision=root_before,
                started_ns=started_ns,
                ended_ns=ended_ns,
                elapsed_us=elapsed_us,
                budget_ms_x1000=budget_us,
                clock_basis=CLOCK_BASIS,
                truth_source=TRUTH_SOURCE,
                details={"stale": stale, "requested_revision": revision},
                error_code=(E_LATENCY if exceeded and error is None else None if error is None else error.code),
            )
        except Exception:
            event = None
        if exceeded and error is None:
            self.logger.emit(
                E_LATENCY,
                "UI query exceeded the explicit latency budget",
                severity="warning",
                details={
                    "cluster_id": cluster_id,
                    "bound_root_revision": root_before,
                    "elapsed_ms_x1000": elapsed_us,
                    "budget_ms_x1000": budget_us,
                    "overage_ms_x1000": elapsed_us - budget_us,
                },
            )
        try:
            self._save_session()
        except Exception:
            # Preserve the query failure; a later recovery path will report ledger errors.
            pass
        if error is not None:
            self.logger.error(error)
            raise error
        assert node is not None
        stats = self._ledger_latency_stats()
        return {
            "ok": True,
            "cluster_id": cluster_id,
            "revision": node.revision,
            "root_revision": root_before,
            "node": node.as_dict(),
            "stale": stale,
            "excluded_from_current_statistics": stale,
            "latency": {
                "started_ns": started_ns,
                "ended_ns": ended_ns,
                "elapsed_ms": elapsed_us // 1000,
                "elapsed_ms_x1000": elapsed_us,
                "budget_ms_x1000": budget_us,
                "exceeded": exceeded,
                "overage_ms_x1000": max(0, elapsed_us - budget_us),
                "clock_basis": CLOCK_BASIS,
                "truth_source": TRUTH_SOURCE,
                "event_id": None if event is None else event.id,
                "error_code": E_LATENCY if exceeded else None,
            },
            "latency_stats": stats,
        }

    def latency_stats(self) -> dict[str, Any]:
        return self._ledger_latency_stats()

    def authoritative(self) -> dict[str, Any]:
        self._require_entered()
        assert self.services is not None
        verifier = AuthoritativeVerifier(
            spec=self.spec,
            ledger=self.ledger,
            services=self.services,
            logger=self.logger,
            session_id=self.session_id,
        )
        result = verifier.run()
        self._save_session()
        result["latency_stats"] = self._ledger_latency_stats()
        root = self.current_snapshot.root_revision
        authority_events = self.ledger.latency_events(
            bound_root_revision=root,
            kind="authoritative",
        )
        if authority_events:
            event = authority_events[-1]
            result["latency"] = {
                "started_ns": event.started_ns,
                "ended_ns": event.ended_ns,
                "elapsed_ms": event.elapsed_ms_x1000 // 1000,
                "elapsed_ms_x1000": event.elapsed_ms_x1000,
                "clock_basis": event.clock_basis,
                "truth_source": event.truth_source,
                "event_id": event.id,
            }
        result["root_switch_generation"] = self.view.switch_generation
        return result

    def recover(self, *, replay: bool = True) -> dict[str, Any]:
        self._require_entered()
        strict_snapshot = self.ledger.load_current_snapshot(
            verify_parents=True,
            verify_snapshot_digest=True,
            verify_root_pointer=True,
        )
        self.current_snapshot = strict_snapshot
        self.unknown_uncommitted_nodes = self._compute_unknown_nodes(strict_snapshot.root_revision)
        opened = open_session(
            self.snapshot_path,
            snapshot=strict_snapshot,
            current_latency_stats=self._ledger_latency_stats(),
        )
        self.session_open = opened
        if opened.replaced:
            self.view.initialize(strict_snapshot)
            self.session_document = opened.active
            save_session(self.snapshot_path, opened.active)
            self.logger.info(
                "RECOVERY_SESSION_REPLACED",
                "snapshot mismatch caused a whole session-state replacement",
                details={"reasons": list(opened.reasons)},
            )
        else:
            self.session_document = opened.active

        replay_result = None
        if replay:
            assert self.services is not None
            verifier = ReplayVerifier(
                spec=self.spec,
                ledger=self.ledger,
                services=self.services,
                logger=self.logger,
                snapshot_path=self.snapshot_path,
            )
            replay_result = verifier.run(
                self.session_document,
                known_unknown=self.unknown_uncommitted_nodes,
            )
            self._save_session()
        return {
            "ok": True,
            "snapshot_replaced": opened.replaced,
            "snapshot_reasons": list(opened.reasons),
            "root_revision": strict_snapshot.root_revision,
            "unknown_uncommitted_nodes": self.unknown_uncommitted_nodes,
            "replay": replay_result,
        }

    def _require_entered(self) -> None:
        if not self._entered:
            raise DiagError(E_VIEW, "diagnostic runtime is not open")

    def current_items(self) -> list[dict[str, Any]]:
        return [
            {
                **item.as_dict(),
                "stale": self.view.is_stale(item),
            }
            for item in self.view.items
        ]
