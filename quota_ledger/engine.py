"""Top-level orchestration: validation consumer, event barriers, recovery, reports."""
from __future__ import annotations

import os
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .client import NodeClient, NodePool
from .errors import BusinessError, StructuredLogger
from .invalidation import propagate_invalidation
from .ledger import Ledger
from .scheduler import Scheduler
from .spec import EventSpec, Spec, direct_invalidation_reasons
from .util import atomic_write_json, canonical_json, monotonic_ms


@dataclass
class EngineResult:
    success: bool
    interrupted: bool
    exit_code: int
    report: dict[str, Any]
    ledger_digest: str
    workdir: str


class Engine:
    def __init__(
        self,
        spec: Spec,
        workdir: str | Path,
        *,
        crash_after_commits: int = 0,
        graceful_after_commits: int = 0,
        max_attempts: int = 2,
    ):
        self.spec = spec
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.logger = StructuredLogger(self.workdir / "structured.jsonl")
        self.ledger = Ledger(self.workdir / "ledger.sqlite3")
        self.pool: NodePool | None = None
        self.scheduler: Scheduler | None = None
        self.stop_event = threading.Event()
        self.stop_reason: str | None = None
        self.crash_after_commits = int(crash_after_commits)
        self.graceful_after_commits = int(graceful_after_commits)
        self.max_attempts = int(max_attempts)
        self._commit_count = 0
        self._commit_lock = threading.Lock()
        self.recovery_info: dict[str, Any] = {}
        self.active_revision = 1

    # --------------------------------------------------------------- controls
    def request_stop(self, reason: str) -> None:
        if self.stop_reason is None:
            self.stop_reason = reason
            self.stop_event.set()
            if self.scheduler is not None:
                self.scheduler.request_shutdown()
            self.logger.write("STOP_REQUESTED", f"graceful stop requested: {reason}")

    def _after_commit(self, info: dict[str, Any]) -> None:
        with self._commit_lock:
            self._commit_count += 1
            count = self._commit_count
        self.logger.write(
            "COMMIT",
            "shard transaction committed",
            shard_id=info["shard_id"],
            extra={
                "generation": info["generation"],
                "result_digest": info["result_digest"],
                "ledger_digest": info["ledger_digest"],
            },
        )
        if self.crash_after_commits and count >= self.crash_after_commits:
            self.logger.write(
                "CRASH_SIMULATED",
                "simulated forced termination after committed shard count",
                shard_id=info["shard_id"],
                extra={"commits": count, "exit_report_written": False},
            )
            os._exit(70)
        if self.graceful_after_commits and count >= self.graceful_after_commits:
            self.request_stop(f"graceful-after-{count}-commits")

    # ---------------------------------------------------------------- helpers
    def _initial_nodes(self) -> dict[str, dict[str, int]]:
        revision = self.spec.revisions[1]
        return {node_id: dict(values) for node_id, values in sorted(revision.node_changes.items())}

    def _ledger_node_limits(self) -> dict[str, dict[str, int]]:
        return {
            row["node_id"]: {
                "concurrency": int(row["concurrency"]),
                "lease_limit": int(row["lease_limit"]),
                "capacity_version": int(row["capacity_version"]),
            }
            for row in self.ledger.node_quotas()
        }

    def _wait_for_recovered_ports(self, timeout_s: float = 3.0) -> None:
        """Give orphaned child watchers a short real-time window to release ports."""
        deadline = time.monotonic() + timeout_s
        pending = {node.port for node in self.spec.nodes.values()}
        while pending and time.monotonic() < deadline:
            free: set[int] = set()
            for port in sorted(pending):
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                try:
                    if os.name == "nt" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                        sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
                    elif hasattr(socket, "SO_REUSEADDR"):
                        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    sock.bind(("127.0.0.1", port))
                    free.add(port)
                except OSError:
                    pass
                finally:
                    sock.close()
            pending -= free
            if pending:
                time.sleep(0.05)
        if pending:
            self.logger.write(
                "E_PORT",
                "recovery ports remain occupied after grace window",
                extra={"ports": sorted(pending)},
            )

    def _configure_nodes(self, nodes: Mapping[str, Mapping[str, int]]) -> None:
        assert self.pool is not None
        for node_id in sorted(nodes):
            values = nodes[node_id]
            self.pool.clients[node_id].configure(
                concurrency=int(values["concurrency"]),
                lease_limit=int(values["lease_limit"]),
                capacity_version=int(values["capacity_version"]),
            )

    def _pending_generations(self, shard_ids: list[str]) -> dict[str, int]:
        results = self.ledger.results()
        requests = self.ledger.dispatch_requests()
        generations: dict[str, int] = {}
        for shard_id in sorted(shard_ids):
            base = 0
            if shard_id in results:
                base = max(base, int(results[shard_id]["generation"]))
            if shard_id in requests:
                base = max(base, int(requests[shard_id]["generation"]))
            generations[shard_id] = base + 1 if base > 0 else 1
        return generations

    def _invalidation_checks(self, affected: set[str], before: Mapping[str, str]) -> list[dict[str, Any]]:
        checks: list[dict[str, Any]] = []
        after = self.ledger.result_digests()
        for shard_id in sorted(set(after) - affected):
            digest = after[shard_id]
            checks.append(
                {
                    "shard_id": shard_id,
                    "digest_before": before.get(shard_id, digest),
                    "digest_after": digest,
                    "unchanged": before.get(shard_id, digest) == digest,
                }
            )
        return checks

    def _apply_invalidation(
        self,
        event: EventSpec,
        *,
        revision_version: int,
        global_budget: int,
        nodes: Mapping[str, Mapping[str, int]],
        direct_reasons: Mapping[str, list[str]],
        event_type: str,
    ) -> None:
        before = self.ledger.result_digests()
        entries = propagate_invalidation(self.spec, direct_reasons)
        affected = {str(entry["shard_id"]) for entry in entries}
        generations = self._pending_generations(sorted(affected))
        checks = self._invalidation_checks(affected, before)
        self._configure_nodes(nodes)
        self.ledger.apply_revision_event(
            event.raw,
            revision_version=revision_version,
            global_budget=global_budget,
            nodes=nodes,
            invalidation_entries=entries,
            requested_generations=generations,
            checks=checks,
            event_type=event_type,
        )
        after_digests = self.ledger.result_digests()
        for check in checks:
            actual = after_digests.get(str(check["shard_id"]))
            if not check["unchanged"] or actual != check["digest_before"]:
                raise BusinessError(
                    "E_REPLAY",
                    "an unaffected shard changed during invalidation",
                    shard_id=str(check["shard_id"]),
                    field="result_digest",
                    detail={"before": check["digest_before"], "after": actual},
                )
        assert self.scheduler is not None
        self.scheduler.update_config(
            global_budget=global_budget,
            node_limits=nodes,
            revision_version=revision_version,
        )
        self.scheduler.sync_from_ledger()
        for entry in entries:
            self.logger.write(
                "INVALIDATE",
                "shard invalidated",
                shard_id=str(entry["shard_id"]),
                event_seq=event.seq,
                extra={
                    "direct": bool(entry["direct"]),
                    "reason": entry["reason"],
                    "topo_rank": entry["topo_rank"],
                },
            )
        self.active_revision = int(revision_version)

    def _run_pending(self) -> str:
        assert self.scheduler is not None
        self.scheduler.pump()
        outcome = self.scheduler.wait_epoch(stop_event=self.stop_event)
        self.scheduler.mark_unreachable()
        return outcome

    def _apply_revision(self, event: EventSpec) -> str:
        target_version = int(event.raw["version"])
        target = self.spec.revisions[target_version]
        old_version = self.active_revision
        direct = direct_invalidation_reasons(self.spec, old_version, target_version)
        nodes = {node_id: dict(values) for node_id, values in sorted(target.node_changes.items())}
        self._apply_invalidation(
            event,
            revision_version=target_version,
            global_budget=target.global_budget,
            nodes=nodes,
            direct_reasons=direct,
            event_type="revision",
        )
        return self._run_pending()

    def _quota_targets(self, event: EventSpec) -> tuple[int, dict[str, dict[str, int]], dict[str, list[str]]]:
        quotas = {
            row["node_id"]: {
                "concurrency": int(row["concurrency"]),
                "lease_limit": int(row["lease_limit"]),
                "capacity_version": int(row["capacity_version"]),
            }
            for row in self.ledger.node_quotas()
        }
        budget = int(self.ledger.budget_state()["global_budget"])
        changed_nodes: list[str] = []
        if "global_budget" in event.raw:
            new_budget = int(event.raw["global_budget"])
        else:
            new_budget = budget
        node_id = str(event.raw["node_id"])
        for field_name in ("concurrency", "lease_limit", "capacity_version"):
            if field_name in event.raw and int(event.raw[field_name]) != quotas[node_id][field_name]:
                quotas[node_id][field_name] = int(event.raw[field_name])
                if field_name != "capacity_version" and node_id not in changed_nodes:
                    changed_nodes.append(node_id)
        direct: dict[str, list[str]] = {}
        if new_budget != budget:
            for shard in self.spec.shards.values():
                direct.setdefault(shard.id, []).append(f"quota:{event.seq}:global_budget")
        for changed_node in changed_nodes:
            for shard in self.spec.shards.values():
                if shard.node_id == changed_node:
                    direct.setdefault(shard.id, []).append(f"quota:{event.seq}:node:{changed_node}")
        return new_budget, quotas, direct

    def _apply_quota(self, event: EventSpec) -> str:
        budget, nodes, direct = self._quota_targets(event)
        self._apply_invalidation(
            event,
            revision_version=self.active_revision,
            global_budget=budget,
            nodes=nodes,
            direct_reasons=direct,
            event_type="quota",
        )
        return self._run_pending()

    # ------------------------------------------------------------------- run
    def run(self) -> EngineResult:
        initial_nodes = self._initial_nodes()
        self.recovery_info = self.ledger.begin_run(
            self.spec.source_digest,
            initial_nodes,
            int(self.spec.revisions[1].global_budget),
        )
        for shard_id in self.recovery_info.get("unknown_shards", []):
            self.logger.write(
                "E_UNKNOWN",
                "uncommitted shard marked unknown and requires rerun",
                shard_id=shard_id,
                extra={"recovery": True},
            )
        self.pool = NodePool(self.spec.nodes, self.spec.default_lease_ttl_ms)
        try:
            if self.recovery_info.get("recovery"):
                self._wait_for_recovered_ports()
            self.pool.start_all()
            ledger_nodes = self._ledger_node_limits()
            self._configure_nodes(ledger_nodes)
            self.active_revision = self.ledger.active_revision()
            budget_state = self.ledger.budget_state()
            self.scheduler = Scheduler(
                self.spec,
                self.ledger,
                self.pool.clients,
                self.logger,
                global_budget=int(budget_state["global_budget"]),
                node_limits=ledger_nodes,
                revision_version=self.active_revision,
                max_attempts=self.max_attempts,
                after_commit=self._after_commit,
            )
            applied = set(self.ledger.applied_event_seqs())
            interrupted = False
            for event in self.spec.events:
                if event.seq in applied:
                    continue
                if event.type in {"dispatch", "cancel"}:
                    if event.type == "dispatch":
                        self.ledger.apply_dispatch_event(
                            event.raw,
                            revision_version=self.active_revision,
                            default_ttl_ms=self.spec.default_lease_ttl_ms,
                        )
                        self.logger.write(
                            "EVENT_APPLIED",
                            "dispatch request recorded",
                            shard_id=event.shard_id,
                            event_seq=event.seq,
                            extra={"revision_version": self.active_revision},
                        )
                    else:
                        result = self.ledger.apply_cancel_event(event.raw)
                        self.logger.write(
                            "CANCEL",
                            "shard cancelled; future retries disabled",
                            shard_id=event.shard_id,
                            event_seq=event.seq,
                            extra={
                                "reason": event.raw.get("reason", "cancelled"),
                                "released": result.get("released"),
                            },
                        )
                    self.scheduler.sync_from_ledger()
                    continue
                # Configuration barrier: deterministically drain the prior event batch.
                outcome = self._run_pending()
                if outcome == "shutdown":
                    interrupted = True
                    break
                if event.type == "revision":
                    outcome = self._apply_revision(event)
                elif event.type == "quota":
                    outcome = self._apply_quota(event)
                else:  # validated earlier; defensive
                    raise BusinessError("E_SPEC", f"unsupported event type: {event.type}", event_seq=event.seq)
                if outcome == "shutdown":
                    interrupted = True
                    break
            if not interrupted:
                outcome = self._run_pending()
                interrupted = outcome == "shutdown"
            status = self.scheduler.status_snapshot()
            ledger_digest = self.ledger.content_digest()
            results = self.ledger.results()
            unfinished = [
                task
                for task in status["tasks"]
                if task["status"] not in {"completed", "cancelled"}
            ]
            failed = [task for task in status["tasks"] if task["status"] in {"failed", "cancelled_dependency"}]
            complete = not unfinished and not failed and not interrupted
            report: dict[str, Any] = {
                "schema_version": 1,
                "reason": self.stop_reason or ("completed" if complete else "incomplete"),
                "clean_exit": True,
                "complete": complete,
                "interrupted": interrupted,
                "source_digest": self.spec.source_digest,
                "source_label": self.spec.source_label,
                "active_revision": self.active_revision,
                "ledger_format_version": self.ledger.state_snapshot()["ledger_format_version"],
                "ledger_digest": ledger_digest,
                "ledger_file": str(self.workdir / "ledger.sqlite3"),
                "completed_shards": sorted(results),
                "completed_count": len(results),
                "unfinished_shards": [
                    {"shard_id": task["shard_id"], "status": task["status"]} for task in unfinished
                ],
                "cancelled_shards": [
                    task["shard_id"] for task in status["tasks"] if task["status"] == "cancelled"
                ],
                "cancellations": [
                    row for _, row in sorted(self.ledger.cancelled_shards().items())
                ],
                "unknown_shards_on_recovery": list(self.recovery_info.get("unknown_shards", [])),
                "recovery": self.recovery_info,
                "scheduler": status,
                "invalidations": self.ledger.invalidations(),
                "incremental_checks": self.ledger.incremental_checks(),
                "leases": self.ledger.leases(),
                "node_quota": self.ledger.node_quotas(),
                "budget_state": self.ledger.budget_state(),
                "snapshots": self.ledger.snapshots(limit=20),
                "exit_report_available": True,
                "generated_monotonic_ms": monotonic_ms(),
            }
            report["scheduler"]["node_snapshots"] = self.pool.snapshot()
            atomic_write_json(self.workdir / "run_report.json", report)
            (self.workdir / "ledger_digest.txt").write_text(ledger_digest + "\n", encoding="utf-8")
            self.ledger.finish_run(report, clean=True)
            success = complete
            exit_code = 0 if success else (130 if interrupted else 3)
            return EngineResult(
                success=success,
                interrupted=interrupted,
                exit_code=exit_code,
                report=report,
                ledger_digest=ledger_digest,
                workdir=str(self.workdir),
            )
        except BusinessError as exc:
            self.logger.write_error(exc)
            try:
                self.ledger.finish_run(
                    {
                        "schema_version": 1,
                        "reason": exc.code,
                        "clean_exit": True,
                        "complete": False,
                        "error": exc.to_dict(),
                        "exit_report_available": True,
                    },
                    clean=True,
                )
            except Exception:
                pass
            raise
        finally:
            if self.scheduler is not None:
                self.scheduler.close()
            if self.pool is not None:
                self.pool.stop_all()
            self.ledger.close()
