from __future__ import annotations

import os
import signal
from typing import Any, Optional

from .cluster import NodeCluster
from .errors import MeshError
from .protocol import label_for
from .scheduler import BoundedScheduler
from .store import Ledger


class Engine:
    def __init__(
        self,
        spec: dict[str, Any],
        ledger: Ledger,
        cluster: NodeCluster,
        *,
        crash_before_commit_request: Optional[str] = None,
    ) -> None:
        self.spec = spec
        self.ledger = ledger
        self.cluster = cluster
        self.crash_before_commit_request = crash_before_commit_request

    def execute(self, requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
        scheduler = BoundedScheduler(
            int(self.spec["global_concurrency"]),
            int(self.spec["per_node_lease_limit"]),
        )
        try:
            stages = sorted({int(request.get("stage", 1)) for request in requests})
            summaries: list[dict[str, Any]] = []
            for stage in stages:
                stage_requests = [request for request in requests if int(request.get("stage", 1)) == stage]
                futures = []
                for request in stage_requests:
                    self.ledger.plan_request(request)
                    futures.append(scheduler.submit(request, self._execute_one))
                outcomes = [future.result() for future in futures]
                stage_summaries = [self._summarize(outcome, scheduler) for outcome in outcomes]
                summaries.extend(stage_summaries)
                if any(not item.get("ok") for item in stage_summaries):
                    break
            return summaries
        finally:
            scheduler.shutdown(wait=True)

    def _execute_one(self, job: dict[str, Any]) -> dict[str, Any]:
        request = {key: value for key, value in job.items() if key not in {"wait_started_ms", "wait_ended_ms", "wait_ms", "active_global_at_start", "active_node_at_start", "queue_epoch"}}
        wire = {
            "request_id": request["request_id"],
            "op": request["op"],
            "node_id": request["node_id"],
            "lease_id": request.get("lease_id"),
            "slot_id": request.get("slot_id"),
        }
        if request["op"] in {"renew", "release"}:
            generation = request.get("generation")
            label = request.get("label")
            if label is None and generation is not None:
                label = label_for(request["lease_id"], int(generation))
            wire["generation"] = generation
            wire["label"] = label
        if request.get("quota_version") is not None:
            wire["quota_version"] = request["quota_version"]

        wait = {
            "wait_started_ms": job.get("wait_started_ms"),
            "wait_ended_ms": job.get("wait_ended_ms"),
            "wait_ms": job.get("wait_ms"),
        }
        try:
            response = self.cluster.request(request["node_id"], wire)
            if self.crash_before_commit_request == request["request_id"]:
                try:
                    self.ledger.conn.execute("PRAGMA wal_checkpoint(FULL)")
                except Exception:
                    pass
                os.kill(os.getpid(), signal.SIGKILL)
            committed = self.ledger.commit_live_result(request, response, wait=wait)
            response["_ledger"] = {"committed": True, "event": committed["event"]}
            response["_wait"] = wait
            return response
        except Exception as exc:
            error = exc if isinstance(exc, MeshError) else MeshError(str(exc), code="E_IO")
            try:
                self.ledger.record_rejection(request, error.to_dict(), wait=wait)
            except Exception:
                pass
            raise

    def _summarize(self, outcome: dict[str, Any], scheduler: BoundedScheduler) -> dict[str, Any]:
        request = outcome["request"]
        summary: dict[str, Any] = {
            "request_id": request["request_id"],
            "node_id": request["node_id"],
            "op": request["op"],
            "wait_ms": request.get("wait_ms"),
            "scheduler_peak": {
                "global": scheduler.max_active_global,
                "by_node": dict(scheduler.max_active_by_node),
            },
        }
        if outcome["ok"]:
            response = outcome["response"]
            summary.update(
                {
                    "ok": True,
                    "code": "OK",
                    "lease_id": response.get("lease_id"),
                    "generation": response.get("generation"),
                    "state": response.get("state"),
                    "checksum_digest": response.get("checksum_digest"),
                    "compute_ms": response.get("compute_ms"),
                    "quota_version": response.get("quota_version"),
                }
            )
        else:
            error = outcome["error"]
            payload = error.to_dict() if isinstance(error, MeshError) else {"code": "E_IO", "error": str(error)}
            summary.update({"ok": False, **payload})
        return summary
