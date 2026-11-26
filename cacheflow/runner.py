from __future__ import annotations

import base64
import threading
from pathlib import Path
from typing import Any

from .cluster import Cluster
from .content import expected_shard
from .engine import Engine
from .errors import CacheFlowError, StaleGenerationError
from .ledger import Ledger
from .report import conclusion_from_engine
from .scheduler import Job, JobContext, Scheduler
from .spec import Bundle


class DigestMismatch(CacheFlowError):
    code = "E_DIGEST"


class Runner:
    def __init__(
        self,
        bundle: Bundle,
        work_dir: str | Path,
        *,
        rpc_timeout_seconds: float = 8.0,
        cluster: Cluster | None = None,
        start_cluster: bool = True,
    ) -> None:
        self.bundle = bundle
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.ledger_path = self.work_dir / "cacheflow.sqlite3"
        self.ledger = Ledger(self.ledger_path)
        self.ledger.initialize_schema()
        self.ledger.initialize_spec(bundle.spec_digest(), self._public_spec())
        self.db_lock = threading.RLock()
        self.rpc_timeout_seconds = rpc_timeout_seconds
        self.owns_cluster = cluster is None
        if cluster is not None:
            self.cluster = cluster
        elif start_cluster:
            self.cluster = Cluster(list(bundle.nodes), ledger_path=self.ledger_path).start_all()
        else:
            self.cluster = Cluster(list(bundle.nodes), ledger_path=self.ledger_path)
        self.planner = Engine(bundle)
        self.plans: dict[int, dict[str, Any]] = {}
        self.node_by_seq: dict[int, str | None] = {}
        self._plan_events()

    def _public_spec(self) -> dict[str, Any]:
        return {
            "nodes": {k: v for k, v in self.bundle.nodes_doc.items() if not k.startswith("__")},
            "artifacts": {k: v for k, v in self.bundle.artifacts_doc.items() if not k.startswith("__")},
            "events": None
            if self.bundle.events_doc is None
            else {k: v for k, v in self.bundle.events_doc.items() if not k.startswith("__")},
        }

    def _plan_events(self) -> None:
        for event in self.bundle.events:
            seq = int(event["seq"])
            event_type = event["type"]
            if event_type == "grant":
                plan = self.planner.apply_spec_event(event)
                self.node_by_seq[seq] = plan["leases"][0]["node_id"]
            elif event_type == "quota_bump":
                plan = self.planner.apply_spec_event(event)
                self.node_by_seq[seq] = None
            else:
                lease_id = str(event["lease_id"])
                self.node_by_seq[seq] = self.planner.leases[lease_id]["node_id"]
                plan = self.planner.apply_spec_event(event)
            self.plans[seq] = plan

    def close(self) -> None:
        if self.owns_cluster:
            self.cluster.close()
        self.ledger.close()

    def __enter__(self) -> "Runner":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def _inflight_id(self, event: dict[str, Any]) -> str:
        return f"req:{int(event['seq'])}:{event.get('lease_id', event.get('type'))}"

    def _execute_event(self, event: dict[str, Any], context: JobContext) -> dict[str, Any]:
        seq = int(event["seq"])
        event_type = str(event["type"])
        plan = self.plans[seq]
        node_id = self.node_by_seq[seq]

        if event_type == "quota_bump":
            with self.db_lock:
                digest = self.ledger.commit_bundle(
                    event=event,
                    status="committed",
                    result=plan.get("result"),
                    transitions=plan.get("transitions"),
                    leases=plan.get("leases"),
                    holds=plan.get("holds"),
                    decisions=plan.get("decisions"),
                    reclaims=plan.get("reclaims"),
                )
            return {"committed": True, "commit_digest": digest}

        assert node_id is not None
        inflight_id = self._inflight_id(event)
        provisional = []
        if event_type == "grant":
            provisional = [dict(hold, status="provisional") for hold in plan.get("holds", [])]
        with self.db_lock:
            self.ledger.begin_inflight(
                request_id=inflight_id,
                event_seq=seq,
                node_id=node_id,
                lease_id=event.get("lease_id"),
                op=event_type,
                payload=event,
                wait_started_ms=context.wait_started_ms,
                wait_ended_ms=context.wait_ended_ms,
                provisional_holds=provisional,
            )

        client = self.cluster.client(node_id, timeout_seconds=self.rpc_timeout_seconds)
        try:
            response = self._rpc(event, node_id, client)
        except StaleGenerationError as exc:
            if event.get("expect_error") == "E_STALE":
                if exc.current_generation != plan.get("current_generation"):
                    raise
                with self.db_lock:
                    self.ledger.commit_bundle(
                        event=event,
                        status="rejected",
                        result={"current_generation": exc.current_generation},
                        error_code="E_STALE",
                        inflight_id=inflight_id,
                    )
                return {"expected_stale": True, "current_generation": exc.current_generation}
            raise
        finally:
            client.close()

        with self.db_lock:
            digest = self.ledger.commit_bundle(
                event=event,
                status="committed",
                result=response,
                transitions=plan.get("transitions"),
                leases=plan.get("leases"),
                holds=plan.get("holds"),
                decisions=plan.get("decisions"),
                reclaims=plan.get("reclaims"),
                inflight_id=inflight_id,
            )
        return {"response": response, "commit_digest": digest}

    def _rpc(self, event: dict[str, Any], node_id: str, client: Any) -> dict[str, Any]:
        event_type = str(event["type"])
        lease_id = str(event["lease_id"])
        if event_type == "grant":
            return client.call(
                "grant",
                lease_id=lease_id,
                artifact_key=event["artifact_key"],
                quota_version=int(event.get("quota_version", self.bundle.quota_version)),
                ttl_ms=int(event.get("ttl_ms", 600_000)),
            )
        if event_type == "write":
            shard_index = int(event.get("shard_index", 0))
            node = self.bundle.node(node_id)
            artifact = self.bundle.artifact(str(event.get("artifact_key", self.planner.leases[lease_id]["artifact_key"])))
            shard = artifact.shard(shard_index)
            content, expected_digest = expected_shard(node, artifact.artifact_key, shard)
            response = client.call(
                "write",
                lease_id=lease_id,
                artifact_key=artifact.artifact_key,
                shard_index=shard_index,
                content_b64=base64.b64encode(content).decode("ascii"),
                generation=event.get("generation"),
            )
            if response.get("digest") != expected_digest:
                raise DigestMismatch(
                    "node write digest does not match deterministic digest",
                    entry=lease_id,
                    field="digest",
                    details={"expected": expected_digest, "actual": response.get("digest")},
                )
            return response
        if event_type == "read":
            shard_index = int(event.get("shard_index", 0))
            lease = self.planner.leases[lease_id]
            artifact = self.bundle.artifact(lease["artifact_key"])
            shard = artifact.shard(shard_index)
            node = self.bundle.node(node_id)
            content, expected_digest = expected_shard(node, artifact.artifact_key, shard)
            response = client.call(
                "read",
                lease_id=lease_id,
                artifact_key=artifact.artifact_key,
                shard_index=shard_index,
                generation=event.get("generation"),
            )
            if response.get("digest") != expected_digest:
                raise DigestMismatch("read digest mismatch", entry=lease_id, field="digest")
            decoded = base64.b64decode(response["content_b64"], validate=True)
            if decoded != content:
                raise DigestMismatch("read bytes mismatch", entry=lease_id, field="content_b64")
            return response
        if event_type == "renew":
            return client.call(
                "renew",
                lease_id=lease_id,
                generation=int(event["generation"]),
                ttl_ms=int(event.get("ttl_ms", 600_000)),
            )
        if event_type == "release":
            return client.call("release", lease_id=lease_id, generation=int(event["generation"]))
        if event_type == "expire":
            return client.call(
                "expire",
                lease_id=lease_id,
                generation=int(event["generation"]),
                wait=True,
            )
        raise CacheFlowError(f"unsupported runtime event type: {event_type}", field="type")

    def run(self) -> dict[str, Any]:
        node_limits = {node.node_id: node.max_leases for node in self.bundle.nodes}
        scheduler = Scheduler(global_limit=self.bundle.global_concurrency, node_limits=node_limits)
        jobs: list[Job] = []
        for event in self.bundle.events:
            seq = int(event["seq"])
            event_type = str(event["type"])
            group = None if event_type == "quota_bump" else str(event.get("lease_id"))
            jobs.append(
                Job(
                    seq=seq,
                    node_id=self.node_by_seq[seq],
                    group=group,
                    exclusive=event_type == "quota_bump",
                    fn=lambda context, current=dict(event): self._execute_event(current, context),
                )
            )
        results = scheduler.run_all(jobs)
        wait_records = scheduler.ordered_wait_records()
        with self.db_lock:
            self.ledger.mark_wait_records(wait_records)

        errors: list[dict[str, Any]] = []
        for seq in sorted(results):
            result = results[seq]
            if not result.ok:
                error = result.error
                errors.append(
                    {
                        "seq": seq,
                        "type": type(error).__name__,
                        "code": getattr(error, "code", "E_RUNTIME"),
                        "message": str(error),
                    }
                )
        if errors:
            raise CacheFlowError("pipeline execution failed", details={"errors": errors})

        cutoff = max((int(event["seq"]) for event in self.bundle.events), default=0)
        conclusion = conclusion_from_engine(self.bundle, self.planner, cutoff_seq=cutoff)
        with self.db_lock:
            digest, raw = self.ledger.save_snapshot(cutoff_seq=cutoff, report=conclusion)
        return {
            "conclusion": conclusion,
            "snapshot_digest": digest,
            "snapshot_bytes": raw,
            "wait_records": wait_records,
            "peaks": scheduler.peaks(),
        }
