from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from .arbitration import ClusterCapacity, arbitrate_candidates
from .content import build_shard
from .errors import ArbitrationError, SpecError, UnknownBatchError
from .ledger import Ledger, wall_ms
from .process import NodeCluster
from .protocol import monotonic_ms
from .scheduler import DeliveryScheduler, QuotaGate
from .specs import Bundle, EventSpec, load_bundle


class SampleRunner:
    def __init__(
        self,
        sample_dir: str | Path,
        state_dir: str | Path,
        *,
        crash_before_seq: int | None = None,
        crash_attempt: int = 1,
    ) -> None:
        self.sample_dir = Path(sample_dir)
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.inflight_dir = self.state_dir / "inflight"
        self.unknown_dir = self.state_dir / "unknown"
        self.resolved_dir = self.state_dir / "resolved"
        for path in (self.inflight_dir, self.unknown_dir, self.resolved_dir):
            path.mkdir(parents=True, exist_ok=True)
        self.db_path = self.state_dir / "ledger.sqlite"
        self.crash_before_seq = crash_before_seq
        self.crash_attempt = crash_attempt
        self.current_attempt = 1
        self.bundle: Bundle = load_bundle(self.sample_dir)
        self.ledger = Ledger(self.db_path)
        self.ledger.initialize_bundle(self.bundle)
        self.active_nodes = json.loads(self.ledger.meta("active_nodes", "[]"))
        self.quota_version = int(self.ledger.meta("current_quota_version", "1"))
        quota_config_raw = self.ledger.meta("quota_config")
        if quota_config_raw:
            self.quota_config = json.loads(quota_config_raw)
        else:
            self.quota_config = {
                "global_concurrency": self.bundle.global_concurrency,
                "stage_quotas": {stage.stage_id: stage.concurrency_quota for stage in self.bundle.pipeline.stages},
                "node_concurrency": dict(self.bundle.node_concurrency),
            }
        self.cluster: NodeCluster | None = None
        self.scheduler: DeliveryScheduler | None = None
        self.outcomes: list[dict[str, Any]] = []

    def close(self) -> None:
        if self.scheduler is not None:
            self.scheduler.shutdown()
        if self.cluster is not None:
            self.cluster.stop_all()
        self.ledger.close()

    def __enter__(self) -> "SampleRunner":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def inflight_markers(self) -> list[Path]:
        return sorted(self.inflight_dir.glob("*.json"))

    def recover_inflight(self) -> list[dict[str, Any]]:
        recovered: list[dict[str, Any]] = []
        for path in self.inflight_markers():
            marker = json.loads(path.read_text(encoding="utf-8"))
            self.ledger.mark_unknown_batch(marker)
            target = self.unknown_dir / path.name
            path.replace(target)
            recovered.append(marker)
        return recovered

    def _write_marker(self, event: EventSpec, attempt: int, replacement_for: str | None) -> tuple[str, Path]:
        batch_id = f"batch-{event.seq:05d}-a{attempt}"
        marker = {
            "batch_id": batch_id,
            "event_seq": event.seq,
            "attempt": attempt,
            "replacement_for": replacement_for,
            "pid": os.getpid(),
            "started_wall_ms": wall_ms(),
            "started_monotonic_ms": monotonic_ms(),
            "event": event.data,
        }
        path = self.inflight_dir / f"{batch_id}.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(marker, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, path)
        return batch_id, path

    def _crash_if_requested(self, event_seq: int) -> None:
        if (
            self.crash_before_seq is not None
            and event_seq == self.crash_before_seq
            and self.current_attempt == self.crash_attempt
        ):
            os._exit(97)

    def _capacity(self) -> ClusterCapacity:
        capacity = ClusterCapacity.from_specs(self.bundle.nodes)
        for node_id, state in self.ledger.node_states().items():
            item = capacity.nodes[node_id]
            item.used_slots = int(state["used_slots"])
            item.active_leases = int(state["active_leases"])
        for lease in self.ledger.active_leases():
            capacity.leases[lease["lease_id"]] = lease["node_id"]
        return capacity

    def _gate(self, version: int, event: EventSpec | None = None) -> QuotaGate:
        stages = dict(self.quota_config.get("stage_quotas", {}))
        nodes = dict(self.quota_config.get("node_concurrency", {}))
        global_concurrency = int(self.quota_config.get("global_concurrency", self.bundle.global_concurrency))
        if event is not None and event.type == "quota_change":
            global_concurrency = int(event.get("global_concurrency", global_concurrency))
            stages.update({key: int(value) for key, value in (event.get("stage_quotas") or {}).items()})
            nodes.update({key: int(value) for key, value in (event.get("node_concurrency") or {}).items()})
        return QuotaGate(
            version=version,
            global_concurrency=global_concurrency,
            stage_quotas=stages,
            node_concurrency=nodes,
        )

    def _watermarks(
        self,
        capacity: ClusterCapacity,
        version: int,
        event_seq: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
        scale = self.bundle.pipeline.watermark_scale
        records: list[dict[str, Any]] = []
        conclusions: list[dict[str, Any]] = []
        states: dict[str, dict[str, Any]] = {}
        for node_id in sorted(capacity.nodes):
            node_capacity = capacity.nodes[node_id]
            watermark = node_capacity.watermark(scale)
            watermark["quota_version"] = version
            watermark["high_watermark"] = (
                int(watermark["watermark_fixed"]) >= self.bundle.pipeline.high_watermark_fixed
            )
            records.append(
                {
                    "record_type": "watermark",
                    "node_id": node_id,
                    "quota_version": version,
                    "payload": watermark,
                }
            )
            conclusions.append(
                {
                    "conclusion_id": f"wm-v{version}-e{event_seq}-{node_id}",
                    "kind": "watermark",
                    "node_id": node_id,
                    "quota_version": version,
                    "payload": watermark,
                }
            )
            spec = self.bundle.node(node_id)
            states[node_id] = {
                "quota_version": version,
                "capacity_slots": spec.capacity_slots,
                "lease_limit": spec.lease_limit,
                "used_slots": node_capacity.used_slots,
                "active_leases": node_capacity.active_leases,
            }
        return records, conclusions, states

    def _commit(
        self,
        *,
        event: EventSpec,
        batch_id: str,
        attempt: int,
        records: list[dict[str, Any]],
        capacity: ClusterCapacity,
        leases: list[dict[str, Any]] | None = None,
        lease_updates: list[dict[str, Any]] | None = None,
        conclusions: list[dict[str, Any]] | None = None,
        quota_replacement: bool = False,
        replacement_for: str | None = None,
        active_nodes: list[str] | None = None,
        meta_updates: dict[str, str] | None = None,
    ) -> str:
        watermark_records, watermark_conclusions, node_states = self._watermarks(
            capacity, self.quota_version, event.seq
        )
        all_records = [*records, *watermark_records]
        all_conclusions = [*(conclusions or []), *watermark_conclusions]
        self._crash_if_requested(event.seq)
        digest = self.ledger.commit_event_batch(
            batch_id=batch_id,
            attempt=attempt,
            event=event,
            records=all_records,
            node_states=node_states,
            leases=leases,
            lease_updates=lease_updates,
            conclusions=all_conclusions,
            current_quota_version=self.quota_version,
            active_nodes=active_nodes if active_nodes is not None else self.active_nodes,
                quota_replacement=quota_replacement,
                replacement_for=replacement_for,
                meta_updates=meta_updates,
            )
        return digest

    def run(self) -> dict[str, Any]:
        previous_report = self.state_dir / "run-report.json"
        previous_report.unlink(missing_ok=True)
        if self.inflight_markers():
            raise UnknownBatchError(
                "inflight batch marker exists; run recover before rerunning",
                field="inflight",
                details={"markers": [path.name for path in self.inflight_markers()]},
            )
        self.cluster = NodeCluster(
            self.bundle.nodes,
            run_dir=self.state_dir / "cluster",
            quota_version=self.quota_version,
            shard_bytes=self.bundle.shard_bytes,
            handshake_timeout_ms=self.bundle.handshake_timeout_ms,
            request_timeout_ms=self.bundle.request_timeout_ms,
        )
        self.cluster.start_all(self.active_nodes)
        self.scheduler = DeliveryScheduler(self._gate(self.quota_version))
        self._rehydrate_active_leases()
        try:
            for event in self.bundle.events:
                if event.seq <= self.ledger.last_event_seq:
                    continue
                unknown = self.ledger.conn.execute(
                    "SELECT * FROM unknown_batches WHERE event_seq=? AND status='open' ORDER BY rowid DESC LIMIT 1",
                    (event.seq,),
                ).fetchone()
                replacement_for = unknown["batch_id"] if unknown else None
                attempt = int(json.loads(unknown["marker_json"])["attempt"]) + 1 if unknown else 1
                self.current_attempt = attempt
                batch_id, marker_path = self._write_marker(event, attempt, replacement_for)
                outcome = self._dispatch(event, batch_id, attempt, replacement_for)
                marker_path.unlink(missing_ok=True)
                if replacement_for:
                    old = self.unknown_dir / f"{replacement_for}.json"
                    if old.exists():
                        old.replace(self.resolved_dir / old.name)
                self.outcomes.append(outcome)
        finally:
            if self.scheduler is not None:
                self.scheduler.shutdown(wait=False)
                self.scheduler = None
            if self.cluster is not None:
                self.cluster.stop_all()
                self.cluster = None

        report = {
            "status": "completed",
            "last_event_seq": self.ledger.last_event_seq,
            "outcomes": self.outcomes,
            "ledger": str(self.db_path),
            "ledger_format_version": self.ledger.format_version,
            "ledger_digest": self.ledger.head_digest,
        }
        (self.state_dir / "run-report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
        )
        return report

    def _rehydrate_active_leases(self) -> None:
        assert self.cluster is not None
        now_wall = wall_ms()
        for lease in self.ledger.active_leases():
            remaining_ms = max(0, int(lease["expiry_wall_ms"]) - now_wall)
            self.cluster.call(
                lease["node_id"],
                {
                    "action": "rehydrate_lease",
                    "lease_id": lease["lease_id"],
                    "lease_version": lease["lease_version"],
                    "remaining_ms": remaining_ms,
                    "artifact_key": lease["artifact_key"],
                    "shard_index": lease["shard_index"],
                    "quota_version": lease["quota_version"],
                    "request_seq": "recovery-rehydrate",
                },
            )

    def _dispatch(
        self,
        event: EventSpec,
        batch_id: str,
        attempt: int,
        replacement_for: str | None,
    ) -> dict[str, Any]:
        if event.type == "write":
            return self._handle_write(event, batch_id, attempt, replacement_for)
        if event.type == "read":
            return self._handle_read(event, batch_id, attempt, replacement_for)
        if event.type == "renew":
            return self._handle_renew(event, batch_id, attempt, replacement_for)
        if event.type == "release":
            return self._handle_release(event, batch_id, attempt, replacement_for)
        if event.type == "wait":
            return self._handle_wait(event, batch_id, attempt, replacement_for)
        if event.type == "quota_change":
            return self._handle_quota_change(event, batch_id, attempt, replacement_for)
        raise SpecError(f"unsupported event type {event.type}", entry_id=f"seq={event.seq}", field="type")

    def _queue_records(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for result in results:
            metric = result.get("queue")
            if metric:
                records.append(
                    {
                        "record_type": "queue_metric",
                        "node_id": metric.get("node_id"),
                        "quota_version": metric.get("quota_version"),
                        "payload": metric,
                    }
                )
        return records

    def _error_record(self, event: EventSpec, code: str, message: str, **extra: Any) -> dict[str, Any]:
        payload = {"code": code, "message": message, "event_seq": event.seq, **extra}
        return {"record_type": "error", "quota_version": self.quota_version, "payload": payload}

    def _handle_write(
        self,
        event: EventSpec,
        batch_id: str,
        attempt: int,
        replacement_for: str | None,
    ) -> dict[str, Any]:
        assert self.cluster is not None and self.scheduler is not None
        key = str(event.get("artifact_key"))
        shard_index = int(event.get("shard_index", 0))
        lease_ms = int(event.get("lease_ms", 60_000))
        artifact = self.bundle.artifact(key)
        candidate_ids = [str(value) for value in (event.get("candidate_nodes") or self.active_nodes)]
        capacity = self._capacity()
        request_seq = int(event.get("request_seq", event.seq))

        tasks = []
        for node_id in candidate_ids:
            tasks.append(
                {
                    "event_seq": event.seq,
                    "stage": artifact.stage,
                    "node_id": node_id,
                    "fn": lambda node_id=node_id: self.cluster.call(  # type: ignore[union-attr]
                        node_id,
                        {
                            "action": "probe",
                            "artifact_key": key,
                            "shard_index": shard_index,
                            "request_seq": f"probe-{request_seq}-{node_id}",
                        },
                    ),
                }
            )
        probe_results = self.scheduler.parallel(tasks)
        records = self._queue_records(probe_results)
        candidates: list[dict[str, Any]] = []
        local_shards: dict[str, dict[str, Any]] = {}

        for node_id, result in zip(candidate_ids, probe_results):
            expected = build_shard(
                node_id=node_id,
                node_seed=self.bundle.node(node_id).seed,
                artifact_key=key,
                shard_index=shard_index,
                byte_count=self.bundle.shard_bytes,
            )
            local_shards[node_id] = expected
            node_capacity = capacity.nodes[node_id]
            candidate: dict[str, Any] = {
                "node_id": node_id,
                "remaining_slots": node_capacity.remaining_slots,
                "active_leases": node_capacity.active_leases,
                "lease_remaining": node_capacity.lease_remaining,
                "digest_match": False,
                "eligible": True,
                "reasons": [],
            }
            if node_capacity.remaining_slots <= 0:
                candidate["eligible"] = False
                candidate["reasons"].append("NO_CAPACITY_SLOTS")
            if node_capacity.lease_remaining <= 0:
                candidate["eligible"] = False
                candidate["reasons"].append("LEASE_LIMIT_REACHED")
            if not result["ok"]:
                candidate["eligible"] = False
                candidate["reasons"].append("PROBE_FAILED")
                candidate["probe_error"] = result["error"]
            else:
                response = result["value"]
                candidate["probe"] = {
                    "digest": response.get("digest"),
                    "shard_bytes": response.get("shard_bytes"),
                    "source_version": response.get("source_version"),
                    "elapsed_ms": response.get("elapsed_ms"),
                    "quota_version": response.get("quota_version"),
                }
                candidate["digest_match"] = (
                    response.get("digest") == expected["digest"]
                    and int(response.get("shard_bytes", -1)) == expected["shard_bytes"]
                    and int(response.get("source_version", -1)) == expected["source_version"]
                )
                if not candidate["digest_match"]:
                    candidate["reasons"].append("DIGEST_MISMATCH")
            candidates.append(candidate)

        decision = arbitrate_candidates(candidates)
        winner = decision.get("winner_node_id")
        records.append(
            {
                "record_type": "arbitration",
                "artifact_key": key,
                "node_id": winner,
                "quota_version": self.quota_version,
                "payload": {**decision, "shard_index": shard_index, "artifact_key": key},
            }
        )
        conclusions = [
            {
                "conclusion_id": f"arb-v{self.quota_version}-e{event.seq}-{key}-{shard_index}",
                "kind": "arbitration",
                "artifact_key": key,
                "shard_index": shard_index,
                "node_id": winner,
                "quota_version": self.quota_version,
                "payload": {**decision, "shard_index": shard_index, "artifact_key": key},
            }
        ]

        if not winner:
            records.append(self._error_record(event, "E_NO_ELIGIBLE_NODE", "arbitration found no eligible node", decision=decision))
            digest = self._commit(
                event=event,
                batch_id=batch_id,
                attempt=attempt,
                records=records,
                capacity=capacity,
                conclusions=[],
                replacement_for=replacement_for,
            )
            return {"seq": event.seq, "outcome": "no_eligible_node", "digest": digest}

        reservation_id = f"RSV-e{event.seq}-a{attempt}-{winner}"
        capacity.reserve(reservation_id, winner)
        reserved_watermark = capacity.nodes[winner].watermark(self.bundle.pipeline.watermark_scale)
        records.append(
            {
                "record_type": "reserve",
                "artifact_key": key,
                "node_id": winner,
                "reservation_id": reservation_id,
                "quota_version": self.quota_version,
                "payload": {
                    "reservation_id": reservation_id,
                    "phase": "reserved_before_write",
                    "watermark": reserved_watermark,
                },
            }
        )

        if bool(event.get("cancel_after_reserve", False)):
            capacity.rollback_reservation(reservation_id)
            cancelled_watermark = capacity.nodes[winner].watermark(self.bundle.pipeline.watermark_scale)
            records.append(
                {
                    "record_type": "rollback",
                    "artifact_key": key,
                    "node_id": winner,
                    "reservation_id": reservation_id,
                    "quota_version": self.quota_version,
                    "payload": {
                        "reservation_id": reservation_id,
                        "reason": "event_cancelled_after_reserve",
                        "watermark_after_rollback": cancelled_watermark,
                    },
                }
            )
            digest = self._commit(
                event=event,
                batch_id=batch_id,
                attempt=attempt,
                records=records,
                capacity=capacity,
                conclusions=conclusions,
                replacement_for=replacement_for,
            )
            return {"seq": event.seq, "outcome": "cancelled_rolled_back", "winner": winner, "digest": digest}

        write_error: Exception | None = None
        response: dict[str, Any] | None = None
        queue_metric: dict[str, Any] | None = None
        try:
            response, queue_metric = self.scheduler.run(
                event_seq=event.seq,
                stage=artifact.stage,
                node_id=winner,
                fn=lambda: self.cluster.call(  # type: ignore[union-attr]
                    winner,
                    {
                        "action": "write",
                        "artifact_key": key,
                        "shard_index": shard_index,
                        "lease_ms": lease_ms,
                        "request_seq": request_seq,
                        "quota_version": self.quota_version,
                        "fail_write": bool(event.get("fail_write", False)),
                    },
                ),
            )
        except Exception as exc:  # rollback path is asserted by tests/sample
            write_error = exc
            queue_metric = getattr(exc, "queue_metric", None)

        if queue_metric:
            records.append({"record_type": "queue_metric", "node_id": winner, "payload": queue_metric})

        required = ("lease_id", "digest", "quota_version", "elapsed_ms")
        field_error: str | None = None
        if write_error is None and response is not None:
            for field_name in required:
                if response.get(field_name) is None:
                    field_error = field_name
                    break
            if field_error is None and response.get("digest") != local_shards[winner]["digest"]:
                field_error = "digest"
                write_error = ArbitrationError("write receipt digest does not match local derivation", code="E_DIGEST", field="digest")
            elif field_error is None and response.get("node_id", winner) != winner:
                field_error = "node_id"

        if write_error is not None or field_error is not None:
            error_info = (
                write_error.to_dict()
                if write_error is not None and hasattr(write_error, "to_dict")
                else {"code": getattr(write_error, "code", "E_FIELD" if field_error else "E_WRITE"), "message": str(write_error or f"missing field {field_error}")}
            )
            capacity.rollback_reservation(reservation_id)
            rolled_watermark = capacity.nodes[winner].watermark(self.bundle.pipeline.watermark_scale)
            records.append(
                {
                    "record_type": "rollback",
                    "artifact_key": key,
                    "node_id": winner,
                    "reservation_id": reservation_id,
                    "quota_version": self.quota_version,
                    "payload": {
                        "reservation_id": reservation_id,
                        "reason": "write_failed_or_cancelled",
                        "error": error_info,
                        "watermark_after_rollback": rolled_watermark,
                    },
                }
            )
            records.append(self._error_record(event, str(error_info.get("code", "E_WRITE")), str(error_info.get("message")), reservation_id=reservation_id))
            digest = self._commit(
                event=event,
                batch_id=batch_id,
                attempt=attempt,
                records=records,
                capacity=capacity,
                conclusions=conclusions,
                replacement_for=replacement_for,
            )
            return {"seq": event.seq, "outcome": "rolled_back", "winner": winner, "error": error_info, "digest": digest}

        assert response is not None
        lease_id = str(response["lease_id"])
        new_node_id = capacity.grant(reservation_id, lease_id)
        now_wall = wall_ms()
        now_monotonic = monotonic_ms()
        lease = {
            "lease_id": lease_id,
            "lease_ref": event.get("lease_ref"),
            "node_id": new_node_id,
            "artifact_key": key,
            "shard_index": shard_index,
            "reservation_id": reservation_id,
            "quota_version": self.quota_version,
            "lease_version": int(response.get("lease_version", 1)),
            "status": "active",
            "grant_wall_ms": now_wall,
            "expiry_wall_ms": now_wall + lease_ms,
            "grant_monotonic_ms": now_monotonic,
            "expiry_monotonic_ms": now_monotonic + lease_ms,
            "ttl_ms": lease_ms,
            "close_reason": None,
            "closed_event_seq": None,
        }
        records.append(
            {
                "record_type": "lease_grant",
                "artifact_key": key,
                "node_id": winner,
                "lease_id": lease_id,
                "lease_ref": event.get("lease_ref"),
                "reservation_id": reservation_id,
                "quota_version": self.quota_version,
                "payload": {
                    **lease,
                    "content_digest": response["digest"],
                    "shard_bytes": local_shards[winner]["shard_bytes"],
                    "source_version": local_shards[winner]["source_version"],
                    "node_elapsed_ms": response.get("elapsed_ms"),
                },
            }
        )
        digest = self._commit(
            event=event,
            batch_id=batch_id,
            attempt=attempt,
            records=records,
            capacity=capacity,
            leases=[lease],
            conclusions=conclusions,
            replacement_for=replacement_for,
        )
        return {"seq": event.seq, "outcome": "granted", "winner": winner, "lease_id": lease_id, "digest": digest}

    def _latest_owner(self, key: str, shard_index: int) -> str | None:
        row = self.ledger.conn.execute(
            """
            SELECT node_id FROM conclusions
             WHERE kind='arbitration' AND artifact_key=? AND shard_index=?
             ORDER BY event_seq DESC, rowid DESC LIMIT 1
            """,
            (key, shard_index),
        ).fetchone()
        return str(row["node_id"]) if row and row["node_id"] else None

    def _handle_read(self, event, batch_id, attempt, replacement_for):
        assert self.cluster is not None and self.scheduler is not None
        key = str(event.get("artifact_key"))
        shard_index = int(event.get("shard_index", 0))
        node_id = str(event.get("node_id") or self._latest_owner(key, shard_index) or "")
        capacity = self._capacity()
        records: list[dict[str, Any]] = []
        if node_id not in self.bundle.node_ids:
            records.append(self._error_record(event, "E_OWNERSHIP", "no committed arbitration owner for artifact"))
            digest = self._commit(event=event, batch_id=batch_id, attempt=attempt, records=records, capacity=capacity, replacement_for=replacement_for)
            return {"seq": event.seq, "outcome": "no_owner", "digest": digest}
        try:
            response, metric = self.scheduler.run(
                event_seq=event.seq,
                stage=self.bundle.artifact(key).stage,
                node_id=node_id,
                fn=lambda: self.cluster.call(
                    node_id,
                    {"action": "read", "artifact_key": key, "shard_index": shard_index, "request_seq": event.seq},
                ),
            )
        except Exception as exc:
            metric = getattr(exc, "queue_metric", None)
            error = exc.to_dict() if hasattr(exc, "to_dict") else {"code": "E_READ", "message": str(exc)}
            if metric:
                records.append({"record_type": "queue_metric", "node_id": node_id, "payload": metric})
            records.append(self._error_record(event, str(error.get("code")), str(error.get("message")), node_id=node_id))
            digest = self._commit(event=event, batch_id=batch_id, attempt=attempt, records=records, capacity=capacity, replacement_for=replacement_for)
            return {"seq": event.seq, "outcome": "read_failed", "error": error, "digest": digest}
        if metric:
            records.append({"record_type": "queue_metric", "node_id": node_id, "payload": metric})
        expected = build_shard(
            node_id=node_id,
            node_seed=self.bundle.node(node_id).seed,
            artifact_key=key,
            shard_index=shard_index,
            byte_count=self.bundle.shard_bytes,
        )
        match = response.get("digest") == expected["digest"]
        records.append(
            {
                "record_type": "read_receipt",
                "artifact_key": key,
                "node_id": node_id,
                "quota_version": self.quota_version,
                "payload": {"node_id": node_id, "digest": response.get("digest"), "digest_match": match, "elapsed_ms": response.get("elapsed_ms")},
            }
        )
        digest = self._commit(event=event, batch_id=batch_id, attempt=attempt, records=records, capacity=capacity, replacement_for=replacement_for)
        return {"seq": event.seq, "outcome": "read", "winner": node_id, "digest_match": match, "digest": digest}

    def _active_lease_by_ref(self, lease_ref: str) -> dict[str, Any] | None:
        row = self.ledger.conn.execute(
            "SELECT * FROM leases WHERE lease_ref=? AND status='active' ORDER BY grant_wall_ms DESC LIMIT 1",
            (lease_ref,),
        ).fetchone()
        return dict(row) if row else None

    def _handle_renew(self, event, batch_id, attempt, replacement_for):
        assert self.cluster is not None and self.scheduler is not None
        lease_ref = str(event.get("lease_ref"))
        expected_version = int(event.get("expected_version"))
        lease_ms = int(event.get("lease_ms", 60_000))
        lease = self._active_lease_by_ref(lease_ref)
        capacity = self._capacity()
        records: list[dict[str, Any]] = []
        if lease is None:
            records.append(self._error_record(event, "E_LEASE", "no active lease for lease_ref", lease_ref=lease_ref))
            digest = self._commit(event=event, batch_id=batch_id, attempt=attempt, records=records, capacity=capacity, replacement_for=replacement_for)
            return {"seq": event.seq, "outcome": "rejected", "code": "E_LEASE", "digest": digest}
        try:
            response, metric = self.scheduler.run(
                event_seq=event.seq,
                stage=self.bundle.artifact(lease["artifact_key"]).stage,
                node_id=lease["node_id"],
                fn=lambda: self.cluster.call(
                    lease["node_id"],
                    {
                        "action": "renew",
                        "lease_id": lease["lease_id"],
                        "lease_version": expected_version,
                        "lease_ms": lease_ms,
                        "request_seq": event.seq,
                        "quota_version": self.quota_version,
                    },
                ),
            )
        except Exception as exc:
            metric = getattr(exc, "queue_metric", None)
            error = exc.to_dict() if hasattr(exc, "to_dict") else {"code": "E_LEASE", "message": str(exc)}
            if metric:
                records.append({"record_type": "queue_metric", "node_id": lease["node_id"], "payload": metric})
            records.append(
                {
                    "record_type": "renewal_rejected",
                    "artifact_key": lease["artifact_key"],
                    "node_id": lease["node_id"],
                    "lease_id": lease["lease_id"],
                    "lease_ref": lease_ref,
                    "quota_version": self.quota_version,
                    "payload": {"expected_version": expected_version, "current_version": lease["lease_version"], "error": error},
                }
            )
            digest = self._commit(event=event, batch_id=batch_id, attempt=attempt, records=records, capacity=capacity, replacement_for=replacement_for)
            return {"seq": event.seq, "outcome": "renew_rejected", "code": "E_LEASE", "digest": digest}
        if metric:
            records.append({"record_type": "queue_metric", "node_id": lease["node_id"], "payload": metric})
        new_version = int(response["lease_version"])
        now_wall = wall_ms()
        now_monotonic = monotonic_ms()
        lease.update(
            {
                "lease_version": new_version,
                "ttl_ms": lease_ms,
                "expiry_wall_ms": now_wall + lease_ms,
                "expiry_monotonic_ms": now_monotonic + lease_ms,
            }
        )
        records.append(
            {
                "record_type": "renew",
                "artifact_key": lease["artifact_key"],
                "node_id": lease["node_id"],
                "lease_id": lease["lease_id"],
                "lease_ref": lease_ref,
                "quota_version": self.quota_version,
                "payload": {"old_version": expected_version, "new_version": new_version, "extended_ms": lease_ms},
            }
        )
        digest = self._commit(event=event, batch_id=batch_id, attempt=attempt, records=records, capacity=capacity, leases=[lease], replacement_for=replacement_for)
        return {"seq": event.seq, "outcome": "renewed", "lease_version": new_version, "digest": digest}

    def _handle_release(self, event, batch_id, attempt, replacement_for):
        assert self.cluster is not None and self.scheduler is not None
        lease_ref = str(event.get("lease_ref"))
        lease = self._active_lease_by_ref(lease_ref)
        capacity = self._capacity()
        records: list[dict[str, Any]] = []
        if lease is None:
            records.append(self._error_record(event, "E_LEASE", "no active lease for release", lease_ref=lease_ref))
            digest = self._commit(event=event, batch_id=batch_id, attempt=attempt, records=records, capacity=capacity, replacement_for=replacement_for)
            return {"seq": event.seq, "outcome": "release_unknown", "digest": digest}
        try:
            self.cluster.call(
                lease["node_id"],
                {"action": "release", "lease_id": lease["lease_id"], "request_seq": event.seq},
            )
        except Exception as exc:
            error = exc.to_dict() if hasattr(exc, "to_dict") else {"code": "E_RELEASE", "message": str(exc)}
            records.append(self._error_record(event, str(error.get("code")), str(error.get("message")), lease_id=lease["lease_id"]))
            digest = self._commit(event=event, batch_id=batch_id, attempt=attempt, records=records, capacity=capacity, replacement_for=replacement_for)
            return {"seq": event.seq, "outcome": "release_failed", "digest": digest}
        capacity.release_lease(lease["lease_id"])
        update = {"lease_id": lease["lease_id"], "status": "released", "close_reason": "user_release", "closed_event_seq": event.seq}
        records.append(
            {
                "record_type": "release",
                "artifact_key": lease["artifact_key"],
                "node_id": lease["node_id"],
                "lease_id": lease["lease_id"],
                "lease_ref": lease_ref,
                "quota_version": self.quota_version,
                "payload": update,
            }
        )
        digest = self._commit(event=event, batch_id=batch_id, attempt=attempt, records=records, capacity=capacity, lease_updates=[update], replacement_for=replacement_for)
        return {"seq": event.seq, "outcome": "released", "digest": digest}

    def _handle_wait(self, event, batch_id, attempt, replacement_for):
        assert self.cluster is not None
        duration_ms = int(event.get("duration_ms", 0))
        started = monotonic_ms()
        time.sleep(duration_ms / 1000.0)
        ended = monotonic_ms()
        capacity = self._capacity()
        records: list[dict[str, Any]] = [
            {"record_type": "wait", "quota_version": self.quota_version, "payload": {"duration_ms": duration_ms, "start_monotonic_ms": started, "end_monotonic_ms": ended}}
        ]
        updates: list[dict[str, Any]] = []
        now = monotonic_ms()
        rows = self.ledger.conn.execute("SELECT * FROM leases WHERE status='active' AND expiry_monotonic_ms <= ? ORDER BY lease_id", (now,)).fetchall()
        for row in rows:
            lease = dict(row)
            try:
                self.cluster.call(lease["node_id"], {"action": "release", "lease_id": lease["lease_id"], "request_seq": f"expire-{event.seq}"}, timeout_ms=500)
            except Exception:
                pass
            capacity.release_lease(lease["lease_id"])
            update = {"lease_id": lease["lease_id"], "status": "expired", "close_reason": "expiry", "closed_event_seq": event.seq}
            updates.append(update)
            records.append(
                {
                    "record_type": "reclaim",
                    "artifact_key": lease["artifact_key"],
                    "node_id": lease["node_id"],
                    "lease_id": lease["lease_id"],
                    "quota_version": lease["quota_version"],
                    "payload": {
                        "entitlement_id": lease["lease_id"],
                        "reason": "expiry",
                        "released_slot": True,
                        "released_lease_count": True,
                        "basis": "real_monotonic_clock_ttl_elapsed",
                        "at_monotonic_ms": now,
                    },
                }
            )
        digest = self._commit(event=event, batch_id=batch_id, attempt=attempt, records=records, capacity=capacity, lease_updates=updates, replacement_for=replacement_for)
        return {"seq": event.seq, "outcome": "waited", "expired": len(updates), "digest": digest}

    def _handle_quota_change(self, event, batch_id, attempt, replacement_for):
        assert self.cluster is not None and self.scheduler is not None
        new_version = int(event.get("quota_version"))
        if new_version <= self.quota_version:
            raise SpecError(
                f"quota_version must increase from {self.quota_version}",
                entry_id=f"seq={event.seq}",
                field="quota_version",
            )
        old_active_nodes = list(self.active_nodes)
        new_active_nodes = [str(value) for value in (event.get("active_nodes") or old_active_nodes)]
        added = [node_id for node_id in new_active_nodes if node_id not in old_active_nodes]
        removed = [node_id for node_id in old_active_nodes if node_id not in new_active_nodes]

        self.quota_version = new_version
        self.active_nodes = new_active_nodes
        self.cluster.quota_version = new_version
        for node_id in added:
            self.cluster.start_node(node_id)

        capacity = self._capacity()
        records: list[dict[str, Any]] = [
            {
                "record_type": "quota_replacement",
                "quota_version": new_version,
                "payload": {
                    "from_version": new_version - 1 if not event.get("active_nodes") else None,
                    "quota_version": new_version,
                    "active_nodes": new_active_nodes,
                    "added_nodes": added,
                    "removed_nodes": removed,
                    "old_conclusions": "marked_stale",
                },
            }
        ]
        updates: list[dict[str, Any]] = []
        rows = self.ledger.conn.execute(
            "SELECT * FROM leases WHERE status='active' AND quota_version < ? ORDER BY lease_id",
            (new_version,),
        ).fetchall()
        for row in rows:
            lease = dict(row)
            capacity.release_lease(lease["lease_id"])
            update = {"lease_id": lease["lease_id"], "status": "revoked_quota", "close_reason": "quota_replacement", "closed_event_seq": event.seq}
            updates.append(update)
            records.append(
                {
                    "record_type": "reclaim",
                    "artifact_key": lease["artifact_key"],
                    "node_id": lease["node_id"],
                    "lease_id": lease["lease_id"],
                    "quota_version": new_version,
                    "payload": {
                        "entitlement_id": lease["lease_id"],
                        "reason": "quota_replacement",
                        "released_slot": True,
                        "released_lease_count": True,
                        "basis": "quota_version_or_topology_replacement",
                        "old_quota_version": lease["quota_version"],
                        "new_quota_version": new_version,
                    },
                }
            )

        gate = self._gate(new_version, event)
        next_quota_config = {
            "global_concurrency": gate.global_concurrency,
            "stage_quotas": gate.stage_quotas,
            "node_concurrency": gate.node_concurrency,
        }
        digest = self._commit(
            event=event,
            batch_id=batch_id,
            attempt=attempt,
            records=records,
            capacity=capacity,
            lease_updates=updates,
            quota_replacement=True,
            replacement_for=replacement_for,
            active_nodes=new_active_nodes,
            meta_updates={"quota_config": json.dumps(next_quota_config, sort_keys=True)},
        )
        self.quota_config = next_quota_config
        self.scheduler.replace_gate(gate)
        try:
            self.cluster.configure_quota(new_version, new_active_nodes)
            for node_id in removed:
                self.cluster.stop_node(node_id)
        except Exception as exc:
            records.append(self._error_record(event, "E_HANDSHAKE", f"post-commit quota propagation warning: {exc}"))
        return {"seq": event.seq, "outcome": "quota_replaced", "revoked": len(updates), "added": added, "removed": removed, "digest": digest}
