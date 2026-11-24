from __future__ import annotations

from pathlib import Path
from typing import Any

from .adjudication import choose_authority, plan_mandatory_evictions, victim_sort_key
from .cluster import NodeCluster
from .constants import (
    E_CAPACITY_EXCEEDED,
    E_EVENT_INVALID,
    E_PROTOCOL_VIOLATION,
    ERROR_CODE_ORDER,
    WATERMARK_RULE_METADATA,
)
from .hashing import derive_cache_key, fingerprint
from .watermark import build_watermark, watermark_state


class Workbench:
    def __init__(
        self,
        validation: dict[str, Any],
        *,
        parallel: bool = False,
        log_dir: str | Path | None = None,
        snapshot_store: Any | None = None,
        protocol_self_check: bool = False,
    ) -> None:
        self.validation = validation
        self.parallel = parallel
        self.snapshot_store = snapshot_store
        self.protocol_self_check_requested = protocol_self_check
        self.nodes_cfg = validation["nodes"]
        self.policies = validation["capacity_policies"]
        self.manifest = validation.get("manifest") or {"stages": [], "stage_index": {}}
        self.state = self._fresh_state()
        self.cluster = NodeCluster(
            self.nodes_cfg,
            self.policies,
            clock_mode=validation.get("clock_mode", "seeded"),
            log_dir=log_dir,
        )
        self.reconcile_caches(0)

    def _fresh_state(self) -> dict[str, Any]:
        code_counts = {code: 0 for code in ERROR_CODE_ORDER}
        failures = [dict(item) for item in self.validation.get("failures", [])]
        for failure in failures:
            code = failure.get("code", E_EVENT_INVALID)
            code_counts[code] = code_counts.get(code, 0) + 1

        node_versions: dict[str, dict[str, Any]] = {}
        residents: dict[str, dict[str, Any]] = {}
        for node_id, node in sorted(self.nodes_cfg.items()):
            node_versions[node_id] = {
                "quota_version": int(node.get("quota_version", 1)),
                "capacity_version": int(node["capacity_version"]),
                "lease_limit": int(node["lease_limit"]),
                "capacity_slots": int(node["capacity_slots"]),
            }
            residents[node_id] = {}

        budget_stages: dict[str, dict[str, Any]] = {}
        for stage in self.manifest.get("stages", []):
            budget_stages[stage["stage_id"]] = {
                "stage_id": stage["stage_id"],
                "artifact_key": stage["artifact_key"],
                "concurrency_quota": int(stage["concurrency_quota"]),
                "active_units": 0,
                "reserved_units": 0,
                "granted_total": 0,
            }

        return {
            "schema_version": "workbench.state.v1",
            "rules": WATERMARK_RULE_METADATA,
            "config_hash": self.validation["config_hash"],
            "cursor": 0,
            "processed_event_count": 0,
            "last_now_ms": None,
            "node_versions": node_versions,
            "residents": residents,
            "replicas": {},
            "authority": {},
            "authority_history": [],
            "overwrites": [],
            "applications": {},
            "pending_decisions": {},
            "decision_history": [],
            "plans": {},
            "open_plans": {},
            "reconciliations": [],
            "evictions": [],
            "manual_releases": [],
            "read_results": [],
            "renewals": [],
            "quota_changes": [],
            "budget": {"stages": budget_stages, "ledger": [], "deduction_sequence": []},
            "watermarks": {},
            "node_summaries": {},
            "cache_entries": {},
            "cache_products": {},
            "cache_ledger": [],
            "protocol_violations": [],
            "failure_list": failures,
            "code_counts": code_counts,
        }

    def __enter__(self) -> "Workbench":
        self.cluster.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def close(self) -> None:
        self.cluster.close()

    def run(
        self,
        *,
        max_events: int | None = None,
        stop_after_seq: int | None = None,
        close_cluster: bool = True,
        resume: bool = False,
    ) -> dict[str, Any]:
        self.cluster.start()
        try:
            start_index = int(self.state["processed_event_count"]) if resume else 0
            completed = start_index
            events = self.validation.get("events", [])[start_index:]
            processed_this_call = 0
            for event in events:
                if max_events is not None and processed_this_call >= max_events:
                    break
                if stop_after_seq is not None and int(event["seq"]) > stop_after_seq:
                    break
                self.dispatch(event)
                processed_this_call += 1
                completed += 1
                self.state["processed_event_count"] = completed
                self.state["cursor"] = int(event["seq"])
                self.reconcile_caches(int(event["seq"]))
                if self.snapshot_store is not None:
                    self.snapshot_store.commit(self)
            if self.snapshot_store is not None:
                self.snapshot_store.commit(self, final=True)
            return self.state
        finally:
            if close_cluster:
                self.close()

    def dispatch(self, event: dict[str, Any]) -> None:
        handlers = {
            "APPLY": self._on_apply,
            "WRITE": self._on_write,
            "READ": self._on_read,
            "RELEASE": self._on_release,
            "LEASE_RENEW": self._on_lease_renew,
            "TICK": self._on_tick,
            "QUOTA_CHANGED": self._on_quota_changed,
            "POLICY_CHANGED": self._on_policy_changed,
            "RUN_EVICTION": self._on_run_eviction,
        }
        handler = handlers.get(event["type"])
        if handler is None:
            self._add_failure({
                "code": E_EVENT_INVALID,
                "seq": event.get("seq"),
                "subject": "dispatch",
                "message": f"unsupported event type {event.get('type')!r}",
            })
            return

        now_ms = event.get("now_ms")
        if now_ms is not None:
            if self.state["last_now_ms"] is not None and now_ms < self.state["last_now_ms"]:
                self._add_failure({
                    "code": E_EVENT_INVALID,
                    "seq": event.get("seq"),
                    "subject": f"event#{event.get('seq')}:clock",
                    "message": "now_ms is earlier than the previously observed monotonic time",
                    "last_now_ms": self.state["last_now_ms"],
                    "now_ms": now_ms,
                })
                return
            self.state["last_now_ms"] = now_ms
        handler(event)

    def _add_failure(self, failure: dict[str, Any]) -> None:
        code = failure.get("code", E_EVENT_INVALID)
        self.state["failure_list"].append(failure)
        self.state["code_counts"][code] = self.state["code_counts"].get(code, 0) + 1

    def _record_node_error(self, event: dict[str, Any], node_id: str, response: dict[str, Any]) -> None:
        error = response.get("error") or {}
        code = error.get("code") or E_PROTOCOL_VIOLATION
        self._add_failure({
            "code": code,
            "seq": event.get("seq"),
            "event_type": event.get("type"),
            "node_id": node_id,
            "subject": f"event#{event.get('seq')}:{event.get('type')}:{node_id}",
            "message": error.get("message", "node request failed"),
            "detail": {k: v for k, v in error.items() if k not in {"code", "message"}},
        })

    def _request_id(self, seq: int, node_id: str, kind: str, suffix: str = "") -> str:
        safe_suffix = f":{suffix}" if suffix else ""
        return f"{kind}-seq{seq:05d}-{node_id}{safe_suffix}"

    def _target_nodes(self, event: dict[str, Any]) -> list[str]:
        if event.get("node_ids"):
            return sorted(event["node_ids"])
        return [event["node_id"]]

    # ---------- event handlers ----------

    def _on_apply(self, event: dict[str, Any]) -> None:
        app_id = event.get("application_id") or f"app-seq{event['seq']:05d}"
        if app_id in self.state["applications"]:
            self._add_failure({
                "code": E_EVENT_INVALID,
                "seq": event["seq"],
                "subject": f"APPLY:{app_id}",
                "message": "duplicate application_id",
            })
            return
        node_id = event["node_id"]
        self.state["applications"][app_id] = {
            "application_id": app_id,
            "arrival_seq": event["seq"],
            "key": event["key"],
            "node_id": node_id,
            "slots": int(event.get("slots", 1)),
            "priority": int(event.get("priority", 0)),
            "ttl_ms": int(event["ttl_ms"]),
            "stage_id": event.get("stage_id"),
            "content": event.get("content"),
            "digest": event.get("digest"),
            "now_ms": int(event["now_ms"]),
            "status": "pending",
            "decision_id": None,
            "admitted_seq": None,
        }

    def _on_write(self, event: dict[str, Any]) -> None:
        node_ids = self._target_nodes(event)
        requests = []
        for node_id in node_ids:
            payload = {
                "request_id": self._request_id(event["seq"], node_id, "write", event["key"]),
                "type": "write",
                "key": event["key"],
                "slots": int(event.get("slots", 1)),
                "priority": int(event.get("priority", 0)),
                "ttl_ms": int(event["ttl_ms"]),
                "now_ms": int(event.get("now_ms", self.state["last_now_ms"] or 0)),
                "stage_id": event.get("stage_id"),
            }
            if "content" in event:
                payload["content"] = event["content"]
            if "digest" in event:
                payload["digest"] = event["digest"]
            requests.append((node_id, payload))

        responses = self.cluster.request_many(requests, parallel=self.parallel)
        for node_id in node_ids:
            response = responses[node_id]
            if not response.get("ok"):
                self._record_node_error(event, node_id, response)
                continue
            self._apply_write_success(event, node_id, response)

    def _apply_write_success(
        self,
        event: dict[str, Any],
        node_id: str,
        response: dict[str, Any],
        *,
        application_id: str | None = None,
    ) -> dict[str, Any]:
        result = response.get("result") or {}
        key = event["key"]
        old = self.state["residents"][node_id].get(key)
        overwritten = bool(result.get("overwritten", old is not None))
        resident = {
            "node_id": node_id,
            "key": key,
            "slots": int(result.get("slots", event.get("slots", 1))),
            "priority": int(event.get("priority", 0)),
            "stage_id": event.get("stage_id"),
            "content_digest": response.get("content_digest"),
            "lease_id": result.get("lease_id"),
            "issued_ms": int(result.get("issued_ms", 0)),
            "expires_ms": int(result.get("expires_ms", 0)),
            "capacity_version": int(response.get("capacity_version")),
            "quota_version": int(response.get("quota_version")),
            "last_write_seq": int(event["seq"]),
            "budget_deducted": bool(old.get("budget_deducted")) if old else False,
            "application_id": application_id,
        }
        self.state["residents"][node_id][key] = resident

        if overwritten:
            self.state["overwrites"].append({
                "seq": int(event["seq"]),
                "node_id": node_id,
                "artifact_key": key,
                "overwritten": True,
                "overwritten_digest": result.get("overwritten_digest"),
                "new_digest": response.get("content_digest"),
                "lease_id": resident["lease_id"],
            })
        elif event.get("stage_id"):
            self._deduct_budget(
                seq=int(event["seq"]),
                stage_id=event["stage_id"],
                key=key,
                node_id=node_id,
                reason="lease_granted",
                application_id=application_id,
            )
            resident["budget_deducted"] = True

        candidate = {
            "key": key,
            "node_id": node_id,
            "request_seq": int(event["seq"]),
            "capacity_version": int(response.get("capacity_version")),
            "elapsed_ms": int(response.get("elapsed_ms", 0)),
            "content_digest": response.get("content_digest"),
            "quota_version": int(response.get("quota_version")),
            "clock_source": response.get("clock_source"),
        }
        self.state["replicas"].setdefault(key, {})[node_id] = candidate
        decision = choose_authority(list(self.state["replicas"][key].values()))
        if decision is not None:
            decision["overwritten_references"] = [
                item for item in self.state["overwrites"] if item["artifact_key"] == key
            ]
            self.state["authority"][key] = decision
            self.state["authority_history"].append({
                "seq": int(event["seq"]),
                "artifact_key": key,
                "winner_node_id": decision["winner_node_id"],
                "cache_input_fingerprint": fingerprint(list(self.state["replicas"][key].values())),
            })

        if application_id and application_id in self.state["applications"]:
            app = self.state["applications"][application_id]
            app["status"] = "admitted"
            app["admitted_seq"] = int(event["seq"])
            stage_id = app.get("stage_id")
            if stage_id and stage_id in self.state["budget"]["stages"]:
                budget = self.state["budget"]["stages"][stage_id]
                if budget["reserved_units"] > 0:
                    budget["reserved_units"] -= 1
        return resident

    def _deduct_budget(
        self,
        *,
        seq: int,
        stage_id: str,
        key: str,
        node_id: str,
        reason: str,
        application_id: str | None = None,
    ) -> None:
        budget = self.state["budget"]["stages"].get(stage_id)
        if budget is None:
            return
        budget["active_units"] += 1
        budget["granted_total"] += 1
        entry = {
            "seq": seq,
            "stage_id": stage_id,
            "artifact_key": key,
            "node_id": node_id,
            "delta": -1,
            "kind": "deduct",
            "reason": reason,
            "application_id": application_id,
            "active_units_after": budget["active_units"],
            "available_after": budget["concurrency_quota"] - budget["active_units"] - budget["reserved_units"],
        }
        self.state["budget"]["ledger"].append(entry)
        self.state["budget"]["deduction_sequence"].append({
            "seq": seq,
            "stage_id": stage_id,
            "artifact_key": key,
            "node_id": node_id,
            "application_id": application_id,
            "delta": -1,
            "reason": reason,
        })

    def _refund_budget(self, *, seq: int, resident: dict[str, Any], reason: str) -> None:
        stage_id = resident.get("stage_id")
        if not stage_id or not resident.get("budget_deducted"):
            return
        budget = self.state["budget"]["stages"].get(stage_id)
        if budget is None:
            return
        budget["active_units"] = max(0, budget["active_units"] - 1)
        resident["budget_deducted"] = False
        self.state["budget"]["ledger"].append({
            "seq": seq,
            "stage_id": stage_id,
            "artifact_key": resident["key"],
            "node_id": resident["node_id"],
            "delta": 1,
            "kind": "refund",
            "reason": reason,
            "lease_id": resident.get("lease_id"),
            "active_units_after": budget["active_units"],
            "available_after": budget["concurrency_quota"] - budget["active_units"] - budget["reserved_units"],
        })

    def _on_read(self, event: dict[str, Any]) -> None:
        node_id = event["node_id"]
        payload = {
            "request_id": self._request_id(event["seq"], node_id, "read", event["key"]),
            "type": "read",
            "key": event["key"],
        }
        if event.get("now_ms") is not None:
            payload["now_ms"] = event["now_ms"]
        if event.get("renew_ttl_ms") is not None:
            payload["renew_ttl_ms"] = event["renew_ttl_ms"]
        response = self.cluster.request(node_id, payload)
        if not response.get("ok"):
            self._record_node_error(event, node_id, response)
            return
        result = response.get("result") or {}
        record = {
            "seq": event["seq"],
            "node_id": node_id,
            "key": event["key"],
            "found": bool(response.get("found", result.get("found"))),
            "content_digest": response.get("content_digest"),
            "renewed": bool(result.get("renewed", False)),
            "renewal_reason": result.get("renewal_reason"),
            "expires_ms": result.get("expires_ms"),
        }
        self.state["read_results"].append(record)
        if record["renewed"]:
            resident = self.state["residents"][node_id].get(event["key"])
            if resident is not None:
                resident["expires_ms"] = int(result["expires_ms"])
            self.state["renewals"].append({
                "seq": event["seq"],
                "node_id": node_id,
                "key": event["key"],
                "renewed": True,
                "expires_ms": record["expires_ms"],
            })
        elif event.get("renew_ttl_ms") is not None:
            self.state["renewals"].append({
                "seq": event["seq"],
                "node_id": node_id,
                "key": event["key"],
                "renewed": False,
                "reason": result.get("renewal_reason") or "not_renewed",
            })

    def _on_release(self, event: dict[str, Any]) -> None:
        node_id = event["node_id"]
        payload: dict[str, Any] = {
            "request_id": self._request_id(event["seq"], node_id, "release", event.get("key") or event.get("lease_id", "")),
            "type": "release",
        }
        if event.get("key"):
            payload["key"] = event["key"]
        if event.get("lease_id"):
            payload["lease_id"] = event["lease_id"]
        if event.get("now_ms") is not None:
            payload["now_ms"] = event["now_ms"]
        response = self.cluster.request(node_id, payload)
        if not response.get("ok"):
            self._record_node_error(event, node_id, response)
            return
        key = event.get("key") or (response.get("result") or {}).get("key")
        resident = self._remove_resident(node_id, key, event["seq"], "manual_release") if key else None
        self.state["manual_releases"].append({
            "seq": event["seq"],
            "node_id": node_id,
            "key": key,
            "released": bool(response.get("released", True)),
            "released_slots": (response.get("result") or {}).get("released_slots"),
            "budget_refunded": bool(resident),
        })

    def _remove_resident(self, node_id: str, key: str, seq: int, reason: str) -> dict[str, Any] | None:
        resident = self.state["residents"][node_id].pop(key, None)
        if resident is None:
            return None
        self._refund_budget(seq=seq, resident=resident, reason=reason)
        replicas = self.state["replicas"].get(key)
        if replicas is not None:
            replicas.pop(node_id, None)
            if not replicas:
                self.state["replicas"].pop(key, None)
                self.state["authority"].pop(key, None)
        return resident

    def _on_lease_renew(self, event: dict[str, Any]) -> None:
        read_event = dict(event)
        read_event["type"] = "READ"
        read_event["renew_ttl_ms"] = event["ttl_ms"]
        self._on_read(read_event)

    def _on_tick(self, event: dict[str, Any]) -> None:
        now_ms = int(event["now_ms"])
        node_filter = event.get("node_id")
        for node_id in sorted(self.state["residents"]):
            if node_filter and node_id != node_filter:
                continue
            expired = [
                record
                for record in self.state["residents"][node_id].values()
                if int(record["expires_ms"]) <= now_ms
            ]
            for rank, record in enumerate(sorted(expired, key=victim_sort_key), start=1):
                payload = {
                    "request_id": self._request_id(event["seq"], node_id, "expire", record["key"]),
                    "type": "release",
                    "key": record["key"],
                    "lease_id": record["lease_id"],
                    "now_ms": now_ms,
                }
                response = self.cluster.request(node_id, payload)
                if not response.get("ok"):
                    self._record_node_error(event, node_id, response)
                    continue
                result = response.get("result") or {}
                removed = self._remove_resident(node_id, record["key"], event["seq"], "lease_expired")
                self.state["evictions"].append({
                    "seq": event["seq"],
                    "plan_id": None,
                    "node_id": node_id,
                    "key": record["key"],
                    "lease_id": record["lease_id"],
                    "reason": "lease_expired",
                    "selection_rank_within_node": rank,
                    "selection_order": list(victim_sort_key(record)),
                    "selection_basis": WATERMARK_RULE_METADATA["victim_order"],
                    "released_slots": result.get("released_slots", record["slots"]),
                    "released_digest": result.get("released_digest", record["content_digest"]),
                    "budget_refunded": removed is not None and not removed.get("budget_deducted", True),
                    "now_ms": now_ms,
                })

    def _on_quota_changed(self, event: dict[str, Any]) -> None:
        node_ids = self._target_nodes(event)
        for node_id in node_ids:
            payload: dict[str, Any] = {
                "request_id": self._request_id(event["seq"], node_id, "quota"),
                "type": "quota_change",
                "quota_version": int(event["quota_version"]),
            }
            if event.get("lease_limit") is not None:
                payload["lease_limit"] = int(event["lease_limit"])
            if event.get("now_ms") is not None:
                payload["now_ms"] = event["now_ms"]
            response = self.cluster.request(node_id, payload)
            if not response.get("ok"):
                self._record_node_error(event, node_id, response)
                continue
            self.state["node_versions"][node_id]["quota_version"] = int(response["quota_version"])
            if event.get("lease_limit") is not None:
                self.state["node_versions"][node_id]["lease_limit"] = int(event["lease_limit"])
            self.state["quota_changes"].append({
                "seq": event["seq"],
                "node_id": node_id,
                "quota_version": int(response["quota_version"]),
                "lease_limit": event.get("lease_limit"),
            })

        quotas = event.get("concurrency_quotas") or {}
        for stage_id, quota in sorted(quotas.items()):
            budget = self.state["budget"]["stages"].get(stage_id)
            if budget is not None:
                budget["concurrency_quota"] = int(quota)

    def _on_policy_changed(self, event: dict[str, Any]) -> None:
        node_ids = self._target_nodes(event)
        for node_id in node_ids:
            if node_id not in self.state["node_versions"]:
                self._add_failure({
                    "code": E_EVENT_INVALID,
                    "seq": event["seq"],
                    "subject": f"POLICY_CHANGED:{node_id}",
                    "message": "unknown node",
                })
                continue
            current = self.state["node_versions"][node_id]
            new_capacity_version = event.get("capacity_version", current["capacity_version"])
            if event.get("capacity_version") is not None and new_capacity_version <= current["capacity_version"]:
                self._add_failure({
                    "code": E_EVENT_INVALID,
                    "seq": event["seq"],
                    "node_id": node_id,
                    "subject": f"POLICY_CHANGED:{node_id}",
                    "message": "capacity_version must be strictly increasing",
                    "current_capacity_version": current["capacity_version"],
                    "requested_capacity_version": new_capacity_version,
                })
                continue

            payload: dict[str, Any] = {
                "request_id": self._request_id(event["seq"], node_id, "policy"),
                "type": "quota_change",
                "now_ms": event.get("now_ms", self.state["last_now_ms"]),
            }
            if event.get("capacity_version") is not None:
                payload["capacity_version"] = int(event["capacity_version"])
            if event.get("lease_limit") is not None:
                payload["lease_limit"] = int(event["lease_limit"])
            if event.get("quota_version") is not None:
                payload["quota_version"] = int(event["quota_version"])
            response = self.cluster.request(node_id, payload)
            if not response.get("ok"):
                self._record_node_error(event, node_id, response)
                continue

            current["capacity_version"] = int(response["capacity_version"])
            current["quota_version"] = int(response["quota_version"])
            result = response.get("result") or {}
            if "after" in result:
                current["lease_limit"] = int(result["after"]["lease_limit"])
            elif event.get("lease_limit") is not None:
                current["lease_limit"] = int(event["lease_limit"])

            old_plan_id = self.state["open_plans"].get(node_id)
            if old_plan_id:
                self._rollback_reserved_plan(old_plan_id)

            watermark = self._watermark_for(node_id)
            plan_core = plan_mandatory_evictions(
                self.state["residents"][node_id],
                lease_limit=current["lease_limit"],
                hard_limit=watermark.hard_limit,
            )
            plan_id = f"plan-seq{event['seq']:05d}-{node_id}"
            plan = {
                "plan_id": plan_id,
                "trigger_seq": event["seq"],
                "node_id": node_id,
                "status": "open",
                "capacity_version": current["capacity_version"],
                "quota_version": current["quota_version"],
                "lease_limit": current["lease_limit"],
                "soft_limit": watermark.soft_limit,
                "hard_limit": watermark.hard_limit,
                **plan_core,
                "decisions": [],
                "reconciliation": None,
            }
            self.state["plans"][plan_id] = plan
            self.state["open_plans"][node_id] = plan_id
            self._adjudicate_pending(plan)

    def _watermark_for(self, node_id: str):
        versions = self.state["node_versions"][node_id]
        policy = self.policies[int(versions["capacity_version"])]
        return build_watermark(
            capacity_slots=int(versions["capacity_slots"]),
            capacity_version=int(versions["capacity_version"]),
            soft_permille=int(policy["soft_permille"]),
            hard_permille=int(policy["hard_permille"]),
        )

    def _rollback_reserved_plan(self, plan_id: str) -> None:
        plan = self.state["plans"].get(plan_id)
        if not plan or plan.get("status") != "open":
            return
        for decision in plan.get("decisions", []):
            if decision.get("decision") != "reserved":
                continue
            app = self.state["applications"].get(decision["application_id"])
            if app and app["status"] == "reserved":
                app["status"] = "pending"
                app["decision_id"] = None
                budget = self.state["budget"]["stages"].get(app.get("stage_id") or "")
                if budget:
                    budget["reserved_units"] = max(0, budget["reserved_units"] - 1)
        plan["status"] = "superseded"

    def _adjudicate_pending(self, plan: dict[str, Any]) -> None:
        node_id = plan["node_id"]
        candidates = [
            app for app in self.state["applications"].values()
            if app["node_id"] == node_id and app["status"] == "pending"
        ]
        candidates.sort(key=lambda app: (app["key"], app["arrival_seq"], app["application_id"]))

        sim_occupancy = int(plan["post_eviction_occupancy"])
        sim_leases = int(plan["post_eviction_active_leases"])
        sim_stage_used = {
            stage_id: int(budget["active_units"]) + int(budget["reserved_units"])
            for stage_id, budget in self.state["budget"]["stages"].items()
        }
        for victim in plan["victims"]:
            resident = self.state["residents"][node_id].get(victim["key"])
            stage_id = victim.get("stage_id") or (resident or {}).get("stage_id")
            if stage_id and resident and resident.get("budget_deducted"):
                sim_stage_used[stage_id] = max(0, sim_stage_used.get(stage_id, 0) - 1)

        for app in candidates:
            new_lease = app["key"] not in self.state["residents"][node_id]
            stage_id = app.get("stage_id")
            stage_quota = None
            if stage_id:
                budget = self.state["budget"]["stages"].get(stage_id)
                stage_quota = budget["concurrency_quota"] if budget else None

            rejection_reason = None
            if sim_occupancy + app["slots"] > plan["hard_limit"]:
                rejection_reason = "hard_watermark_would_be_exceeded"
            elif sim_occupancy + app["slots"] > self.state["node_versions"][node_id]["capacity_slots"]:
                rejection_reason = "physical_capacity_exceeded"
            elif sim_leases + (1 if new_lease else 0) > plan["lease_limit"]:
                rejection_reason = "lease_limit_would_be_exceeded"
            elif stage_quota is not None and sim_stage_used.get(stage_id, 0) + 1 > stage_quota:
                rejection_reason = "stage_concurrency_quota_exceeded"

            decision_id = f"decision-{plan['trigger_seq']:05d}-{app['application_id']}"
            decision = {
                "decision_id": decision_id,
                "plan_id": plan["plan_id"],
                "trigger_seq": plan["trigger_seq"],
                "application_id": app["application_id"],
                "artifact_key": app["key"],
                "node_id": node_id,
                "decision": "rejected" if rejection_reason else "reserved",
                "reason": rejection_reason or "fits_after_planned_evictions",
                "slots": app["slots"],
                "capacity_version": plan["capacity_version"],
                "quota_version": plan["quota_version"],
                "lease_limit": plan["lease_limit"],
                "hard_limit": plan["hard_limit"],
                "planned_eviction_ids": [item["lease_id"] for item in plan["victims"]],
                "planned_eviction_keys": list(plan["planned_victim_keys"]),
                "assumed_eviction_sequence": list(plan["planned_victim_keys"]),
                "simulated_occupancy_after_decision": sim_occupancy + app["slots"] if not rejection_reason else sim_occupancy,
                "simulated_active_leases_after_decision": sim_leases + (1 if new_lease else 0) if not rejection_reason else sim_leases,
                "admitted": False,
            }
            plan["decisions"].append(decision)
            self.state["decision_history"].append(decision)
            self.state["pending_decisions"][app["application_id"]] = decision
            app["decision_id"] = decision_id

            if rejection_reason:
                app["status"] = "rejected"
                continue

            app["status"] = "reserved"
            sim_occupancy += app["slots"]
            if new_lease:
                sim_leases += 1
            if stage_id:
                sim_stage_used[stage_id] = sim_stage_used.get(stage_id, 0) + 1
                budget = self.state["budget"]["stages"].get(stage_id)
                if budget:
                    budget["reserved_units"] += 1

    def _on_run_eviction(self, event: dict[str, Any]) -> None:
        node_id = event["node_id"]
        plan_id = self.state["open_plans"].get(node_id)
        if plan_id is None:
            watermark = self._watermark_for(node_id)
            plan = {
                "plan_id": f"adhoc-plan-seq{event['seq']:05d}-{node_id}",
                "trigger_seq": event["seq"],
                "node_id": node_id,
                "status": "open",
                "capacity_version": self.state["node_versions"][node_id]["capacity_version"],
                "quota_version": self.state["node_versions"][node_id]["quota_version"],
                "lease_limit": self.state["node_versions"][node_id]["lease_limit"],
                "soft_limit": watermark.soft_limit,
                "hard_limit": watermark.hard_limit,
                **plan_mandatory_evictions(
                    self.state["residents"][node_id],
                    lease_limit=self.state["node_versions"][node_id]["lease_limit"],
                    hard_limit=watermark.hard_limit,
                ),
                "decisions": [],
                "reconciliation": None,
            }
            self.state["plans"][plan["plan_id"]] = plan
        else:
            plan = self.state["plans"][plan_id]

        actual_keys: list[str] = []
        for victim in plan.get("victims", []):
            key = victim["key"]
            payload = {
                "request_id": self._request_id(event["seq"], node_id, "evict", key),
                "type": "release",
                "key": key,
                "lease_id": victim.get("lease_id"),
            }
            if event.get("now_ms") is not None:
                payload["now_ms"] = event["now_ms"]
            response = self.cluster.request(node_id, payload)
            if not response.get("ok"):
                self._record_node_error(event, node_id, response)
                continue
            result = response.get("result") or {}
            removed = self._remove_resident(node_id, key, event["seq"], "policy_eviction")
            released = bool(response.get("released", result.get("released", removed is not None)))
            if released:
                actual_keys.append(key)
            self.state["evictions"].append({
                "seq": event["seq"],
                "plan_id": plan["plan_id"],
                "node_id": node_id,
                "key": key,
                "lease_id": victim.get("lease_id"),
                "reason": "policy_eviction",
                "selection_order": victim.get("victim_rank"),
                "selection_basis": WATERMARK_RULE_METADATA["victim_order"],
                "selected_for": victim.get("selected_for"),
                "released_slots": result.get("released_slots", victim.get("slots")),
                "released_digest": result.get("released_digest", victim.get("content_digest")),
                "actual_released": released,
            })

        planned_keys = list(plan.get("planned_victim_keys", []))
        matched = actual_keys == planned_keys
        reconciliation = {
            "seq": event["seq"],
            "plan_id": plan["plan_id"],
            "node_id": node_id,
            "planned_eviction_keys": planned_keys,
            "actual_eviction_keys": actual_keys,
            "matched": matched,
            "reserved_admitted": [],
            "admission_failures": [],
        }

        for decision in plan.get("decisions", []):
            if decision.get("decision") != "reserved":
                continue
            app = self.state["applications"].get(decision["application_id"])
            if app is None or app["status"] != "reserved":
                continue
            admission_now_ms = int(event.get("now_ms") or self.state["last_now_ms"] or app["now_ms"])
            write_event = {
                "seq": event["seq"],
                "type": "WRITE",
                "key": app["key"],
                "slots": app["slots"],
                "priority": app["priority"],
                "ttl_ms": app["ttl_ms"],
                "now_ms": admission_now_ms,
                "stage_id": app.get("stage_id"),
                "content": app.get("content"),
                "digest": app.get("digest"),
            }
            response = self.cluster.request(node_id, {
                "request_id": self._request_id(event["seq"], node_id, "admit", app["application_id"]),
                "type": "write",
                "key": app["key"],
                "slots": app["slots"],
                "priority": app["priority"],
                "ttl_ms": app["ttl_ms"],
                "now_ms": admission_now_ms,
                "stage_id": app.get("stage_id"),
                **({"content": app["content"]} if app.get("content") is not None else {}),
                **({"digest": app["digest"]} if app.get("digest") is not None else {}),
            })
            if not response.get("ok"):
                self._record_node_error({"seq": event["seq"], "type": "RESERVED_ADMIT", "node_id": node_id}, node_id, response)
                decision["admitted"] = False
                decision["admission_error"] = (response.get("error") or {}).get("code")
                reconciliation["admission_failures"].append({
                    "application_id": app["application_id"],
                    "error": response.get("error"),
                })
                continue
            self._apply_write_success(
                write_event,
                node_id,
                response,
                application_id=app["application_id"],
            )
            decision["admitted"] = True
            reconciliation["reserved_admitted"].append(app["application_id"])

        plan["status"] = "executed"
        plan["reconciliation"] = reconciliation
        self.state["reconciliations"].append(reconciliation)
        self.state["open_plans"].pop(node_id, None)

    # ---------- cache / fingerprints ----------

    def _cache_lookup(
        self,
        *,
        stage: str,
        node_id: str,
        artifact_key: str,
        scope_id: str,
        upstream_fingerprint: str,
        quota_version: Any,
        capacity_version: Any,
        compute,
        seq: int,
    ) -> Any:
        identity = fingerprint([stage, node_id, artifact_key, scope_id])
        cache_key = derive_cache_key(
            stage=stage,
            upstream_fingerprint=upstream_fingerprint,
            quota_version=quota_version,
            capacity_version=capacity_version,
        )
        entry = self.state["cache_entries"].get(identity)
        if entry is not None and entry["cache_key"] == cache_key:
            entry["status"] = "hit"
            entry["last_result"] = "reused"
            entry["hit_count"] += 1
            entry["last_evaluated_seq"] = seq
            return self.state["cache_products"][identity]

        product = compute()
        recompute_count = int(entry["recompute_count"]) + 1 if entry else 1
        hit_count = int(entry["hit_count"]) if entry else 0
        self.state["cache_entries"][identity] = {
            "identity": identity,
            "stage": stage,
            "scope_id": scope_id,
            "node_id": node_id,
            "artifact_key": artifact_key,
            "cache_key": cache_key,
            "upstream_fingerprint": upstream_fingerprint,
            "quota_version": quota_version,
            "capacity_version": capacity_version,
            "status": "recomputed",
            "last_result": "recomputed",
            "recompute_count": recompute_count,
            "hit_count": hit_count,
            "last_evaluated_seq": seq,
        }
        self.state["cache_products"][identity] = product
        return product

    def reconcile_caches(self, seq: int) -> None:
        ledger_rows: list[dict[str, Any]] = []
        for node_id in sorted(self.state["residents"]):
            versions = self.state["node_versions"][node_id]
            residents = self.state["residents"][node_id]
            resident_fingerprint = fingerprint({
                key: {
                    "slots": item["slots"],
                    "priority": item["priority"],
                    "expires_ms": item["expires_ms"],
                    "content_digest": item["content_digest"],
                    "lease_id": item["lease_id"],
                    "capacity_version": item["capacity_version"],
                    "quota_version": item["quota_version"],
                }
                for key, item in sorted(residents.items())
            })
            summary = self._cache_lookup(
                stage="resident_summary",
                node_id=node_id,
                artifact_key="",
                scope_id="resident_summary",
                upstream_fingerprint=resident_fingerprint,
                quota_version=versions["quota_version"],
                capacity_version=versions["capacity_version"],
                seq=seq,
                compute=lambda nid=node_id, fp=resident_fingerprint: {
                    "node_id": nid,
                    "resident_count": len(self.state["residents"][nid]),
                    "occupancy": sum(int(x["slots"]) for x in self.state["residents"][nid].values()),
                    "resident_fingerprint": fp,
                    "keys": sorted(self.state["residents"][nid]),
                },
            )
            self.state["node_summaries"][node_id] = summary

            policy = self.policies[int(versions["capacity_version"])]
            policy_fingerprint = fingerprint(policy)
            wm_upstream = fingerprint({"resident_fingerprint": resident_fingerprint, "policy": policy_fingerprint})

            def compute_watermark(nid: str = node_id) -> dict[str, Any]:
                wm = self._watermark_for(nid)
                occupancy = sum(int(x["slots"]) for x in self.state["residents"][nid].values())
                return {
                    **wm.as_dict(),
                    "occupancy": occupancy,
                    "watermark_state": watermark_state(occupancy, wm),
                }

            watermark_product = self._cache_lookup(
                stage="watermark",
                node_id=node_id,
                artifact_key="",
                scope_id="watermark",
                upstream_fingerprint=wm_upstream,
                quota_version=versions["quota_version"],
                capacity_version=versions["capacity_version"],
                seq=seq,
                compute=compute_watermark,
            )
            self.state["watermarks"][node_id] = watermark_product

            eviction_upstream = fingerprint({
                "resident_fingerprint": resident_fingerprint,
                "watermark": watermark_product,
                "lease_limit": versions["lease_limit"],
            })
            self._cache_lookup(
                stage="eviction_plan",
                node_id=node_id,
                artifact_key="",
                scope_id="current_mandatory_plan",
                upstream_fingerprint=eviction_upstream,
                quota_version=versions["quota_version"],
                capacity_version=versions["capacity_version"],
                seq=seq,
                compute=lambda nid=node_id: plan_mandatory_evictions(
                    self.state["residents"][nid],
                    lease_limit=self.state["node_versions"][nid]["lease_limit"],
                    hard_limit=self.state["watermarks"][nid]["hard_limit"],
                ),
            )

        for artifact_key in sorted(self.state["replicas"]):
            candidates = list(self.state["replicas"][artifact_key].values())
            upstream = fingerprint(candidates)
            node_versions = {
                item["node_id"]: {
                    "quota_version": item["quota_version"],
                    "capacity_version": item["capacity_version"],
                }
                for item in candidates
            }
            self._cache_lookup(
                stage="authority",
                node_id="",
                artifact_key=artifact_key,
                scope_id="authority",
                upstream_fingerprint=upstream,
                quota_version=fingerprint({nid: v["quota_version"] for nid, v in sorted(node_versions.items())}),
                capacity_version=fingerprint({nid: v["capacity_version"] for nid, v in sorted(node_versions.items())}),
                seq=seq,
                compute=lambda key=artifact_key, items=candidates: choose_authority(items),
            )

        for app_id in sorted(self.state["pending_decisions"]):
            decision = self.state["pending_decisions"][app_id]
            node_id = decision["node_id"]
            versions = self.state["node_versions"][node_id]
            upstream = fingerprint(decision)
            self._cache_lookup(
                stage="pending_decision",
                node_id=node_id,
                artifact_key=decision["artifact_key"],
                scope_id=app_id,
                upstream_fingerprint=upstream,
                quota_version=versions["quota_version"],
                capacity_version=versions["capacity_version"],
                seq=seq,
                compute=lambda item=decision: item,
            )

        for entry in self.state["cache_entries"].values():
            ledger_rows.append({k: v for k, v in entry.items() if k != "identity"})
        ledger_rows.sort(key=lambda item: (
            item.get("node_id") or "",
            item.get("artifact_key") or "",
            item.get("stage") or "",
            item.get("scope_id") or "",
        ))
        self.state["cache_ledger"] = ledger_rows

    def protocol_self_check(self) -> list[dict[str, Any]]:
        if not self.nodes_cfg:
            return []
        node_id = sorted(self.nodes_cfg)[0]
        cases = [
            ("malformed_json", b'{"request_id":"self-check-malformed","type":"write",'),
            ("unknown_type", b'{"request_id":"self-check-unknown","type":"not_allowed"}'),
        ]
        results = []
        for case_name, raw in cases:
            line = self.cluster.send_raw_line(node_id, raw)
            response = __import__("json").loads(line.decode("utf-8"))
            code = ((response.get("error") or {}).get("code")) if isinstance(response, dict) else None
            record = {
                "case": case_name,
                "node_id": node_id,
                "code": code or E_PROTOCOL_VIOLATION,
                "response": response,
            }
            results.append(record)
            self.state["protocol_violations"].append(record)
            self._add_failure({
                "code": record["code"],
                "subject": f"protocol-self-check:{case_name}",
                "node_id": node_id,
                "message": (response.get("error") or {}).get("message", "protocol violation"),
            })
        return results
