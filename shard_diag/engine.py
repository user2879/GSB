from __future__ import annotations

import concurrent.futures
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from .errors import DiagnosticError
from .graph import stable_topology
from .ledger import Ledger, now_monotonic_ns
from .protocol import dispatch_job
from .quota import candidate_water_decision, compute_water, effective_node_quota, resolve_quota
from .samples import compute_loss_summary, generate_rows, load_overrides
from .spec import SpecBundle, load_spec_bundle


class DiagnosticEngine:
    def __init__(
        self,
        root: str | Path,
        ledger: Ledger,
        bundle: SpecBundle | None = None,
        *,
        quota_version: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.ledger = ledger
        self.bundle = bundle or load_spec_bundle(self.root)
        self.topology = stable_topology(self.bundle.shards)
        self.quota_version = quota_version or max(item["version"] for item in self.bundle.quota_versions)
        self.resolved_quota = resolve_quota(self.bundle.quota_versions, self.quota_version)
        self.overrides = load_overrides(self.root)

    def reload_declarations(self) -> None:
        self.bundle = load_spec_bundle(self.root)
        self.topology = stable_topology(self.bundle.shards)
        self.quota_version = max(item["version"] for item in self.bundle.quota_versions)
        self.resolved_quota = resolve_quota(self.bundle.quota_versions, self.quota_version)
        self.overrides = load_overrides(self.root)

    def mark_invalidated(self, shard_ids: list[str] | set[str] | tuple[str, ...], reason: dict[str, Any]) -> dict[str, Any]:
        ids = sorted(set(shard_ids))
        if not ids:
            return {"invalidated": []}
        self.ledger.conn.execute("BEGIN IMMEDIATE")
        try:
            for shard_id in ids:
                self.ledger.conn.execute(
                    "UPDATE shard_outcomes SET status='invalidated' WHERE shard_id=?",
                    (shard_id,),
                )
                self.ledger.conn.execute(
                    """INSERT INTO shard_state(shard_id,status,current_lease_id,format_version,content_digest,updated_wall_ms)
                       VALUES(?,?,NULL,1,?,?)
                       ON CONFLICT(shard_id) DO UPDATE SET status='invalidated',current_lease_id=NULL,content_digest=excluded.content_digest,updated_wall_ms=excluded.updated_wall_ms""",
                    (
                        shard_id,
                        "invalidated",
                        __import__("shard_diag.canonical", fromlist=["content_digest"]).content_digest({"shard_id": shard_id, "invalidated": reason}),
                        __import__("shard_diag.ledger", fromlist=["now_wall_ms"]).now_wall_ms(),
                    ),
                )
            self.ledger._append_event("invalidation", None, {"shard_ids": ids, "reason": reason})
            self.ledger.conn.commit()
        except Exception:
            self.ledger.conn.rollback()
            raise
        return {"invalidated": ids, "reason": reason}

    def active_lease_shards(self) -> set[str]:
        rows = self.ledger.conn.execute("SELECT shard_id FROM leases WHERE status='active'").fetchall()
        return {row["shard_id"] for row in rows}

    def evaluate_readiness(self) -> dict[str, dict[str, Any]]:
        committed = set(self.ledger.committed_outcomes())
        state_rows = self.ledger.conn.execute("SELECT shard_id,status FROM shard_state").fetchall()
        states = {row["shard_id"]: row["status"] for row in state_rows}
        active_leases = self.active_lease_shards()
        known = set(self.bundle.shard_by_id)
        result: dict[str, dict[str, Any]] = {}
        cyclic = set(self.topology.cycles)

        for shard_id in self.topology.order:
            shard = self.bundle.shard_by_id[shard_id]
            missing = [up for up in shard["upstream"] if up not in known]
            known_upstreams = [up for up in shard["upstream"] if up in known]
            blocked = [
                up
                for up in known_upstreams
                if up not in committed or not self._outcome_has_digest(up)
            ]
            failed_upstreams = [
                up
                for up in blocked
                if states.get(up) in {"unknown", "failed", "frozen_reclaimed"}
            ]
            if shard_id in committed and self._outcome_has_digest(shard_id):
                status = "committed"
                reason = "loss_summary_committed"
            elif missing:
                status = "frozen_missing_dependency"
                reason = "missing_upstream"
            elif shard_id in cyclic:
                status = "frozen_cycle"
                reason = "dependency_cycle"
            elif failed_upstreams:
                status = "frozen_upstream_failure"
                reason = "upstream_outcome_not_committed"
            elif blocked:
                status = "waiting_upstream"
                reason = "upstream_not_ready"
            elif shard_id in active_leases or states.get(shard_id) == "running":
                status = "leased"
                reason = "lease_active"
            else:
                status = "ready"
                reason = "all_upstream_loss_digests_available"
            result[shard_id] = {
                "shard_id": shard_id,
                "status": status,
                "reason": reason,
                "missing_upstream": missing,
                "blocked_upstream": blocked if status.startswith("frozen") or status == "waiting_upstream" else [],
                "topology_index": self.topology.rank[shard_id],
                "has_active_lease": shard_id in active_leases,
                "ledger_state": states.get(shard_id),
                "rerun_required": states.get(shard_id) in {"unknown", "frozen_reclaimed", "invalidated"},
            }
        return result

    def _outcome_has_digest(self, shard_id: str) -> bool:
        row = self.ledger.conn.execute(
            "SELECT loss_digest FROM shard_outcomes WHERE shard_id=? AND status='committed'",
            (shard_id,),
        ).fetchone()
        return row is not None and bool(row["loss_digest"])

    def current_water(self) -> dict[str, Any]:
        usage = deepcopy(self.ledger.active_usage())
        for node in self.bundle.nodes:
            usage.setdefault(node["node_id"], {"resource_units": 0, "leases": 0, "executions": 0})
        active_rows = self.ledger.conn.execute(
            "SELECT l.node_id AS node_id, l.shard_id AS shard_id FROM leases l WHERE l.status='active'"
        ).fetchall()
        active_by_node: dict[str, list[str]] = {}
        batch_counts: dict[str, int] = {}
        for row in active_rows:
            active_by_node.setdefault(row["node_id"], []).append(row["shard_id"])
            shard = self.bundle.shard_by_id.get(row["shard_id"])
            if shard:
                batch_counts[row["shard_id"]] = int(shard.get("batch_count_hint", 1))
        return compute_water(
            self.bundle.nodes,
            self.resolved_quota,
            usage,
            batch_counts=batch_counts,
            active_shard_ids_by_node=active_by_node,
        )

    def candidate_decisions(self, readiness: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        water = self.current_water()
        decisions: list[dict[str, Any]] = []
        for shard_id in sorted(readiness):
            if readiness[shard_id]["status"] != "ready":
                continue
            shard = self.bundle.shard_by_id[shard_id]
            node = self.bundle.node_by_id[shard["node_id"]]
            decision = candidate_water_decision(
                shard, node, self.resolved_quota, water, self.topology.order
            )
            decision["priority"] = shard["priority"]
            decision["scheduling_order_key"] = [-int(shard["priority"]), decision["topology_rank"], shard_id]
            decisions.append(decision)
        affected = [d["shard_id"] for d in decisions if d["error_affected"]]
        if affected:
            from .quota import choose_error_tie

            winner = choose_error_tie(affected, self.bundle.shards)
            for decision in decisions:
                decision["error_tie_winner"] = winner
                decision["error_tie_selected"] = decision["shard_id"] == winner
        return sorted(decisions, key=lambda item: tuple(item["scheduling_order_key"]))

    def _execute_granted(
        self,
        lease: dict[str, Any],
        *,
        batch_size: int,
        timeout_ms: int | None,
        delay_ms: int = 0,
    ) -> dict[str, Any]:
        shard_id = lease["shard_id"]
        shard = self.bundle.shard_by_id[shard_id]
        node = self.bundle.node_by_id[lease["node_id"]]
        rows = generate_rows(node, shard, self.quota_version, self.overrides)
        effective_timeout = int(timeout_ms or max(1000, int(node["socket_timeout_ms"]) * 3))
        receipt = dispatch_job(
            host=node["host"],
            port=int(node["port"]),
            node_id=node["node_id"],
            shard=shard,
            quota_version=self.quota_version,
            rows=rows,
            lease_version=int(lease["lease_version"]),
            batch_size=max(1, int(batch_size)),
            timeout_ms=effective_timeout,
            delay_ms=delay_ms,
        )
        expected_summary = compute_loss_summary(node, shard_id, self.quota_version, rows)
        if receipt["loss_summary"].get("loss_digest") != expected_summary["loss_digest"]:
            raise DiagnosticError(
                "节点损失摘要与主进程独立复算不一致",
                entry=shard_id,
                field="loss_summary.loss_digest",
                details={"node": receipt["loss_summary"], "expected": expected_summary},
            )
        receipt["loss_summary"]["main_recomputed_digest"] = expected_summary["loss_digest"]
        return {"receipt": receipt, "lease": lease, "row_count": len(rows)}

    def _water_records_for_completion(self, result: dict[str, Any]) -> list[dict[str, Any]]:
        receipt = result["receipt"]
        lease = result["lease"]
        shard_id = receipt["shard_id"]
        node_id = receipt["node_id"]
        held_water = self.current_water()
        post_usage = deepcopy(self.ledger.active_usage())
        current = post_usage.setdefault(node_id, {"resource_units": 0, "leases": 0, "executions": 0})
        current["resource_units"] = max(0, int(current["resource_units"]) - int(lease["resource_units"]))
        current["leases"] = max(0, int(current["leases"]) - 1)
        current["executions"] = max(0, int(current["executions"]) - 1)
        post_water = compute_water(self.bundle.nodes, self.resolved_quota, post_usage)
        decision = next(
            (item for item in self.candidate_decisions(self.evaluate_readiness()) if item["shard_id"] == shard_id),
            None,
        )
        shard_record = {
            "record_key": f"shard:{shard_id}",
            "scope": "shard_completion_water",
            "shard_id": shard_id,
            "node_id": node_id,
            "quota_version": self.quota_version,
            "decision_at_grant": decision,
            "held_water": held_water,
        }
        global_record = {
            "record_key": "current:global",
            "scope": "global_water",
            "quota_version": self.quota_version,
            "water": post_water,
        }
        node_record = {
            "record_key": f"current:node:{node_id}",
            "scope": "node_water",
            "node_id": node_id,
            "quota_version": self.quota_version,
            "water": post_water["node_water"][node_id],
        }
        return [shard_record, global_record, node_record]

    def run_all(
        self,
        cluster: Any,
        *,
        batch_size: int = 4,
        lease_ttl_ms: int = 30000,
        timeout_ms: int | None = None,
        max_seconds: float = 60.0,
        delay_ms_by_shard: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        delay_ms_by_shard = delay_ms_by_shard or {}
        failures: dict[str, dict[str, Any]] = {}
        committed_now: list[str] = []
        start = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(self.bundle.shards) * 2)) as pool:
            futures: dict[concurrent.futures.Future[Any], str] = {}
            while time.monotonic() - start < max_seconds:
                self.ledger.reap_expired_and_generation(
                    now_ns=now_monotonic_ns(), current_quota_version=self.quota_version
                )
                readiness = self.evaluate_readiness()
                decisions = self.candidate_decisions(readiness)
                granted_any = False
                for decision in decisions:
                    shard_id = decision["shard_id"]
                    if shard_id in futures.values():
                        continue
                    shard = self.bundle.shard_by_id[shard_id]
                    node = self.bundle.node_by_id[shard["node_id"]]
                    limits = effective_node_quota(node, self.resolved_quota)
                    try:
                        lease = self.ledger.grant_lease(
                            shard=shard,
                            quota_version=self.quota_version,
                            ttl_ms=lease_ttl_ms,
                            limits=limits,
                            global_budget_units=int(self.resolved_quota["global_budget_units"]),
                        )
                    except Exception as exc:
                        failures[shard_id] = getattr(exc, "to_dict", lambda: {"code": "E_LEASE", "error": str(exc)})()
                        continue
                    future = pool.submit(
                        self._execute_granted,
                        lease,
                        batch_size=batch_size,
                        timeout_ms=timeout_ms,
                        delay_ms=int(delay_ms_by_shard.get(shard_id, 0)),
                    )
                    futures[future] = shard_id
                    granted_any = True

                if futures:
                    done, _ = concurrent.futures.wait(
                        futures, timeout=0.1, return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    for future in done:
                        shard_id = futures.pop(future)
                        try:
                            result = future.result()
                            records = self._water_records_for_completion(result)
                            before_count = len(self.ledger.committed_outcomes())
                            commit = self.ledger.commit_shard_completion(
                                receipt=result["receipt"],
                                lease=result["lease"],
                                water_records=records,
                                topology_index=self.topology.rank[shard_id],
                                completed_count=before_count + 1,
                                snapshot_version=before_count + 1,
                            )
                            snapshot_path = self.root / "ledger" / "session_snapshot.json"
                            self.ledger.save_snapshot_file(snapshot_path, snapshot_version=before_count + 1)
                            committed_now.append(shard_id)
                            failures.pop(shard_id, None)
                            _ = commit["content_digest"]
                        except Exception as exc:
                            error_dict = getattr(exc, "to_dict", None)
                            failures[shard_id] = error_dict() if callable(error_dict) else {"code": "E_RUN", "error": str(exc)}
                    continue

                readiness = self.evaluate_readiness()
                ready = [item for item in readiness.values() if item["status"] == "ready"]
                if not ready and not granted_any:
                    break
                if not granted_any:
                    time.sleep(0.05)

        readiness = self.evaluate_readiness()
        return {
            "quota_version": self.quota_version,
            "committed_now": sorted(committed_now),
            "failures": failures,
            "readiness": readiness,
            "water": self.current_water(),
            "replay": self.ledger.replay_and_verify(),
        }

    def byte_records(self) -> dict[str, dict[str, str | None]]:
        result: dict[str, dict[str, str | None]] = {}
        rows = self.ledger.conn.execute("SELECT shard_id,loss_digest FROM shard_outcomes WHERE status='committed' ORDER BY shard_id").fetchall()
        for row in rows:
            shard_id = row["shard_id"]
            outcome = self.ledger.conn.execute("SELECT outcome_blob FROM shard_outcomes WHERE shard_id=?", (shard_id,)).fetchone()
            outcome_obj = __import__("json").loads(bytes(outcome["outcome_blob"]).decode())
            lease_id = outcome_obj["lease_attribution"]["lease_id"]
            lease_blob = self.ledger.canonical_blob("leases", "lease_id", lease_id, "lease_blob")
            water_blob = self.ledger.canonical_blob("water_records", "record_key", f"shard:{shard_id}", "record_blob")
            result[shard_id] = {
                "loss_digest": row["loss_digest"],
                "outcome_sha256": None if outcome["outcome_blob"] is None else __import__("shard_diag.canonical", fromlist=["sha256_hex"]).sha256_hex(bytes(outcome["outcome_blob"])),
                "lease_sha256": None if lease_blob is None else __import__("shard_diag.canonical", fromlist=["sha256_hex"]).sha256_hex(lease_blob),
                "water_sha256": None if water_blob is None else __import__("shard_diag.canonical", fromlist=["sha256_hex"]).sha256_hex(water_blob),
            }
        return result


def compare_deterministic_outcomes(left: dict[str, Any], right: dict[str, Any]) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []

    def walk(a: Any, b: Any, path: str, shard_id: str) -> None:
        ignored_leaf_keys = {
            "monotonic_ms", "monotonic_ns", "granted_wall_ms", "expires_wall_ms",
            "granted_monotonic_ns", "expires_monotonic_ns", "lease_id",
            "completed_count", "snapshot_version",
        }
        if path.rsplit(".", 1)[-1] in ignored_leaf_keys:
            return
        if type(a) != type(b):
            mismatches.append({"shard_id": shard_id, "field": path, "left_type": type(a).__name__, "right_type": type(b).__name__})
            return
        if isinstance(a, dict):
            for key in sorted(set(a) | set(b)):
                if key not in a or key not in b:
                    mismatches.append({"shard_id": shard_id, "field": f"{path}.{key}", "issue": "missing_key"})
                else:
                    walk(a[key], b[key], f"{path}.{key}", shard_id)
        elif isinstance(a, list):
            if len(a) != len(b):
                mismatches.append({"shard_id": shard_id, "field": path, "issue": "length", "left": len(a), "right": len(b)})
            for index, (ia, ib) in enumerate(zip(a, b)):
                walk(ia, ib, f"{path}[{index}]", shard_id)
        elif a != b:
            mismatches.append({"shard_id": shard_id, "field": path, "left": a, "right": b})

    for shard_id in sorted(set(left) | set(right)):
        if shard_id not in left or shard_id not in right:
            mismatches.append({"shard_id": shard_id, "field": "$", "issue": "missing_shard"})
            continue
        walk(left[shard_id], right[shard_id], "$", shard_id)
    return mismatches
