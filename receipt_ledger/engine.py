from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .config import validate_rules
from .constants import Code
from .normalizer import attribute_requests, build_receipt_stage, calculate_resources
from .utils import sha256_json


class LedgerEngine:
    def __init__(
        self,
        inputs: dict[str, Any],
        receipt_texts: dict[str, str],
        config_hash: str,
        *,
        jobs: int = 4,
        injection_results: list[dict[str, Any]] | None = None,
    ) -> None:
        self.workers = list(inputs.get("workers", []))
        self.batches = list(inputs.get("batches", []))
        self.requests = {item["request_id"]: dict(item) for item in inputs.get("requests", [])}
        self.rules = dict(inputs.get("rules") or {})
        self.input_failures = list(inputs.get("failures", []))
        self.receipt_texts = dict(receipt_texts)
        self.config_hash = config_hash
        self.jobs = max(1, int(jobs))
        self.injection_results = list(injection_results or [])

        self.command_cursor = 0
        self.command_log: list[dict[str, Any]] = []
        self.command_failures: list[dict[str, Any]] = []
        self.entries: dict[str, dict[str, Any]] = {}
        self.attributions: dict[str, dict[str, Any]] = {}
        self.batch_metrics: dict[str, dict[str, Any]] = {}
        self.pending: dict[str, dict[str, Any]] = {}
        self.param_digests: dict[str, str] = {}
        self.diagnostic_events: list[dict[str, Any]] = []
        self.event_code_counts: dict[str, int] = {}
        self.cycles: list[dict[str, Any]] = []
        self.dependency_index: dict[str, list[str]] = {}
        self.pending_affected: list[str] = []
        self.pending_propagation: list[dict[str, Any]] = []
        self.receipt_stage: dict[str, Any] | None = None
        self.parse_input_hash: str | None = None
        self.rules_hash = self.merge_rules_hash(self.rules)
        self._refresh_param_digests_from_receipts(initial=True)

    @staticmethod
    def merge_rules_hash(rules: dict[str, Any]) -> str:
        authoritative = {key: value for key, value in rules.items() if key != "budgets"}
        return sha256_json(authoritative)

    def _refresh_param_digests_from_receipts(self, initial: bool = False) -> None:
        if not initial or self.param_digests:
            return
        versions: dict[str, set[str]] = {}
        stage = self._ensure_receipt_stage()
        for receipt in stage["receipts"]:
            digest_value = receipt.get("param_digest")
            worker_id = receipt.get("worker_id")
            version = receipt.get("dependency_param_version")
            if digest_value and worker_id and version:
                versions.setdefault(f"{worker_id}:{version}", set()).add(digest_value)
        for key, values in versions.items():
            self.param_digests[key] = next(iter(sorted(values)))

    def _ensure_receipt_stage(self) -> dict[str, Any]:
        parse_hash = sha256_json({
            "receipt_texts": self.receipt_texts,
            "workers": self.workers,
            "batches": self.batches,
        })
        if self.receipt_stage is not None and self.parse_input_hash == parse_hash:
            return self.receipt_stage
        self.receipt_stage = build_receipt_stage(self.receipt_texts, self.workers, self.batches)
        self.parse_input_hash = parse_hash
        return self.receipt_stage

    @classmethod
    def from_state(cls, state: dict[str, Any], *, jobs: int = 1) -> "LedgerEngine":
        engine = cls.__new__(cls)
        engine.__dict__.update(state)
        engine.jobs = max(1, int(jobs))
        engine.__dict__.setdefault("pending_affected", [])
        engine.__dict__.setdefault("pending_propagation", [])
        return engine

    def to_state(self) -> dict[str, Any]:
        return dict(self.__dict__)

    def apply_commands(self, commands: list[dict[str, Any]]) -> None:
        for command in commands[self.command_cursor:]:
            self.apply_command(command)

    def apply_command(self, command: dict[str, Any]) -> dict[str, Any]:
        index = self.command_cursor
        record = {
            "index": index,
            "command_id": command.get("command_id"),
            "op": command.get("op"),
            "status": "applied",
        }
        op = command.get("op")
        if op == "grant_budget":
            budgets = dict(self.rules.get("budgets", {}))
            if isinstance(command.get("batch_budget"), int):
                budgets["batch_budget_units"] = command["batch_budget"]
            if isinstance(command.get("worker_quota"), int):
                budgets["worker_quota_units"] = command["worker_quota"]
            self.rules = {**self.rules, "budgets": budgets}
        elif op == "set_param_version":
            key = f"{command['worker_id']}:{command['param_version']}"
            self.param_digests[key] = command.get("param_digest") or sha256_json(key)
        elif op == "replace_request":
            request_id = command["request_id"]
            if request_id not in self.requests:
                self.command_failures.append({
                    "code": Code.COMMAND_FAILED,
                    "source": "commands",
                    "item_id": request_id,
                    "message": "cannot replace unknown request",
                })
                record["status"] = "failed"
            else:
                replacement = command.get("request", {})
                self.requests[request_id] = {**self.requests[request_id], **replacement, "request_id": request_id}
        elif op == "switch_rules":
            worker_ids = {worker["worker_id"] for worker in self.workers}
            candidate_rules = command.get("rules", {})
            _clean_rules, rule_failures = validate_rules(candidate_rules, worker_ids, source="command.switch_rules")
            if rule_failures:
                self.command_failures.extend(rule_failures)
                record["status"] = "failed"
            else:
                self.rules = dict(candidate_rules)
                self.rules_hash = self.merge_rules_hash(self.rules)
        elif op in {"normalize", "checkpoint"}:
            pass
        else:
            self.command_failures.append({
                "code": Code.COMMAND_FAILED,
                "source": "commands",
                "item_id": str(command.get("command_id")),
                "message": f"unsupported command op {op!r}",
            })
            record["status"] = "failed"

        if op in {"set_param_version", "replace_request", "switch_rules"} and record["status"] == "applied":
            affected, propagation = self._affected_ids_for_command(command)
            self.pending_affected = sorted(set(self.pending_affected) | affected)
            self.pending_propagation.extend(propagation)

        self.command_cursor = index + 1
        self.command_log.append(record)
        if op == "normalize":
            return self._normalize(command, index)
        return {"command_id": command.get("command_id"), "invalidated": [], "reused": [], "recomputed": [], "pending": []}

    def _affected_ids_for_command(self, command: dict[str, Any]) -> tuple[set[str], list[dict[str, Any]]]:
        op = command.get("op")
        affected: set[str] = set()
        propagation: list[dict[str, Any]] = []
        stage = self._ensure_receipt_stage()
        if op == "set_param_version":
            worker_id = command["worker_id"]
            version = int(command["param_version"])
            batch_ids = sorted(
                metric["entry_id"] for metric in self._desired_batch_metrics(stage)
                if metric["worker_id"] == worker_id and metric["param_version"] == version
            )
            request_ids = sorted({
                receipt["request_id"]
                for receipt in stage["receipts"]
                if receipt.get("worker_id") == worker_id
                and receipt.get("dependency_param_version") == version
            })
            attribution_ids = [f"attribution:{request_id}" for request_id in request_ids]
            affected.update(batch_ids)
            affected.update(attribution_ids)
            propagation.append({
                "root": f"param_version:{worker_id}:{version}",
                "batch_entries": batch_ids,
                "attribution_entries": attribution_ids,
            })
        elif op == "replace_request":
            request_id = command["request_id"]
            attribution_id = f"attribution:{request_id}"
            batch_ids = sorted(
                entry_id for entry_id, metric in self.batch_metrics.items()
                if request_id in metric.get("ordered_request_ids", [])
            )
            if not batch_ids:
                batch_ids = sorted(
                    f"batch:{batch['batch_id']}:{batch['worker_id']}"
                    for batch in self.batches if request_id in self.requests
                    and self.requests[request_id].get("batch_id") == batch["batch_id"]
                    and self.requests[request_id].get("worker_id") == batch["worker_id"]
                )
            affected.add(attribution_id)
            affected.update(batch_ids)
            propagation.append({
                "root": f"request:{request_id}",
                "attribution_entries": [attribution_id],
                "batch_entries": batch_ids,
            })
        elif op == "switch_rules":
            attribution_ids = sorted(f"attribution:{request_id}" for request_id in self.requests)
            batch_ids = sorted(f"batch:{batch['batch_id']}:{batch['worker_id']}" for batch in self.batches)
            affected.update(attribution_ids)
            affected.update(batch_ids)
            propagation.append({"root": f"rules:{self.rules_hash}", "attribution_entries": attribution_ids,
                               "batch_entries": batch_ids})
        return affected, propagation

    def _desired_batch_metrics(self, stage: dict[str, Any]) -> list[dict[str, Any]]:
        return calculate_resources(stage, self.batches, self.workers, self.rules, self.param_digests)

    def _build_desired_entries(self) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
        stage = self._ensure_receipt_stage()
        attribution_result = attribute_requests(stage["receipts"])
        metrics = self._desired_batch_metrics(stage)
        desired: dict[str, dict[str, Any]] = {}
        dependencies: dict[str, list[str]] = {}

        parse_id = "stage:text_parse"
        desired[parse_id] = {
            "kind": "stage",
            "value": {"receipt_counts": stage["receipt_counts"], "sidecar_count": len(stage["sidecars"])},
            "input_hash": self.parse_input_hash,
            "batch_id": None,
            "worker_id": None,
        }
        dependencies[parse_id] = ["receipt_files"]

        for attribution in attribution_result["attributions"]:
            request_id = attribution["request_id"]
            entry_id = f"attribution:{request_id}"
            candidate_uids = attribution["candidate_receipt_uids"]
            candidate_receipts = [r for r in stage["receipts"] if r["receipt_uid"] in candidate_uids]
            digest_keys = sorted({
                f"{r.get('worker_id')}:{r.get('dependency_param_version')}"
                for r in candidate_receipts if r.get("worker_id") and r.get("dependency_param_version")
            })
            input_facts = {
                "attribution": attribution,
                "candidate_receipts": sorted(candidate_receipts, key=lambda r: r["receipt_uid"]),
                "request": self.requests.get(request_id),
                "rules_policy": self.rules.get("policy", {}),
                "rules_hash": self.rules_hash,
                "relevant_param_digests": {key: self.param_digests.get(key) for key in digest_keys},
            }
            batch_id = attribution.get("authoritative_batch_id") or min(
                (r.get("batch_id", "") for r in candidate_receipts), default=""
            )
            worker_id = attribution.get("authoritative_worker_id") or min(
                (r.get("worker_id", "") for r in candidate_receipts if r.get("worker_id")), default=""
            )
            desired[entry_id] = {
                "kind": "attribution",
                "value": attribution,
                "input_hash": sha256_json(input_facts),
                "batch_id": batch_id,
                "worker_id": worker_id,
            }
            deps = [f"request:{request_id}", f"rules:{self.rules_hash}"]
            for receipt in candidate_receipts:
                deps.append(f"batch:{receipt.get('batch_id')}:{receipt.get('worker_id')}")
                version = receipt.get("dependency_param_version")
                if receipt.get("worker_id") and version:
                    deps.append(f"param_version:{receipt['worker_id']}:{version}")
            dependencies[entry_id] = sorted(set(deps))

        for metric in metrics:
            entry_id = metric["entry_id"]
            key = (metric["batch_id"], metric["worker_id"])
            batch_def = next((b for b in self.batches if (b["batch_id"], b["worker_id"]) == key), {})
            worker = next((w for w in self.workers if w["worker_id"] == metric["worker_id"]), {})
            batch_receipts = [
                r for r in stage["receipts"]
                if r.get("batch_id") == metric["batch_id"] and r.get("worker_id") == metric["worker_id"]
            ]
            input_facts = {
                "batch": batch_def,
                "worker": worker,
                "receipts": sorted(batch_receipts, key=lambda r: r["receipt_uid"]),
                "rules_resource": self.rules.get("resource", {}),
                "rules_hash": self.rules_hash,
                "param_digest": self.param_digests.get(f"{metric['worker_id']}:{metric['param_version']}"),
            }
            desired[entry_id] = {
                "kind": "batch_metric",
                "value": metric,
                "input_hash": sha256_json(input_facts),
                "batch_id": metric["batch_id"],
                "worker_id": metric["worker_id"],
            }
            dependencies[entry_id] = sorted({
                f"batch:{metric['batch_id']}:{metric['worker_id']}",
                f"param_version:{metric['worker_id']}:{metric['param_version']}",
                f"rules:{self.rules_hash}",
            })

        summary_id = "stage:ledger_summary"
        desired[summary_id] = {
            "kind": "summary",
            "value": {"depends_on": sorted(desired.keys())},
            "input_hash": sha256_json(sorted(desired.keys()) + [self.command_cursor]),
            "batch_id": None,
            "worker_id": None,
        }
        dependencies[summary_id] = sorted(desired.keys())
        self.dependency_index = self._reverse_dependency_index(dependencies)
        return desired, dependencies

    @staticmethod
    def _reverse_dependency_index(dependencies: dict[str, list[str]]) -> dict[str, list[str]]:
        reverse: dict[str, set[str]] = {}
        for entry_id, nodes in dependencies.items():
            for node in nodes:
                reverse.setdefault(node, set()).add(entry_id)
        return {node: sorted(ids) for node, ids in sorted(reverse.items())}

    def _normalize(self, command: dict[str, Any], command_index: int) -> dict[str, Any]:
        desired, _dependencies = self._build_desired_entries()
        affected = set(self.pending_affected)
        propagation = list(self.pending_propagation)
        self.pending_affected = []
        self.pending_propagation = []
        budgets = self.rules.get("budgets", {})
        batch_budget = int(budgets.get("batch_budget_units", 2))
        worker_quota = int(budgets.get("worker_quota_units", 4))

        reused: list[str] = []
        needs_work: dict[str, dict[str, Any]] = {}
        for entry_id, desired_entry in desired.items():
            old = self.entries.get(entry_id)
            if entry_id in affected:
                needs_work[entry_id] = desired_entry
            elif old and old.get("status") == "complete" and old.get("input_hash") == desired_entry["input_hash"]:
                reused.append(entry_id)
            elif old and old.get("status") == "pending":
                needs_work[entry_id] = desired_entry
            else:
                needs_work[entry_id] = desired_entry

        # Text parsing is a full-stage product. Reuse the persisted stage when its hash matches.
        parse_id = "stage:text_parse"
        if parse_id in reused and self.receipt_stage is not None:
            pass
        elif parse_id in needs_work:
            self._ensure_receipt_stage()

        old_pending = [
            pending for pending in self.pending.values()
            if pending["entry_id"] in needs_work or pending["entry_id"] in desired
        ]
        old_pending.sort(key=lambda item: (item["enqueued_command_index"], item["entry_id"]))
        new_ids = sorted(set(needs_work) - {item["entry_id"] for item in old_pending})
        queue = [
            *[item["entry_id"] for item in old_pending],
            *new_ids,
        ]

        admitted: list[str] = []
        still_pending: list[str] = []
        batch_used: dict[str, int] = {}
        worker_used: dict[str, int] = {}
        pending_records: dict[str, dict[str, Any]] = {}
        budget_events: list[dict[str, Any]] = []

        for entry_id in queue:
            desired_entry = desired[entry_id]
            batch_id = desired_entry.get("batch_id") or "_global"
            worker_id = desired_entry.get("worker_id") or "_global"
            reason = None
            if batch_used.get(batch_id, 0) >= batch_budget:
                reason = "batch_budget"
            elif worker_used.get(worker_id, 0) >= worker_quota:
                reason = "worker_quota"
            if reason:
                still_pending.append(entry_id)
                record = {
                    "entry_id": entry_id,
                    "enqueued_command_index": self.pending.get(entry_id, {}).get("enqueued_command_index", command_index),
                    "last_attempt_command_index": command_index,
                    "batch_id": desired_entry.get("batch_id"),
                    "worker_id": desired_entry.get("worker_id"),
                    "reason": reason,
                    "code": Code.NORMALIZE_BUDGET_PENDING,
                }
                pending_records[entry_id] = record
                budget_events.append({
                    "command_index": command_index,
                    "command_id": command.get("command_id"),
                    "code": Code.NORMALIZE_BUDGET_PENDING,
                    **record,
                })
                self.event_code_counts[Code.NORMALIZE_BUDGET_PENDING] = self.event_code_counts.get(Code.NORMALIZE_BUDGET_PENDING, 0) + 1
                continue
            admitted.append(entry_id)
            batch_used[batch_id] = batch_used.get(batch_id, 0) + 1
            worker_used[worker_id] = worker_used.get(worker_id, 0) + 1

        computed: dict[str, Any] = {}
        computable = [entry_id for entry_id in admitted if entry_id != "stage:ledger_summary"]
        if self.jobs == 1 or len(computable) <= 1:
            for entry_id in computable:
                computed[entry_id] = desired[entry_id]["value"]
        else:
            with ThreadPoolExecutor(max_workers=min(self.jobs, max(1, len(computable)))) as pool:
                future_map = {pool.submit(lambda item: item["value"], desired[entry_id]): entry_id for entry_id in computable}
                for future, entry_id in list(future_map.items()):
                    computed[entry_id] = future.result()

        recomputed: list[str] = []
        for entry_id in sorted(admitted):
            desired_entry = desired[entry_id]
            value = desired_entry["value"] if entry_id == "stage:ledger_summary" else computed[entry_id]
            self.entries[entry_id] = {
                "entry_id": entry_id,
                "kind": desired_entry["kind"],
                "status": "complete",
                "input_hash": desired_entry["input_hash"],
                "batch_id": desired_entry.get("batch_id"),
                "worker_id": desired_entry.get("worker_id"),
                "last_command_index": command_index,
            }
            if entry_id.startswith("attribution:"):
                self.attributions[value["request_id"]] = value
            elif entry_id.startswith("batch:"):
                self.batch_metrics[entry_id] = value
            elif entry_id == "stage:text_parse":
                self._ensure_receipt_stage()
            recomputed.append(entry_id)
            self.pending.pop(entry_id, None)

        for entry_id, record in pending_records.items():
            self.pending[entry_id] = record
            self.entries[entry_id] = {
                "entry_id": entry_id,
                "kind": desired[entry_id]["kind"],
                "status": "pending",
                "input_hash": desired[entry_id]["input_hash"],
                "batch_id": desired[entry_id].get("batch_id"),
                "worker_id": desired[entry_id].get("worker_id"),
                "last_command_index": command_index,
            }
        for entry_id in list(self.pending):
            if entry_id not in desired:
                self.pending.pop(entry_id, None)

        self.diagnostic_events.extend(budget_events)
        cycle = {
            "command_index": command_index,
            "command_id": command.get("command_id"),
            "propagation": propagation,
            "invalidated": sorted(affected),
            "reused": sorted(reused),
            "recomputed": sorted(recomputed),
            "pending": sorted(still_pending),
            "admitted_order_canonical": sorted(admitted),
        }
        self.cycles.append(cycle)
        return cycle

    def current_failures(self) -> list[dict[str, Any]]:
        failures = list(self.input_failures) + list(self.command_failures)
        if self.receipt_stage:
            failures.extend(self.receipt_stage.get("parse_failures", []))
        for attribution in sorted(self.attributions.values(), key=lambda item: item["request_id"]):
            if attribution.get("status") == "conflict":
                failures.append({
                    "code": Code.ATTRIBUTION_CONFLICT,
                    "source": "attribution",
                    "item_id": attribution["request_id"],
                    "message": "unresolved attribution conflict",
                    "candidate_receipt_uids": attribution.get("candidate_receipt_uids", []),
                })
            if attribution.get("undecidable"):
                failures.append({
                    "code": Code.ATTRIBUTION_UNDECIDABLE,
                    "source": "attribution",
                    "item_id": attribution["request_id"],
                    "message": "duplicate reference is missing or dangling",
                    "candidate_receipt_uids": attribution.get("candidate_receipt_uids", []),
                })
        failures.sort(key=lambda item: (
            item.get("code", ""), item.get("source", ""), str(item.get("item_id", "")),
            str(item.get("index", "")), item.get("message", ""),
        ))
        return failures

    def business_code_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for failure in self.current_failures():
            code = failure.get("code", Code.SCHEMA_INVALID)
            counts[code] = counts.get(code, 0) + 1
        for code, count in self.event_code_counts.items():
            counts[code] = counts.get(code, 0) + count
        return dict(sorted(counts.items()))

    def latency_statistics(self) -> list[dict[str, Any]]:
        groups: dict[tuple[str, int], list[int]] = {}
        for metric in self.batch_metrics.values():
            if isinstance(metric.get("latency_ms"), int):
                groups.setdefault((metric["worker_id"], metric["param_version"]), []).append(metric["latency_ms"])
        stats = []
        for (worker_id, version), latencies in sorted(groups.items()):
            stats.append({
                "worker_id": worker_id,
                "param_version": version,
                "batch_count": len(latencies),
                "latency_min_ms": min(latencies),
                "latency_max_ms": max(latencies),
                "latency_sum_ms": sum(latencies),
            })
        return stats

    def export_ledger(self) -> dict[str, Any]:
        stage = self._ensure_receipt_stage()
        failures = self.current_failures()
        pending_queue = sorted(self.pending.values(), key=lambda item: (item["enqueued_command_index"], item["entry_id"]))
        reuse_hits = [
            {"command_index": cycle["command_index"], "command_id": cycle["command_id"], "entry_id": entry_id}
            for cycle in self.cycles for entry_id in cycle["reused"]
        ]
        invalidation_items = [
            {"command_index": cycle["command_index"], "command_id": cycle["command_id"], "entry_id": entry_id}
            for cycle in self.cycles for entry_id in cycle["invalidated"]
        ]
        worker_version: dict[tuple[str, int], dict[str, Any]] = {}
        for metric in sorted(self.batch_metrics.values(), key=lambda item: item["entry_id"]):
            key = (metric["worker_id"], metric["param_version"])
            group = worker_version.setdefault(key, {
                "worker_id": metric["worker_id"],
                "param_version": metric["param_version"],
                "request_count": 0,
                "mac": 0,
                "batch_entries": [],
            })
            group["request_count"] += metric["request_count"]
            group["mac"] += metric["mac"]
            group["batch_entries"].append(metric["entry_id"])
        summary = {
            "worker_count": len(self.workers),
            "request_stream_count": len(self.requests),
            "receipt_counts": stage["receipt_counts"],
            "attribution_count": len(self.attributions),
            "batch_metric_count": len(self.batch_metrics),
            "pending_count": len(self.pending),
            "conflict_count": sum(1 for a in self.attributions.values() if a.get("conflict")),
            "undecidable_count": sum(1 for a in self.attributions.values() if a.get("undecidable")),
            "failure_count": len(failures),
            "reuse_hit_count": len(reuse_hits),
            "invalidation_item_count": len(invalidation_items),
        }
        return {
            "schema_version": "receipt-ledger.v1",
            "config_hash": self.config_hash,
            "rules_hash": self.rules_hash,
            "command_cursor": self.command_cursor,
            "command_log": self.command_log,
            "summary": summary,
            "receipt_counts": stage["receipt_counts"],
            "receipts": sorted(stage["receipts"], key=lambda item: item["receipt_uid"]),
            "sidecars": sorted(stage["sidecars"], key=lambda item: item["receipt_uid"]),
            "r3_to_r1_truncated_fields": stage["r3_to_r1_truncated_fields"],
            "attributions": [self.attributions[key] for key in sorted(self.attributions)],
            "batch_metrics": [self.batch_metrics[key] for key in sorted(self.batch_metrics)],
            "latency_statistics": self.latency_statistics(),
            "entries": [self.entries[key] for key in sorted(self.entries)],
            "pending_queue": pending_queue,
            "cycles": self.cycles,
            "reuse_hits": reuse_hits,
            "invalidation_items": invalidation_items,
            "dependency_index": self.dependency_index,
            "diagnostic_events": self.diagnostic_events,
            "failures": failures,
            "business_code_counts": self.business_code_counts(),
            "injection_results": sorted(self.injection_results, key=lambda item: item.get("id", "")),
            "worker_version_metrics": [worker_version[key] for key in sorted(worker_version)],
            "param_digests": dict(sorted(self.param_digests.items())),
        }
