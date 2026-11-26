from __future__ import annotations

import copy
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from .cache import CacheStore, SnapshotStore
from .canonical import fingerprint
from .errors import (
    BusinessError,
    E_BUDGET_PENDING,
    E_DUPLICATE_DELIVERY,
    E_LEASE_EXPIRED,
    E_LEDGER_MISMATCH,
    invalid_input,
)
from .keys import global_cache_key, shard_product_key, slice_cache_key, topic_parameter_fingerprint
from .leases import LeaseManager, SlotPool, monotonic_ms
from .ledger import ReceiptLedger
from .sampling import aggregate_term_counts, compute_shard_receipt
from .scheduler import PendingShard, build_schedule
from .validation import validate_document, validate_inference


class ManualClock:
    def __init__(self, start_ms: int = 1_760_000_000_000) -> None:
        self.now_ms = start_ms
        self.lock = threading.Lock()

    def __call__(self) -> int:
        with self.lock:
            return self.now_ms

    def advance(self, milliseconds: int) -> int:
        with self.lock:
            self.now_ms += int(milliseconds)
            return self.now_ms


def direct_dispatch(node_id: str, request: dict[str, Any]) -> dict[str, Any]:
    request_with_node = {**request, "node_id": node_id}
    return {"ok": True, "node_id": node_id, "receipt": compute_shard_receipt(request_with_node)}


class TopicLedgerEngine:
    def __init__(
        self,
        *,
        topology: dict[str, Any],
        inference: dict[str, Any],
        vocab: list[dict[str, Any]],
        documents: list[dict[str, Any]],
        state_dir: str | Path,
        snapshot_dir: str | Path | None = None,
        dispatcher: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
        clock: Callable[[], int] | str | None = "manual",
        global_concurrency_override: int | None = None,
    ) -> None:
        self.topology = copy.deepcopy(topology)
        self.inference = copy.deepcopy(inference)
        self.vocab = sorted(copy.deepcopy(vocab), key=lambda item: item["term_id"])
        self.term_ids = [int(item["term_id"]) for item in self.vocab]
        self.vocab_size = len(self.vocab)
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.cache = CacheStore(self.state_dir / "cache")
        snapshot_path = Path(snapshot_dir) if snapshot_dir is not None else self.state_dir / "snapshots"
        self.snapshots = SnapshotStore(snapshot_path)

        self.shards = list(self.topology["shards"])
        self.shard_set = set(self.shards)
        self.routing = dict(self.topology["routing"])
        self.node_configs = {node["node_id"]: node for node in self.topology["nodes"]}
        self.lease_limits = {
            node_id: int(node["lease_limit"]) for node_id, node in self.node_configs.items()
        }
        self.quotas = {
            node_id: int(node["concurrency_quota"]) for node_id, node in self.node_configs.items()
        }
        self.effective_concurrency = int(
            global_concurrency_override or self.topology["global_concurrency"]
        )
        self.slots = SlotPool(self.effective_concurrency, self.quotas)

        if clock == "manual" or clock is None:
            self.clock: Callable[[], int] = ManualClock()
            self.manual_clock: ManualClock | None = self.clock  # type: ignore[assignment]
        elif clock == "real":
            self.clock = monotonic_ms
            self.manual_clock = None
        else:
            self.clock = clock
            self.manual_clock = None
        self.lease_manager = LeaseManager(ttl_ms=int(self.inference["lease_ttl_ms"]), clock=self.clock)

        self.documents = {doc["doc_id"]: copy.deepcopy(doc) for doc in documents}
        self.document_revisions = {doc["doc_id"]: int(doc["revision"]) for doc in documents}
        self.dispatcher = dispatcher

        self.global_version = 0
        self.command_count = 0
        self.last_command_id: str | None = None
        self.code_counts: dict[str, int] = {}
        self.pending: list[PendingShard] = []
        self.consumed_budget = 0
        self.config_epoch = 0
        self.reuse_hit_count = 0
        self.input_keys: dict[str, str] = {}
        self.product_keys: dict[str, str] = {}
        self.cache_runs: list[dict[str, Any]] = []
        self.voided_receipts: list[dict[str, Any]] = []
        self.last_digests: dict[str, str] = {}
        self.global_cache_key: str | None = None
        self.config_hash = self._compute_config_hash()
        self.ledger = ReceiptLedger(term_ids=self.term_ids, topic_count=int(self.inference["topic_count"]))
        self.lock = threading.RLock()
        self.latest_summary = self.progress_summary()

    def _compute_config_hash(self) -> str:
        return fingerprint(
            {
                "inference": self.inference,
                "shards": self.shards,
                "routing": self.routing,
                "node_lease_limits": dict(sorted(self.lease_limits.items())),
                "vocab_fingerprint": fingerprint(self.vocab),
            }
        )

    def record_code(self, code: str) -> None:
        self.code_counts[code] = self.code_counts.get(code, 0) + 1

    def progress_summary(self) -> dict[str, Any]:
        completed = len(self.ledger.receipts)
        total = len(self.shards)
        active = self.lease_manager.active_report()
        active_generations = sorted({int(item["generation"]) for item in active})
        effective_generation = max(active_generations) if active_generations else self.lease_manager.generation
        return {
            "completed_shards": completed,
            "total_shards": total,
            "lease_generation": int(effective_generation),
            "latest_issued_generation": int(self.lease_manager.generation),
            "active_lease_generations": active_generations,
            "pending_shards": len(self.pending),
            "progress": f"{completed}/{total}",
            "progress_permille": int(completed * 1000 / total) if total else 1000,
            "config_hash": self.config_hash,
            "global_version": self.global_version,
            "last_command_id": self.last_command_id,
            "stale": False,
        }

    def _shard_inputs(self) -> dict[str, dict[str, Any]]:
        topic_fp = topic_parameter_fingerprint(self.inference, self.vocab_size)
        result: dict[str, dict[str, Any]] = {}
        for shard_id in self.shards:
            docs = sorted(
                (doc for doc in self.documents.values() if doc["shard_id"] == shard_id),
                key=lambda doc: doc["doc_id"],
            )
            refs = [
                {
                    "doc_id": doc["doc_id"],
                    "revision": int(doc["revision"]),
                    "content_fingerprint": fingerprint(
                        {
                            "doc_id": doc["doc_id"],
                            "revision": int(doc["revision"]),
                            "term_ids": list(doc["term_ids"]),
                        }
                    ),
                }
                for doc in docs
            ]
            terms: list[int] = []
            for doc in docs:
                terms.extend(int(term_id) for term_id in doc["term_ids"])
            term_counts = aggregate_term_counts(terms)
            slice_key = slice_cache_key(
                shard_id=shard_id, document_refs=refs, term_counts=term_counts
            )
            product_key = shard_product_key(
                slice_key=slice_key,
                topic_fp=topic_fp,
                bit_width_tier=self.inference["fixed_point"]["bit_width_tier"],
                rounds=int(self.inference["sample_rounds"]),
            )
            result[shard_id] = {
                "documents": docs,
                "document_refs": refs,
                "term_counts": term_counts,
                "slice_key": slice_key,
                "product_key": product_key,
                "cost": len(terms) * int(self.inference["sample_rounds"]),
                "topic_fp": topic_fp,
            }
        return result

    def _invoke_dispatcher(
        self, node_id: str, request: dict[str, Any]
    ) -> dict[str, Any]:
        try:
            if self.dispatcher is not None:
                return self.dispatcher(node_id, request)
            return direct_dispatch(node_id, request)
        except BusinessError as exc:
            if exc.code == E_DUPLICATE_DELIVERY and not request.get("prior_receipt_digest"):
                current_digest = (exc.details or {}).get("current_receipt_digest")
                if current_digest:
                    self.record_code(E_DUPLICATE_DELIVERY)
                    retry_request = {
                        **request,
                        "prior_receipt_digest": current_digest,
                        "batch_id": f"{request['batch_id']}:digest-retry",
                    }
                    if self.dispatcher is not None:
                        return self.dispatcher(node_id, retry_request)
                    return direct_dispatch(node_id, retry_request)
            raise

    def _execute_item(
        self,
        item: Any,
        generation: int,
        shard_input: dict[str, Any],
    ) -> dict[str, Any]:
        ticket = None
        try:
            ticket = self.slots.acquire(item.node_id, timeout_seconds=30.0)
            if not self.lease_manager.attach_ticket(item.shard_id, generation, ticket):
                return {"status": "expired", "code": E_LEASE_EXPIRED}
            request = {
                "op": "sample",
                "batch_id": item.batch_id,
                "shard_id": item.shard_id,
                "term_counts": shard_input["term_counts"],
                "document_refs": shard_input["document_refs"],
                "config": self.inference,
                "vocab_size": self.vocab_size,
                "prior_receipt_digest": self.last_digests.get(item.shard_id),
            }
            response = self._invoke_dispatcher(item.node_id, request)
            receipt = dict(response["receipt"])
            receipt.pop("elapsed_monotonic_ms", None)
            if not self.lease_manager.is_current(item.shard_id, generation):
                return {"status": "expired", "code": E_LEASE_EXPIRED, "receipt": receipt}
            return {"status": "ok", "receipt": receipt}
        except BusinessError as exc:
            return {"status": "error", "code": exc.code, "error": exc.to_dict()}
        except Exception as exc:  # protocol/process failure becomes a recorded shard failure
            return {"status": "error", "code": "E_WORKER_PROTOCOL", "error": {"message": str(exc)}}
        finally:
            if ticket is not None:
                ticket.release()

    def run_sampling(self, command: dict[str, Any]) -> dict[str, Any]:
        base_batch = command.get("batch_id") if isinstance(command.get("batch_id"), str) else f"batch-{self.command_count:04d}"
        force_full = bool(command.get("force_full", False))
        shard_inputs = self._shard_inputs()

        invalidated: list[dict[str, str]] = []
        reused: list[dict[str, str]] = []
        sampled: list[str] = []
        for shard_id in self.shards:
            product_key = shard_inputs[shard_id]["product_key"]
            previous_key = self.input_keys.get(shard_id)
            cached = None if force_full else self.cache.get(product_key)
            if cached is not None:
                receipt = cached["payload"]["receipt"]
                self.ledger.receipts[shard_id] = receipt
                self.last_digests[shard_id] = receipt["receipt_digest"]
                self.reuse_hit_count += 1
                reused.append({"shard_id": shard_id, "product_key": product_key})
                self.pending = [item for item in self.pending if item.shard_id != shard_id]
            else:
                reason = "forced_full" if force_full else "cache_miss"
                if previous_key and previous_key != product_key:
                    reason = "content_or_config_changed"
                invalidated.append(
                    {"shard_id": shard_id, "previous_key": previous_key or "", "product_key": product_key, "reason": reason}
                )
                sampled.append(shard_id)

        cache_run = {
            "command_id": self.last_command_id,
            "batch_id": base_batch,
            "invalidated": invalidated,
            "reused": reused,
            "sampled": list(sampled),
            "reuse_hits": len(reused),
            "force_full": force_full,
        }
        self.cache_runs.append(cache_run)

        batch_ids = {shard_id: f"{base_batch}:{shard_id}" for shard_id in self.shards}
        budget_remaining = max(0, int(self.inference["sample_point_budget"]) - self.consumed_budget)
        schedule = build_schedule(
            ordered_shards=sampled,
            routing=self.routing,
            costs={shard_id: shard_inputs[shard_id]["cost"] for shard_id in self.shards},
            batch_ids=batch_ids,
            budget_limit=budget_remaining,
            lease_limits=self.lease_limits,
            initial_node_lease_counts={
                node_id: self.lease_manager.active_count(node_id) for node_id in self.node_configs
            },
            existing_pending=self.pending,
        )
        self.consumed_budget += schedule.consumed_budget
        self.pending = list(schedule.pending)
        for pending in schedule.pending:
            self.record_code(pending.reason)

        leases: dict[str, int] = {}
        executable: list[Any] = []
        for item in schedule.items:
            try:
                lease = self.lease_manager.issue(
                    shard_id=item.shard_id,
                    node_id=item.node_id,
                    lease_limit=self.lease_limits[item.node_id],
                    batch_id=item.batch_id,
                )
                leases[item.shard_id] = lease.generation
                executable.append(item)
            except BusinessError as exc:
                self.record_code(exc.code)
                if not any(pending.shard_id == item.shard_id for pending in self.pending):
                    self.pending.append(
                        PendingShard(item.shard_id, exc.code, item.batch_id, exc.details or {})
                    )

        outcomes: list[dict[str, Any]] = []
        if executable:
            with ThreadPoolExecutor(max_workers=max(1, self.effective_concurrency)) as executor:
                futures = [
                    executor.submit(
                        self._execute_item,
                        item,
                        leases[item.shard_id],
                        shard_inputs[item.shard_id],
                    )
                    for item in executable
                ]
                outcomes = [future.result() for future in futures]

        item_by_shard = {item.shard_id: item for item in executable}
        for item, outcome in zip(executable, outcomes):
            generation = leases[item.shard_id]
            if outcome["status"] == "ok":
                receipt = outcome["receipt"]
                product_key = shard_inputs[item.shard_id]["product_key"]
                self.cache.put(product_key, {"receipt": receipt})
                try:
                    self.ledger.accept_receipt(
                        receipt,
                        batch_id=item.batch_id,
                        prior_receipt_digest=self.last_digests.get(item.shard_id),
                    )
                    self.last_digests[item.shard_id] = receipt["receipt_digest"]
                    self.record_code("OK")
                    self.pending = [pending for pending in self.pending if pending.shard_id != item.shard_id]
                except BusinessError as exc:
                    self.record_code(exc.code)
                    self.pending.append(
                        PendingShard(item.shard_id, exc.code, item.batch_id, exc.details or {})
                    )
                finally:
                    try:
                        self.lease_manager.release(shard_id=item.shard_id, generation=generation)
                    except BusinessError:
                        pass
            else:
                code = outcome.get("code", "E_WORKER_PROTOCOL")
                if code != E_LEASE_EXPIRED:
                    self.record_code(code)
                if not any(pending.shard_id == item.shard_id for pending in self.pending):
                    self.pending.append(PendingShard(item.shard_id, code, item.batch_id, outcome.get("error", {})))
                try:
                    if self.lease_manager.is_current(item.shard_id, generation):
                        # Leave an explicitly failed lease to TTL/recovery; do not release as success.
                        pass
                except Exception:
                    pass

        closure = self.ledger.rebuild()
        for topic_id in closure.get("mismatched_topics", []):
            self.record_code(E_LEDGER_MISMATCH)

        for shard_id, shard_input in shard_inputs.items():
            self.input_keys[shard_id] = shard_input["product_key"]
            self.product_keys[shard_id] = shard_input["product_key"]

        merged_shard_keys = [
            receipt["cache_keys"]["shard"]
            for receipt in self.ledger.receipts.values()
            if "cache_keys" in receipt
        ]
        if merged_shard_keys:
            topic_fp = next(iter(shard_inputs.values()))["topic_fp"]
            self.global_cache_key = global_cache_key(
                shard_keys=sorted(merged_shard_keys),
                topic_fp=topic_fp,
                bit_width_tier=self.inference["fixed_point"]["bit_width_tier"],
            )
            self.cache.put(
                self.global_cache_key,
                {"distribution": self.ledger.distribution_entries(), "closure": closure},
            )

        return {
            "schedule": schedule.as_dict(),
            "cache": cache_run,
            "closure": closure,
            "global_cache_key": self.global_cache_key,
            "slots": self.slots.metrics(),
        }

    def recover_leases(self) -> dict[str, Any]:
        expired = self.lease_manager.recover_expired()
        expired_reports: list[dict[str, Any]] = []
        for lease in expired:
            self.record_code(E_LEASE_EXPIRED)
            removed = self.ledger.receipts.pop(lease.shard_id, None)
            if removed is not None:
                self.voided_receipts.append(
                    {
                        "shard_id": lease.shard_id,
                        "generation": lease.generation,
                        "receipt_digest": removed.get("receipt_digest"),
                        "code": E_LEASE_EXPIRED,
                    }
                )
                self.ledger.rebuild()
            if not any(item.shard_id == lease.shard_id for item in self.pending):
                self.pending.append(
                    PendingShard(
                        lease.shard_id,
                        E_LEASE_EXPIRED,
                        lease.batch_id,
                        {"generation": lease.generation},
                    )
                )
            expired_reports.append(lease.public_dict())
        return {"expired": expired_reports, "count": len(expired)}

    def _reset_for_config_change(self) -> None:
        self.config_epoch += 1
        self.ledger = ReceiptLedger(
            term_ids=self.term_ids, topic_count=int(self.inference["topic_count"])
        )
        self.pending = []
        self.consumed_budget = 0
        self.voided_receipts = []
        self.global_cache_key = None

    def handle_command(self, command: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            self.command_count += 1
            self.global_version += 1
            command_id = str(command.get("command_id") or f"cmd-{self.command_count:04d}")
            self.last_command_id = command_id
            op = str(command.get("op", ""))
            response: dict[str, Any] = {
                "command_id": command_id,
                "op": op,
                "global_version": self.global_version,
            }
            try:
                if op == "run_sampling":
                    response["result"] = self.run_sampling(command)
                    self.record_code("OK")
                elif op in {"upsert_document", "replace_document"}:
                    raw_document = command.get("document")
                    doc = validate_document(
                        raw_document,
                        f"commands[{command_id}].document",
                        shard_ids=self.shard_set,
                        term_ids=set(self.term_ids),
                        last_revisions=self.document_revisions,
                    )
                    self.documents[doc["doc_id"]] = doc
                    self.record_code("OK")
                    response["document"] = doc
                elif op == "delete_document":
                    doc_id = command.get("doc_id")
                    if not isinstance(doc_id, str) or doc_id not in self.documents:
                        raise invalid_input("doc_id does not reference a current document", "doc_id")
                    del self.documents[doc_id]
                    self.document_revisions[doc_id] = self.document_revisions.get(doc_id, 0)
                    self.record_code("OK")
                    response["deleted_doc_id"] = doc_id
                elif op == "set_config":
                    patch = command.get("patch")
                    if not isinstance(patch, dict):
                        raise invalid_input("set_config requires a patch object", "patch")
                    merged = copy.deepcopy(self.inference)
                    for key, value in patch.items():
                        if key == "fixed_point" and isinstance(value, dict):
                            merged["fixed_point"].update(value)
                        else:
                            merged[key] = value
                    validated = validate_inference(merged, vocab_size=self.vocab_size)
                    old_hash = self.config_hash
                    old_topic_count = self.inference["topic_count"]
                    self.inference = validated
                    self.config_hash = self._compute_config_hash()
                    if self.config_hash != old_hash and validated["topic_count"] != old_topic_count:
                        self._reset_for_config_change()
                    response["config_hash"] = self.config_hash
                    response["restore_decision"] = "rebuild" if self.config_hash != old_hash else "resume"
                    self.record_code("OK")
                elif op == "set_node_lease_limit":
                    node_id = command.get("node_id")
                    if node_id not in self.node_configs:
                        raise invalid_input("node_id is not declared", "node_id")
                    limit = command.get("lease_limit")
                    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
                        raise invalid_input("lease_limit must be a positive integer", "lease_limit")
                    self.node_configs[node_id]["lease_limit"] = limit
                    self.lease_limits[node_id] = limit
                    self.config_hash = self._compute_config_hash()
                    self.record_code("OK")
                elif op == "issue_lease":
                    shard_id = command.get("shard_id")
                    if shard_id not in self.shard_set:
                        raise invalid_input("shard_id is not declared", "shard_id")
                    node_id = command.get("node_id") or self.routing[shard_id]
                    if node_id not in self.node_configs:
                        raise invalid_input("node_id is not declared", "node_id")
                    lease = self.lease_manager.issue(
                        shard_id=shard_id,
                        node_id=node_id,
                        lease_limit=self.lease_limits[node_id],
                        batch_id=str(command.get("batch_id") or f"lease-{self.command_count:04d}:{shard_id}"),
                        ttl_ms=command.get("ttl_ms"),
                    )
                    response["lease"] = lease.public_dict()
                    self.record_code("OK")
                elif op == "renew_lease":
                    lease = self.lease_manager.renew(
                        shard_id=str(command.get("shard_id")),
                        generation=int(command.get("generation")),
                        ttl_ms=command.get("ttl_ms"),
                    )
                    response["lease"] = lease.public_dict()
                    self.record_code("OK")
                elif op == "release_lease":
                    lease = self.lease_manager.release(
                        shard_id=str(command.get("shard_id")),
                        generation=int(command.get("generation")),
                    )
                    response["lease"] = lease.public_dict()
                    self.record_code("OK")
                elif op == "recover_leases":
                    response["result"] = self.recover_leases()
                    self.record_code("OK")
                elif op == "advance_clock_ms":
                    if self.manual_clock is None:
                        raise invalid_input("engine clock is not manually controlled", "advance_clock_ms")
                    milliseconds = command.get("milliseconds")
                    if not isinstance(milliseconds, int) or isinstance(milliseconds, bool) or milliseconds < 0:
                        raise invalid_input("milliseconds must be a non-negative integer", "milliseconds")
                    response["now_ms"] = self.manual_clock.advance(milliseconds)
                    self.record_code("OK")
                elif op == "sleep_ms":
                    milliseconds = command.get("milliseconds")
                    if not isinstance(milliseconds, int) or isinstance(milliseconds, bool) or milliseconds < 0:
                        raise invalid_input("milliseconds must be a non-negative integer", "milliseconds")
                    monotonic_ms_sleep = __import__("time").sleep
                    monotonic_ms_sleep(milliseconds / 1000.0)
                    response["slept_ms"] = milliseconds
                    self.record_code("OK")
                elif op == "snapshot":
                    response["snapshot"] = self.commit_snapshot()
                    self.record_code("OK")
                elif op == "restore":
                    response["restore"] = self.restore_snapshot()
                    self.record_code("OK")
                else:
                    raise invalid_input(f"unknown command op: {op}", "op")

                response["ok"] = True
                response.setdefault("code", "OK")
            except BusinessError as exc:
                self.record_code(exc.code)
                response.update(exc.to_dict())
            except TypeError as exc:
                error = invalid_input(f"invalid command field type: {exc}", "command")
                self.record_code(error.code)
                response.update(error.to_dict())

            response["summary"] = self.progress_summary()
            response["config_hash"] = self.config_hash
            try:
                response.setdefault("snapshot", self.commit_snapshot())
            except BusinessError as exc:
                response["snapshot_error"] = exc.to_dict()
            self.latest_summary = response["summary"]
            return response

    def snapshot_state(self) -> dict[str, Any]:
        ledger_audit = self.ledger.audit()
        return {
            "global_version": self.global_version,
            "config_hash": self.config_hash,
            "snapshot_format_version": 1,
            "documents": sorted(self.documents.values(), key=lambda doc: doc["doc_id"]),
            "config": {"inference": self.inference, "topology": self.topology},
            "receipts": self.ledger.receipts,
            "ledger": ledger_audit,
            "pending": [item.as_dict() for item in self.pending],
            "budget": {
                "limit": int(self.inference["sample_point_budget"]),
                "consumed": self.consumed_budget,
                "remaining": max(0, int(self.inference["sample_point_budget"]) - self.consumed_budget),
            },
            "cache_manifest": {
                "input_keys": dict(sorted(self.input_keys.items())),
                "product_keys": dict(sorted(self.product_keys.items())),
                "runs": self.cache_runs,
                "reuse_hit_count": self.reuse_hit_count,
                "global_cache_key": self.global_cache_key,
                "command_count": self.command_count,
            },
            "code_counts": dict(sorted(self.code_counts.items())),
            "progress": self.progress_summary(),
        }

    def commit_snapshot(self) -> dict[str, Any]:
        return self.snapshots.commit(self.snapshot_state())

    def restore_snapshot(self) -> dict[str, Any]:
        decision = self.snapshots.decide_restore(self.config_hash)
        if decision["decision"] != "resume":
            return decision
        state = self.snapshots.load_state()
        if state is None:
            return {"decision": "start", "reason": "unreadable_snapshot"}
        manifest = state.get("manifest", {})
        documents = state.get("documents", [])
        self.documents = {doc["doc_id"]: doc for doc in documents}
        self.document_revisions = {doc["doc_id"]: int(doc["revision"]) for doc in documents}
        budget = state.get("budget", {})
        self.consumed_budget = int(budget.get("consumed", 0))
        pending_raw = state.get("pending", [])
        self.pending = [
            PendingShard(
                shard_id=item["shard_id"],
                reason=item["reason"],
                batch_id=item["batch_id"],
                detail=item.get("detail", {}),
            )
            for item in pending_raw
        ]
        self.code_counts = {str(key): int(value) for key, value in state.get("code-counts", {}).items()}
        cache_manifest = state.get("cache-manifest", {})
        self.input_keys = dict(cache_manifest.get("input_keys", {}))
        self.product_keys = dict(cache_manifest.get("product_keys", {}))
        self.cache_runs = list(cache_manifest.get("runs", []))
        self.reuse_hit_count = int(cache_manifest.get("reuse_hit_count", 0))
        self.global_cache_key = cache_manifest.get("global_cache_key")
        self.command_count = int(cache_manifest.get("command_count", 0))
        self.global_version = int(manifest.get("global_version", self.command_count))

        receipts = state.get("shard-receipts", {})
        self.ledger = ReceiptLedger(term_ids=self.term_ids, topic_count=int(self.inference["topic_count"]))
        for shard_id in sorted(receipts):
            receipt = receipts[shard_id]
            self.ledger.receipts[shard_id] = receipt
            self.last_digests[shard_id] = receipt.get("receipt_digest")
        if receipts:
            self.ledger.receipt_order = sorted(receipts)
            self.ledger.rebuild()
        decision["restored_global_version"] = self.global_version
        return decision

    def canonical_report(self) -> dict[str, Any]:
        ledger_audit = self.ledger.audit()
        return {
            "schema": "topic-ledger-audit/v1",
            "config_hash": self.config_hash,
            "global_version": self.global_version,
            "documents": sorted(self.documents.values(), key=lambda doc: doc["doc_id"]),
            "shard_receipts": {
                shard_id: ledger_audit["receipts"][shard_id]
                for shard_id in sorted(ledger_audit["receipts"])
            },
            "authority": ledger_audit["authority"],
            "overwrites": ledger_audit["overwrites"],
            "global_distribution": ledger_audit["global_distribution"],
            "closure": {
                "all_closed": ledger_audit["closure"]["all_closed"],
                "mismatched_topics": ledger_audit["closure"]["mismatched_topics"],
                "frozen_topics": ledger_audit["closure"]["frozen_topics"],
                "by_topic": ledger_audit["closure"]["by_topic"],
            },
            "pending": [item.as_dict() for item in self.pending],
            "invalidation_and_reuse": self.cache_runs,
            "cache_totals": {"reuse_hit_count": self.reuse_hit_count},
            "budget": {
                "limit": int(self.inference["sample_point_budget"]),
                "consumed": self.consumed_budget,
                "remaining": max(0, int(self.inference["sample_point_budget"]) - self.consumed_budget),
            },
            "lease_generation": self.lease_manager.generation,
            "active_leases": self.lease_manager.active_report(),
            "lease_ended": self.lease_manager.ended,
            "voided_receipts": self.voided_receipts,
            "code_counts": dict(sorted(self.code_counts.items())),
            "worker_rejections": [],
        }


class CommandGateway:
    def __init__(self, engine: TopicLedgerEngine) -> None:
        self.engine = engine
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="command-gateway")
        self.latest_summary = engine.progress_summary()

    def handle(self, command: dict[str, Any]) -> dict[str, Any]:
        budget_seconds = int(self.engine.inference["interaction_budget_ms"]) / 1000.0
        future = self.executor.submit(self.engine.handle_command, command)
        future.add_done_callback(self._update_summary)
        try:
            return future.result(timeout=budget_seconds)
        except TimeoutError:
            summary = dict(self.latest_summary)
            summary["stale"] = True
            return {
                "ok": False,
                "code": "E_STALE_SUMMARY",
                "message": "command did not produce a current summary within interaction budget",
                "command_id": command.get("command_id"),
                "stale": True,
                "summary": summary,
                "config_hash": summary.get("config_hash"),
            }

    def _update_summary(self, future: Any) -> None:
        try:
            response = future.result()
            self.latest_summary = response.get("summary", self.latest_summary)
        except Exception:
            pass

    def close(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)
