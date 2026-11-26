from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from typing import Any

from .commands import apply_config_command
from .config import normalize_config
from .hashing import canonical_hash
from .index import (
    build_index,
    delete_document,
    empty_index,
    index_config_fingerprint,
    serialize_index,
    upsert_document,
)
from .scheduler import begin_batch, new_scheduler_state, plan_schedule
from .scoring import candidate_plan, compute_fusion_cache_key, score_query
from .store import ArtifactStore, SnapshotStore, atomic_write_json
from .vocab import Vocabulary


def prepare_config(raw_config: dict[str, Any], vocab: Vocabulary) -> dict[str, Any]:
    config = normalize_config(raw_config)
    config["vocab_fingerprint"] = vocab.fingerprint
    config["index_config_fingerprint"] = index_config_fingerprint(config)
    return config


def _score_miss(query: dict[str, Any], index: dict[str, Any], config: dict[str, Any], vocab: Vocabulary):
    return score_query(query, index, config, vocab)


class Engine:
    def __init__(
        self,
        config: dict[str, Any],
        vocab: Vocabulary,
        queries: list[dict[str, Any]] | None = None,
        *,
        root: str | Path | None = None,
        input_failures: list[dict[str, Any]] | None = None,
    ):
        self.config = config
        self.vocab = vocab
        self.queries = list(queries or [])
        self.query_by_id = {query["query_id"]: query for query in self.queries}
        self.root = Path(root) if root is not None else None
        self.last_committed_version = 0
        self.last_committed_config_hash: str | None = None
        self.store = ArtifactStore(self.root / "cache") if self.root is not None else None
        self.snapshots = SnapshotStore(self.root) if self.root is not None else None
        if self.snapshots is not None:
            pointer = self.snapshots.current_pointer()
            if pointer:
                self.last_committed_version = int(pointer["global_version"])
                self.last_committed_config_hash = pointer.get("config_hash")
        self.documents: dict[str, dict[str, Any]] = {}
        self.index = empty_index(config, vocab)
        self.results: dict[str, dict[str, Any]] = {}
        self.global_version = 0
        self.failures: list[dict[str, Any]] = list(input_failures or [])
        self.command_history: list[dict[str, Any]] = []
        self.business_codes: dict[str, int] = {}
        for failure in self.failures:
            self.business_codes[failure["code"]] = self.business_codes.get(failure["code"], 0) + 1
        self.scheduler_state = new_scheduler_state(
            int(config["budget"]["batch_points"]),
            int(config["budget"]["query_quota_points"]),
        )
        self.scheduler_history: list[dict[str, Any]] = []
        self.last_run: dict[str, Any] = self._empty_run_ledger()
        self.last_edit_scope: dict[str, Any] | None = None
        self._schedule_dirty = False

    def _empty_run_ledger(self) -> dict[str, Any]:
        return {
            "workers": 1,
            "buckets_recomputed": [],
            "buckets_reused": [],
            "buckets_removed": [],
            "field_tf_recomputed": [],
            "field_tf_reused": [],
            "fusion_recomputed": [],
            "fusion_reused": [],
            "rankings_recomputed": [],
            "rankings_reused": [],
            "ready": [],
            "quota_failures": [],
            "pending": [],
        }

    def cold_build(self, documents: list[dict[str, Any]], *, global_version: int = 1) -> dict[str, Any]:
        self.documents = {doc["doc_id"]: doc for doc in sorted(documents, key=lambda item: item["doc_id"])}
        self.index = build_index(list(self.documents.values()), self.config, self.vocab)
        self.global_version = global_version
        self.results = {}
        ledger = self._empty_run_ledger()
        ledger["buckets_recomputed"] = list(self.index["bucket_order"])
        ledger["field_tf_recomputed"] = sorted(self.index["field_tf"])
        if self.store is not None:
            for bucket_id in self.index["bucket_order"]:
                self.store.write_bucket(self.index["buckets"][bucket_id])
            for record_id in sorted(self.index["field_tf"]):
                self.store.write_field_tf(self.index["field_tf"][record_id])
        self.last_run = ledger
        self.last_edit_scope = {
            "op": "cold_build",
            "buckets_recomputed": list(self.index["bucket_order"]),
            "buckets_reused": [],
            "buckets_removed": [],
        }
        return self.last_edit_scope

    def _bucket_reuse_scope(self, delta: dict[str, list[str]]) -> dict[str, list[str]]:
        recomputed = set(delta.get("recomputed_buckets", []))
        removed = set(delta.get("removed_buckets", []))
        reused: list[str] = []
        actually_recomputed = list(delta.get("recomputed_buckets", []))
        for bucket_id in self.index["bucket_order"]:
            if bucket_id in recomputed:
                bucket = self.index["buckets"][bucket_id]
                if self.store is not None:
                    self.store.write_bucket(bucket)
                continue
            bucket = self.index["buckets"][bucket_id]
            if self.store is not None and not self.store.has_bucket(
                bucket_id, bucket["bucket_fingerprint"]
            ):
                self.store.write_bucket(bucket)
                actually_recomputed.append(bucket_id)
            else:
                reused.append(bucket_id)
        if self.store is not None:
            for bucket_id in removed:
                self.store.remove_bucket(bucket_id)

        recomputed_fields = set(delta.get("field_tf_recomputed", []))
        removed_fields = set(delta.get("field_tf_removed", []))
        field_reused: list[str] = []
        field_recomputed = list(delta.get("field_tf_recomputed", []))
        for record_id in sorted(self.index["field_tf"]):
            record = self.index["field_tf"][record_id]
            if record_id in recomputed_fields:
                if self.store is not None:
                    self.store.write_field_tf(record)
                continue
            if self.store is not None and not self.store.has_field_tf(
                record_id, record["cache_key"]
            ):
                self.store.write_field_tf(record)
                field_recomputed.append(record_id)
            else:
                field_reused.append(record_id)
        if self.store is not None:
            for record_id in removed_fields:
                self.store.remove_field_tf(record_id)

        scope = {
            "buckets_recomputed": sorted(set(actually_recomputed)),
            "buckets_reused": sorted(reused),
            "buckets_removed": sorted(removed),
            "field_tf_recomputed": sorted(set(field_recomputed)),
            "field_tf_reused": sorted(field_reused),
            "field_tf_removed": sorted(removed_fields),
        }
        self.last_run["buckets_recomputed"] = scope["buckets_recomputed"]
        self.last_run["buckets_reused"] = scope["buckets_reused"]
        self.last_run["buckets_removed"] = scope["buckets_removed"]
        self.last_run["field_tf_recomputed"] = scope["field_tf_recomputed"]
        self.last_run["field_tf_reused"] = scope["field_tf_reused"]
        self.last_edit_scope = scope
        return scope

    def apply_command(self, command: dict[str, Any]) -> dict[str, Any]:
        op = command["op"]
        line = command.get("line")
        self._schedule_dirty = True
        outcome: dict[str, Any] = {
            "line": line,
            "op": op,
            "status": "applied",
            "doc_id": command.get("doc_id") or command.get("document", {}).get("doc_id"),
        }
        if op in {"set_language_weight", "set_field_weight", "set_saturation_k", "set_quant_tier", "set_frac_bits", "set_top_k"}:
            apply_config_command(self.config, command)
            self.index["config_fingerprint"] = self.config["full_config_fingerprint"]
            self.global_version += 1
            outcome["config_fingerprint"] = self.config["full_config_fingerprint"]
            outcome["scope"] = {
                "buckets_recomputed": [],
                "buckets_reused": list(self.index["bucket_order"]),
                "buckets_removed": [],
            }
            self.last_edit_scope = outcome["scope"]
            self.command_history.append(outcome)
            return outcome

        if op == "delete_document":
            doc_id = command["doc_id"]
            if doc_id not in self.documents:
                failure = {
                    "code": "E_INPUT_INVALID",
                    "source": "commands",
                    "line": line,
                    "item_id": doc_id,
                    "message": f"cannot delete missing doc_id: {doc_id}",
                }
                self.failures.append(failure)
                self.business_codes[failure["code"]] = self.business_codes.get(failure["code"], 0) + 1
                outcome.update({"status": "skipped", "failure": failure})
                self.command_history.append(outcome)
                return outcome
            delta = delete_document(self.index, doc_id, self.config, self.vocab)
            del self.documents[doc_id]
            scope = self._bucket_reuse_scope(delta or {})
            self.global_version += 1
            outcome["scope"] = scope
            self.command_history.append(outcome)
            return outcome

        document = command["document"]
        doc_id = document["doc_id"]
        exists = doc_id in self.documents
        if op == "add_document" and exists:
            failure = {
                "code": "E_INPUT_INVALID",
                "source": "commands",
                "line": line,
                "item_id": doc_id,
                "message": f"cannot add existing doc_id: {doc_id}",
            }
            self.failures.append(failure)
            self.business_codes[failure["code"]] = self.business_codes.get(failure["code"], 0) + 1
            outcome.update({"status": "skipped", "failure": failure})
            self.command_history.append(outcome)
            return outcome
        if op == "replace_document" and not exists:
            failure = {
                "code": "E_INPUT_INVALID",
                "source": "commands",
                "line": line,
                "item_id": doc_id,
                "message": f"cannot replace missing doc_id: {doc_id}",
            }
            self.failures.append(failure)
            self.business_codes[failure["code"]] = self.business_codes.get(failure["code"], 0) + 1
            outcome.update({"status": "skipped", "failure": failure})
            self.command_history.append(outcome)
            return outcome

        delta = upsert_document(self.index, document, self.config, self.vocab)
        self.documents[doc_id] = document
        scope = self._bucket_reuse_scope(delta)
        self.global_version += 1
        outcome["replaced_existing"] = exists
        outcome["scope"] = scope
        self.command_history.append(outcome)
        return outcome

    def apply_commands(self, commands: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [self.apply_command(command) for command in sorted(commands, key=lambda item: item.get("line", 0))]

    def run_batch(
        self,
        *,
        batch_points: int | None = None,
        query_quota_points: int | None = None,
        unlimited: bool = False,
        workers: int = 1,
        fresh_cycle: bool = False,
    ) -> dict[str, Any]:
        self.last_run = self._empty_run_ledger()
        self.last_run["workers"] = workers
        if self.last_edit_scope is not None:
            for key in (
                "buckets_recomputed",
                "buckets_reused",
                "buckets_removed",
                "field_tf_recomputed",
                "field_tf_reused",
                "field_tf_removed",
            ):
                self.last_run[key] = list(self.last_edit_scope.get(key, []))
        if fresh_cycle or self._schedule_dirty:
            self.scheduler_state["todo"] = []
            self.scheduler_state["completed"] = []
            self.scheduler_state["quota_exceeded"] = []
            self._schedule_dirty = False
        if unlimited:
            batch_points = 10**18
            query_quota_points = 10**18
        quota = int(query_quota_points if query_quota_points is not None else self.config["budget"]["query_quota_points"])
        batch = int(batch_points if batch_points is not None else self.config["budget"]["batch_points"])
        if unlimited:
            self.scheduler_state["query_quota_points"] = quota
        begin_batch(self.scheduler_state, batch)
        self.scheduler_state["query_quota_points"] = quota

        plans: dict[str, dict[str, Any]] = {}
        costs: dict[str, int] = {}
        fusion_keys: dict[str, str] = {}
        for query in self.queries:
            plan = candidate_plan(query, self.index, self.config)
            plans[query["query_id"]] = plan
            costs[query["query_id"]] = plan["cost_points"]
            fusion_keys[query["query_id"]] = compute_fusion_cache_key(
                query, self.index, plan, self.config
            )

        schedule = plan_schedule(
            self.scheduler_state,
            [query["query_id"] for query in self.queries],
            costs,
            query_quota_points=quota,
        )
        self.scheduler_history.append(deepcopy(schedule))
        self.last_run["ready"] = schedule["ready"]
        self.last_run["quota_failures"] = schedule["quota_failures"]
        self.last_run["pending"] = schedule["pending"]
        for event in schedule["quota_failures"] + schedule["pending"]:
            self.business_codes[event["code"]] = self.business_codes.get(event["code"], 0) + 1

        for event in schedule["quota_failures"]:
            self.results.pop(event["query_id"], None)
        for event in schedule["pending"]:
            self.results.pop(event["query_id"], None)

        cached_results: dict[str, dict[str, Any]] = {}
        missing: list[tuple[str, dict[str, Any]]] = []
        for job in schedule["ready"]:
            qid = job["query_id"]
            cache_key = fusion_keys[qid]
            cached = self.store.read_fusion(cache_key) if self.store is not None and self.store.has_fusion(cache_key) else None
            if cached is not None and cached.get("fusion_cache_key") == cache_key:
                cached_results[qid] = cached
                self.last_run["fusion_reused"].append(qid)
                ranking_artifact = (
                    self.store.read_ranking(cached["ranking_cache_key"])
                    if self.store is not None
                    else None
                )
                if (
                    ranking_artifact is not None
                    and ranking_artifact.get("ranking_cache_key") == cached["ranking_cache_key"]
                ):
                    self.last_run["rankings_reused"].append(qid)
                else:
                    if self.store is not None:
                        self.store.write_ranking(
                            qid,
                            cached["ranking_cache_key"],
                            {
                                "selected": cached["selected"],
                                "boundary": cached["boundary"],
                                "rows": cached["rows"],
                            },
                        )
                    self.last_run["rankings_recomputed"].append(qid)
            else:
                missing.append((qid, self.query_by_id[qid]))

        computed_results: dict[str, dict[str, Any]] = {}
        if missing:
            if workers and workers > 1:
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = [
                        executor.submit(_score_miss, query, self.index, self.config, self.vocab)
                        for _, query in missing
                    ]
                    for (qid, _), future in zip(missing, futures):
                        computed_results[qid] = future.result()
            else:
                for qid, query in missing:
                    computed_results[qid] = _score_miss(query, self.index, self.config, self.vocab)

        for qid in sorted(computed_results):
            result = computed_results[qid]
            self.results[qid] = result
            self.last_run["fusion_recomputed"].append(qid)
            self.last_run["rankings_recomputed"].append(qid)
            if self.store is not None:
                self.store.write_fusion(result)
                self.store.write_ranking(
                    qid,
                    result["ranking_cache_key"],
                    {
                        "selected": result["selected"],
                        "boundary": result["boundary"],
                        "rows": result["rows"],
                    },
                )

        for qid, result in cached_results.items():
            self.results[qid] = result

        for key in (
            "fusion_reused",
            "fusion_recomputed",
            "rankings_reused",
            "rankings_recomputed",
        ):
            self.last_run[key] = sorted(set(self.last_run[key]))

        report = self.build_report(plans=plans, fusion_keys=fusion_keys, schedule=schedule)
        return report

    def _aggregate_candidates(self) -> tuple[dict[str, int], dict[str, int]]:
        unique_by_view: dict[str, set[str]] = {view: set() for view in self.config["view_names"]}
        hits = {view: 0 for view in self.config["view_names"]}
        for result in self.results.values():
            for view, doc_ids in result.get("candidate_docs_by_view", {}).items():
                unique_by_view.setdefault(view, set()).update(doc_ids)
                hits[view] = hits.get(view, 0) + int(result.get("candidate_count_by_view", {}).get(view, 0))
        unique_counts = {view: len(doc_ids) for view, doc_ids in sorted(unique_by_view.items())}
        return unique_counts, hits

    def build_report(
        self,
        *,
        plans: dict[str, dict[str, Any]] | None = None,
        fusion_keys: dict[str, str] | None = None,
        schedule: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if plans is None:
            plans = {
                query["query_id"]: candidate_plan(query, self.index, self.config)
                for query in self.queries
            }
        if fusion_keys is None:
            fusion_keys = {
                qid: compute_fusion_cache_key(self.query_by_id[qid], self.index, plans[qid], self.config)
                for qid in plans
            }
        candidate_unique, candidate_hits = self._aggregate_candidates()
        boundary_items: list[dict[str, Any]] = []
        for qid in sorted(self.results):
            result = self.results[qid]
            for boundary_doc in result.get("boundary", {}).get("boundary_docs", []):
                boundary_items.append({"query_id": qid, **boundary_doc})
        cache_key_entries: list[dict[str, str]] = []
        for record_id in sorted(self.index["field_tf"]):
            record = self.index["field_tf"][record_id]
            cache_key_entries.append(
                {"kind": "field_tf", "id": record_id, "cache_key": record["cache_key"]}
            )
        for bucket_id in self.index["bucket_order"]:
            bucket = self.index["buckets"][bucket_id]
            cache_key_entries.append({"kind": "bucket", "id": bucket_id, "cache_key": bucket["cache_key"]})
        for qid in sorted(fusion_keys):
            cache_key_entries.append({"kind": "fusion", "id": qid, "cache_key": fusion_keys[qid]})
            if qid in self.results:
                cache_key_entries.append(
                    {"kind": "ranking", "id": qid, "cache_key": self.results[qid]["ranking_cache_key"]}
                )
        reused_count = sum(
            len(self.last_run.get(key, []))
            for key in (
                "buckets_reused",
                "field_tf_reused",
                "fusion_reused",
                "rankings_reused",
            )
        )
        recomputed_count = sum(
            len(self.last_run.get(key, []))
            for key in (
                "buckets_recomputed",
                "field_tf_recomputed",
                "fusion_recomputed",
                "rankings_recomputed",
            )
        )
        report = {
            "global_version": self.global_version,
            "config_hash": self.config["full_config_fingerprint"],
            "inverted_ledger": [
                self.index["buckets"][bucket_id]
                for bucket_id in self.index["bucket_order"]
            ],
            "config_fingerprints": {
                "config": self.config["config_fingerprint"],
                "language_weights": self.config["language_weights_fingerprint"],
                "fixed_point": self.config["fixed_point_fingerprint"],
                "index_schema": self.config["index_config_fingerprint"],
                "full": self.config["full_config_fingerprint"],
            },
            "counts": {
                "documents": self.index["document_count"],
                "queries_loaded": len(self.queries),
                "queries_with_results": len(self.results),
                "term_buckets": len(self.index["buckets"]),
                "boundary_entries": len(boundary_items),
                "boundary_unique_documents": len({item["doc_id"] for item in boundary_items}),
                "reused_entries": reused_count,
                "recomputed_entries": recomputed_count,
                "input_failures": len(self.failures),
            },
            "candidate_counts_by_view": candidate_unique,
            "candidate_hits_by_query_and_view": {
                qid: self.results[qid]["candidate_count_by_view"]
                for qid in sorted(self.results)
            },
            "candidate_hits_by_view_sum": candidate_hits,
            "boundary_documents": boundary_items,
            "business_codes": dict(sorted(self.business_codes.items())),
            "failures": self.failures,
            "scheduler": {
                "state": self.scheduler_state,
                "current_plan": schedule,
            },
            "last_run": self.last_run,
            "last_edit_scope": self.last_edit_scope,
            "cache_keys": cache_key_entries,
            "query_summaries": [
                {
                    "query_id": qid,
                    "selected": self.results[qid]["selected"],
                    "rows": self.results[qid]["rows"],
                    "boundary_doc_ids": self.results[qid]["boundary"]["boundary_doc_ids"],
                    "fusion_cache_key": self.results[qid]["fusion_cache_key"],
                    "ranking_cache_key": self.results[qid]["ranking_cache_key"],
                    "cost_points": self.results[qid]["cost_points"],
                    "referenced_buckets": self.results[qid]["referenced_buckets"],
                }
                for qid in sorted(self.results)
            ],
            "shutdown": self.snapshots.shutdown_status() if self.snapshots is not None else {
                "cleanup_status": "no_snapshot_store",
                "can_exit": True,
            },
        }
        report["report_fingerprint"] = canonical_hash(report)
        return report

    def parity_material(self) -> dict[str, Any]:
        report = self.build_report()
        return {
            "global_version": self.global_version,
            "config": self.config,
            "documents": [self.documents[doc_id] for doc_id in sorted(self.documents)],
            "index": serialize_index(self.index),
            "bucket_ledger": [
                self.index["buckets"][bucket_id] for bucket_id in self.index["bucket_order"]
            ],
            "field_tf_records": [
                self.index["field_tf"][record_id] for record_id in sorted(self.index["field_tf"])
            ],
            "query_results": [self.results[qid] for qid in sorted(self.results)],
            "cache_keys": report["cache_keys"],
            "failures": self.failures,
            "command_history": [
                {key: value for key, value in item.items() if key != "scope"}
                for item in self.command_history
            ],
            "counts": {
                key: value
                for key, value in report["counts"].items()
                if key not in {"reused_entries", "recomputed_entries"}
            },
            "candidate_counts_by_view": report["candidate_counts_by_view"],
            "boundary_documents": report["boundary_documents"],
            "business_codes": report["business_codes"],
            "scheduler_state": self.scheduler_state,
        }

    def write_snapshot(self, report: dict[str, Any] | None = None) -> dict[str, Any]:
        if self.snapshots is None or self.root is None:
            raise RuntimeError("Engine was created without a persistent root")
        report = report or self.build_report()
        parity = self.parity_material()
        state_hash = canonical_hash(parity)
        transaction = self.snapshots.new_transaction()
        self.snapshots.write_phase(
            transaction,
            "01_validation",
            {"failures": self.failures, "business_codes": self.business_codes},
        )
        self.snapshots.write_phase(transaction, "02_index", serialize_index(self.index))
        self.snapshots.write_phase(
            transaction,
            "03_statistics",
            {
                "bucket_count": len(self.index["buckets"]),
                "bucket_order": self.index["bucket_order"],
                "candidate_counts_by_view": report["candidate_counts_by_view"],
            },
        )
        self.snapshots.write_phase(
            transaction,
            "04_scores",
            {
                qid: {
                    "rows": result["rows"],
                    "evidence": result["evidence"],
                    "fusion_cache_key": result["fusion_cache_key"],
                    "intermediate_fingerprint": result["intermediate_fingerprint"],
                }
                for qid, result in sorted(self.results.items())
            },
        )
        self.snapshots.write_phase(
            transaction,
            "05_ranking",
            {
                qid: {
                    "selected": result["selected"],
                    "boundary": result["boundary"],
                    "ranking_cache_key": result["ranking_cache_key"],
                }
                for qid, result in sorted(self.results.items())
            },
        )
        self.snapshots.write_phase(transaction, "scheduler", self.scheduler_state)
        self.snapshots.write_phase(transaction, "report", report)
        pointer = self.snapshots.commit(
            transaction,
            global_version=self.global_version,
            config_hash=self.config["full_config_fingerprint"],
            state_hash=state_hash,
            expected_global_version=self.last_committed_version,
            expected_config_hash=self.last_committed_config_hash,
        )
        self.last_committed_version = int(pointer["global_version"])
        self.last_committed_config_hash = pointer["config_hash"]
        report["snapshot_pointer"] = pointer
        report["shutdown"] = self.snapshots.shutdown_status()
        atomic_write_json(self.root / "ledger.json", report)
        return pointer


def full_rebuild_engine(
    config: dict[str, Any],
    vocab: Vocabulary,
    queries: list[dict[str, Any]],
    documents: list[dict[str, Any]],
    *,
    root: str | Path | None = None,
    input_failures: list[dict[str, Any]] | None = None,
    global_version: int = 1,
    workers: int = 1,
) -> Engine:
    engine = Engine(config, vocab, queries, root=root, input_failures=input_failures)
    engine.cold_build(documents, global_version=global_version)
    engine.run_batch(unlimited=True, workers=workers)
    return engine


def normalized_config_from_file(path: str | Path, vocab: Vocabulary) -> dict[str, Any]:
    import json

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return prepare_config(raw, vocab)
