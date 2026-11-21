"""Pipeline engine: staged execution, budget scheduler, cache reuse, snapshots."""
from __future__ import annotations

import copy
import os
from concurrent.futures import ThreadPoolExecutor

from .cache import CacheLedger, stage_cache_key, task_sort_key
from .errors import E_CONFIG_CHANGED, E_SLOT_BUDGET_PENDING
from .fixedpoint import decimal_to_fixed
from .io_utils import read_json, write_json, write_jsonl
from .recalibration import recalibrate
from .routing import route_domain, route_intent
from .slots import budget_guard, fill_slots


def compute_utterance(sample, config) -> dict:
    """Pure per-utterance computation; deterministic and side-effect free."""
    tokens = sample.tokens
    domain_result = route_domain(tokens, config)
    record = {
        "utterance_id": sample.utterance_id,
        "tokens": list(tokens),
        "gold_domain": sample.domain,
        "gold_intent": sample.intent,
        "domain": domain_result,
        "intent": None,
        "slots": None,
        "recalibration": None,
        "errors": [],
        "status": "domain_failed",
        "reused_stages": [],
        "recomputed_stages": [],
    }
    if domain_result["error"]:
        record["errors"].append(domain_result["error"])
        return record
    domain = domain_result["winner"]
    intent_result = route_intent(tokens, domain, config)
    record["intent"] = intent_result
    if intent_result["error"]:
        record["errors"].append(intent_result["error"])
        record["status"] = "intent_failed"
        return record
    intent = intent_result["winner"]
    slot_result = fill_slots(tokens, intent, config)
    record["slots"] = slot_result
    record["errors"].extend(slot_result["errors"])
    raw_score_fixed = decimal_to_fixed(intent_result["score"], config.scale_bits())
    recal_result = recalibrate(
        raw_score_fixed, len(slot_result["required_slots"]),
        len(slot_result["filled_slots"]),
        len(slot_result["suppressed"]), config)
    record["recalibration"] = recal_result
    record["status"] = "completed"
    return record


class Runner:
    """Runs batches over a mutable sample store with cache and budget ledger."""

    def __init__(self, config, snapshot_dir, mode="incremental",
                 cache: CacheLedger | None = None, verify_reuse: bool = False):
        self.config = config
        self.snapshot_dir = snapshot_dir
        self.mode = mode  # incremental | full_rebuild
        self.verify_reuse = verify_reuse or mode == "full_rebuild"
        self.report_label = mode
        self.cache = cache or CacheLedger()
        self.results = {}
        self.failures = []
        self.todo = []
        self.batch_reports = []
        self.global_version = 0
        self.batch_no = 0
        self.ledger = []
        self._sample_lookup = {}
        os.makedirs(snapshot_dir, exist_ok=True)

    # ------------------------------------------------------------- cache keys
    def _domain_key(self, sample):
        cfg = self.config
        return stage_cache_key(sample.utterance_id, "domain", [
            sample.content_fingerprint(), cfg.routing_fingerprint()])

    def _intent_key(self, sample, routed_domain):
        cfg = self.config
        return stage_cache_key(sample.utterance_id, "intent", [
            sample.content_fingerprint(), routed_domain,
            cfg.intent_mapping_fingerprint(),
            cfg.all_templates_fingerprint()])

    def _slot_key(self, sample, intent):
        cfg = self.config
        return stage_cache_key(sample.utterance_id, "slots", [
            sample.content_fingerprint(), intent,
            cfg.template_fingerprint(intent), cfg.slots_fingerprint()])

    def _recal_key(self, sample, intent, slot_key):
        cfg = self.config
        return stage_cache_key(sample.utterance_id, "recalibration", [
            slot_key, intent, cfg.recal_fingerprint()])

    # ------------------------------------------------------------- invalidation
    def invalidate(self, affected_utterances, stage_names, migrated_intents,
                   routing_changed):
        # Identity-based eviction: derived keys may already reflect a new
        # config, so we drop every cached entry owned by the utterance.
        stage_targets = set(stage_names)
        if routing_changed:
            stage_targets.update({"domain", "intent"})
        for utterance_id in affected_utterances:
            if utterance_id not in self._sample_lookup:
                self.results.pop(utterance_id, None)
                continue
            if "domain" in stage_targets:
                self.cache.domain_cache.invalidate_utterance(utterance_id)
            if "intent" in stage_targets:
                self.cache.intent_cache.invalidate_utterance(utterance_id)
            if "slots" in stage_targets:
                self.cache.slot_cache.invalidate_utterance(utterance_id)
            if "recalibration" in stage_targets:
                self.cache.recal_cache.invalidate_utterance(utterance_id)
        for intent in migrated_intents:
            self.cache.drop_intent(intent)

    # ---------------------------------------------------------------- execution
    def run_batch(self, samples: dict, invalidation: dict) -> dict:
        self._sample_lookup = samples
        self.batch_no += 1
        self.global_version += 1
        cfg_hash = self.config.full_fingerprint()

        budget = self.config.slot_budget_per_batch
        remaining = budget
        ordered_ids = sorted(samples.keys())
        queue = list(self.todo)
        queued = set(queue)
        queue.extend(uid for uid in ordered_ids if uid not in queued)
        self.todo = []

        tasks = []
        for uid in queue:
            sample = samples.get(uid)
            if sample is None:
                self.results.pop(uid, None)
                continue
            tasks.append((uid, sample))

        task_records = self._execute_tasks(tasks)

        completed_this_batch = []
        pending_this_batch = []
        reused = []
        recomputed = []
        for uid, record in task_records:
            reused.extend(f"{uid}:{stage}" for stage in record["reused_stages"])
            recomputed.extend(
                f"{uid}:{stage}" for stage in record["recomputed_stages"])
            sample = samples[uid]
            if record["status"] != "completed":
                self.results[uid] = record
                completed_this_batch.append(uid)
                continue
            cost = record["slots"]["points_consumed"]
            is_reuse = not record["recomputed_stages"]
            if is_reuse:
                # Reused stage outputs must not be re-calibrated or re-charged.
                self.ledger.append({
                    "batch": self.batch_no, "utterance_id": uid,
                    "attempted_cost": 0, "remaining_before": remaining,
                    "deducted": 0, "remaining_after": remaining,
                    "outcome": "reused",
                })
                completed_this_batch.append(uid)
                self.results[uid] = record
                continue
            guard = budget_guard(record["slots"], remaining)
            if guard is not None:
                record["status"] = "budget_pending"
                record["errors"].append(guard)
                record["errors"].append({
                    "code": E_SLOT_BUDGET_PENDING,
                    "message": "carried to next batch todo queue",
                })
                pending_this_batch.append(uid)
                self.todo.append(uid)
                self.ledger.append({
                    "batch": self.batch_no, "utterance_id": uid,
                    "attempted_cost": cost, "remaining_before": remaining,
                    "deducted": 0, "remaining_after": remaining,
                    "outcome": "pending",
                })
                self.results[uid] = record
                continue
            remaining -= cost
            self.ledger.append({
                "batch": self.batch_no, "utterance_id": uid,
                "attempted_cost": cost, "remaining_before": remaining + cost,
                "deducted": cost, "remaining_after": remaining,
                "outcome": "charged",
            })
            completed_this_batch.append(uid)
            self.results[uid] = record
            intent = record["intent"]["winner"]
            self.cache.record_reference(
                intent, uid, record["intent"]["trigger_entries"],
                record["domain"]["winner"])

        report = {
            "batch": self.batch_no,
            "global_version": self.global_version,
            "config_hash": cfg_hash,
            "mode": self.report_label,
            "budget_total": budget,
            "budget_remaining": remaining,
            "completed": completed_this_batch,
            "budget_pending": pending_this_batch,
            "todo_carried": list(self.todo),
            "reused": sorted(reused),
            "recomputed": sorted(recomputed),
            "failure_count": len(self.failures),
        }
        self.batch_reports.append(report)
        if self.batch_no % max(1, self.config.snapshot_commit_interval) == 0:
            self.commit_snapshot()
        return report

    def _execute_tasks(self, tasks):
        max_workers = max(1, self.config.max_workers)
        ordered = sorted(tasks, key=lambda item: task_sort_key(item[0], "utterance_pipeline"))
        if max_workers == 1 or len(ordered) <= 1:
            computed = [self._process_one(item) for item in ordered]
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                computed = list(pool.map(self._process_one, ordered))
        computed.sort(key=lambda item: item[0])
        return computed

    def _process_one(self, item):
        uid, sample = item
        d_key = self._domain_key(sample)
        cached_record = self.results.get(uid)
        retried_pending = cached_record is not None and \
            cached_record.get("status") == "budget_pending"
        can_reuse = False
        if self.mode == "incremental" and cached_record is not None and \
                not retried_pending:
            d_hit = self.cache.domain_cache.get(d_key)
            if d_hit is not None:
                routed_domain = d_hit["winner"]
                if routed_domain is not None:
                    i_key = self._intent_key(sample, routed_domain)
                    i_hit = self.cache.intent_cache.get(i_key)
                    if i_hit is not None:
                        slot_hit = recal_hit = False
                        if i_hit.get("winner") and cached_record.get("slots"):
                            intent = i_hit["winner"]
                            s_key = self._slot_key(sample, intent)
                            r_key = self._recal_key(sample, intent, s_key)
                            slot_hit = self.cache.slot_cache.get(s_key) is not None
                            recal_hit = self.cache.recal_cache.get(r_key) is not None
                        can_reuse = cached_record["status"] != "completed" or (
                            slot_hit and recal_hit)

        if can_reuse:
            if self.verify_reuse:
                fresh = copy.deepcopy(self._full_recompute(sample, d_key))
                baseline = copy.deepcopy(cached_record)
                baseline["reused_stages"] = []
                baseline["recomputed_stages"] = fresh["recomputed_stages"]
                if fresh != baseline:
                    raise AssertionError(
                        f"full rebuild mismatch for {uid}: reuse path disagrees "
                        "with recomputation")
            record = copy.deepcopy(cached_record)
            if record["status"] == "completed":
                record["reused_stages"] = ["domain", "intent", "slots", "recalibration"]
            else:
                record["reused_stages"] = [
                    stage for stage in ("domain", "intent")
                    if cached_record.get(stage) is not None]
            record["recomputed_stages"] = []
            return uid, record

        record = self._full_recompute(sample, d_key)
        return uid, record

    def _full_recompute(self, sample, d_key):
        record = compute_utterance(sample, self.config)
        record["reused_stages"] = []
        record["recomputed_stages"] = ["domain", "intent"]
        self.cache.domain_cache.put(d_key, record["domain"], sample.utterance_id)
        if record["domain"]["winner"] is not None:
            i_key = self._intent_key(sample, record["domain"]["winner"])
            self.cache.intent_cache.put(
                i_key, record["intent"], sample.utterance_id)
        if record["status"] == "completed":
            intent = record["intent"]["winner"]
            s_key = self._slot_key(sample, intent)
            r_key = self._recal_key(sample, intent, s_key)
            self.cache.slot_cache.put(
                s_key, record["slots"], sample.utterance_id)
            self.cache.recal_cache.put(
                r_key, record["recalibration"], sample.utterance_id)
            record["recomputed_stages"].extend(["slots", "recalibration"])
        return record

    # ---------------------------------------------------------------- snapshots
    def snapshot_path(self, name):
        return os.path.join(self.snapshot_dir, name)

    def commit_snapshot(self):
        payload = {
            "global_version": self.global_version,
            "config_hash": self.config.full_fingerprint(),
            "batch_no": self.batch_no,
            "mode": self.report_label,
            "intent_index": self.cache.snapshot_index(),
        }
        write_json(self.snapshot_path("manifest.json"), payload)
        write_jsonl(self.snapshot_path("results.jsonl"),
                    [self.results[uid] for uid in sorted(self.results)])
        write_jsonl(self.snapshot_path("failures.jsonl"), self.failures)
        write_jsonl(self.snapshot_path("budget_ledger.jsonl"), self.ledger)
        write_json(self.snapshot_path("batch_reports.json"), self.batch_reports)

    def verify_manifest_config(self, expected_hash):
        manifest_path = self.snapshot_path("manifest.json")
        if not os.path.exists(manifest_path):
            return
        manifest = read_json(manifest_path)
        if manifest.get("config_hash") != expected_hash:
            raise RuntimeError({
                "code": E_CONFIG_CHANGED,
                "message": "snapshot config hash does not match current config",
                "snapshot_hash": manifest.get("config_hash"),
                "current_hash": expected_hash,
            })
