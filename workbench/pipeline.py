"""Top-level orchestration: edit plans, incremental runs, full rebuilds."""
from __future__ import annotations

import copy
import json

from .cache import CacheLedger
from .editing import EditApplier
from .engine import Runner
from .io_utils import canonical_json, fingerprint, write_json
from .loader import load_samples, load_vocabulary

DOMAIN_SECTIONS = ("domain_word_bags",)
INTENT_MAPPING_SECTIONS = ("domain_intents",)
SLOT_SECTIONS = ("slot_definitions", "intent_required_slots", "slot_match_window")
RECAL_SECTIONS = ("recalibration", "quantization_levels",
                  "active_quantization_level")


def _fp_sections(keys, cfg):
    return fingerprint({key: cfg.get(key) for key in keys})


def build_invalidation(before_cfg, after_cfg, sample_affected, migrated_intents,
                       slot_intents, threshold_info, cache, all_utterance_ids):
    """Derive the precise stage invalidation guide for one edit batch."""
    domain_changed = _fp_sections(DOMAIN_SECTIONS, before_cfg) != \
        _fp_sections(DOMAIN_SECTIONS, after_cfg)
    mapping_changed = _fp_sections(INTENT_MAPPING_SECTIONS, before_cfg) != \
        _fp_sections(INTENT_MAPPING_SECTIONS, after_cfg)
    slots_changed = _fp_sections(SLOT_SECTIONS, before_cfg) != \
        _fp_sections(SLOT_SECTIONS, after_cfg)
    recal_changed = _fp_sections(RECAL_SECTIONS, before_cfg) != \
        _fp_sections(RECAL_SECTIONS, after_cfg)

    stages = []
    if domain_changed:
        stages.append("domain")
    if mapping_changed:
        stages.append("intent")
    if slots_changed:
        stages.append("slots")
    if recal_changed or slots_changed:
        stages.append("recalibration")

    affected = set(sample_affected)
    affected.update(cache.referenced_utterances(migrated_intents | slot_intents))
    level = threshold_info.get("level")
    if threshold_info.get("changed") and level == "domain":
        affected.update(all_utterance_ids)
        stages = ["domain", "intent", "slots", "recalibration"]
    elif threshold_info.get("changed") and level == "intent":
        affected.update(all_utterance_ids)
        stages = list(dict.fromkeys(["intent", "slots", "recalibration"]))
    elif level in ("recalibration", "quantization"):
        affected.update(all_utterance_ids)
        stages = list(dict.fromkeys(stages + ["recalibration"]))
    return {
        "affected_utterances": affected,
        "stages": list(dict.fromkeys(stages)),
        "migrated_intents": set(migrated_intents),
        "routing_changed": domain_changed,
    }


class Pipeline:
    def __init__(self, vocab_path: str, initial_config_path: str,
                 initial_samples_path: str, snapshot_dir: str):
        self.vocab = load_vocabulary(vocab_path)
        with open(initial_config_path, encoding="utf-8") as handle:
            self.raw_config = copy.deepcopy(json.load(handle))
        self.applier = EditApplier(self.raw_config, self.vocab)
        self.config = self.applier.current_config()
        self.samples, initial_failures = load_samples(
            initial_samples_path, self.vocab, self.config)
        self.seen_ids = set(self.samples)
        self.snapshot_dir = snapshot_dir
        self.runner = Runner(self.config, snapshot_dir, mode="incremental")
        self.runner.failures.extend(initial_failures)

    def _edit_batch(self, batch: dict):
        before_cfg = copy.deepcopy(self.applier.raw)
        sample_affected = self.applier.apply_samples(
            batch, self.samples, self.seen_ids, self.runner.failures)
        migrated_intents = self.applier.apply_migrations(batch)
        slot_intents = self.applier.apply_slot_edits(batch)
        threshold_info = self.applier.apply_threshold_edits(batch)
        return before_cfg, sample_affected, migrated_intents, slot_intents, \
            threshold_info

    def _guide(self, before_cfg, sample_affected, migrated_intents,
               slot_intents, threshold_info):
        return build_invalidation(
            before_cfg, self.applier.raw, sample_affected,
            migrated_intents, slot_intents, threshold_info,
            self.runner.cache, set(self.samples.keys()))

    def run_plan(self, plan: dict) -> dict:
        summaries = []
        summaries.append(self.runner.run_batch(self.samples, {
            "affected_utterances": set(self.samples),
            "stages": ["domain", "intent", "slots", "recalibration"],
            "migrated_intents": set(),
            "routing_changed": True,
        }))
        for batch in plan.get("batches", []):
            before_cfg, sample_affected, migrated, slot_intents, threshold_info = \
                self._edit_batch(batch)
            # Invalidation guide derives from old-vs-new fingerprints, but the
            # cache entries themselves are dropped here while keys still reflect
            # the prior config.
            invalidation = self._guide(before_cfg, sample_affected, migrated,
                                       slot_intents, threshold_info)
            self.runner.invalidate(
                set(invalidation["affected_utterances"]),
                invalidation["stages"],
                set(invalidation["migrated_intents"]),
                invalidation["routing_changed"],
            )
            self.config = self.applier.current_config()
            self.runner.config = self.config
            summaries.append(self.runner.run_batch(self.samples, invalidation))
        self.runner.commit_snapshot()
        return {"summaries": summaries, "runner": self.runner}


def run_full_rebuild(vocab_path: str, initial_config_path: str,
                     initial_samples_path: str, plan: dict,
                     rebuild_snapshot_dir: str) -> Runner:
    """One-shot full rebuild: same edit sequence, empty cache, identical outputs."""
    vocab = load_vocabulary(vocab_path)
    with open(initial_config_path, encoding="utf-8") as handle:
        raw_config = copy.deepcopy(json.load(handle))
    applier = EditApplier(raw_config, vocab)
    config = applier.current_config()
    samples, failures = load_samples(initial_samples_path, vocab, config)
    seen = set(samples)
    runner = Runner(config, rebuild_snapshot_dir, mode="incremental",
                    cache=CacheLedger(), verify_reuse=True)
    runner.report_label = "full_rebuild"
    runner.failures.extend(failures)

    runner.run_batch(samples, {
        "affected_utterances": set(samples),
        "stages": ["domain", "intent", "slots", "recalibration"],
        "migrated_intents": set(),
        "routing_changed": True,
    })
    for batch in plan.get("batches", []):
        before_cfg = copy.deepcopy(applier.raw)
        sample_affected = applier.apply_samples(
            batch, samples, seen, runner.failures)
        migrated = applier.apply_migrations(batch)
        slot_intents = applier.apply_slot_edits(batch)
        threshold_info = applier.apply_threshold_edits(batch)
        invalidation = build_invalidation(
            before_cfg, applier.raw, sample_affected, migrated, slot_intents,
            threshold_info, runner.cache, set(samples.keys()))
        runner.invalidate(
            invalidation["affected_utterances"], invalidation["stages"],
            invalidation["migrated_intents"], invalidation["routing_changed"])
        runner.config = applier.current_config()
        runner.run_batch(samples, invalidation)
    runner.commit_snapshot()
    return runner


def compare_runners(incremental: Runner, rebuilt: Runner) -> dict:
    """Field-by-field comparison required by acceptance criterion 7."""
    mismatches = []
    inc_ids = sorted(incremental.results)
    reb_ids = sorted(rebuilt.results)
    if inc_ids != reb_ids:
        mismatches.append({"field": "utterance_id_set",
                           "incremental": inc_ids, "rebuilt": reb_ids})
    for uid in inc_ids:
        if uid not in rebuilt.results:
            continue
        left = incremental.results[uid]
        right = rebuilt.results[uid]
        if canonical_json(left) != canonical_json(right):
            for key in left:
                if canonical_json(left.get(key)) != canonical_json(right.get(key)):
                    mismatches.append({"field": f"{uid}.{key}",
                                       "incremental": left.get(key),
                                       "rebuilt": right.get(key)})
    if canonical_json(incremental.failures) != canonical_json(rebuilt.failures):
        mismatches.append({"field": "failures",
                           "incremental": incremental.failures,
                           "rebuilt": rebuilt.failures})
    if canonical_json(incremental.ledger) != canonical_json(rebuilt.ledger):
        mismatches.append({"field": "budget_ledger",
                           "incremental": incremental.ledger,
                           "rebuilt": rebuilt.ledger})

    def _strip_mode(reports):
        return [{k: v for k, v in report.items() if k != "mode"} for report in reports]

    if canonical_json(_strip_mode(incremental.batch_reports)) != \
            canonical_json(_strip_mode(rebuilt.batch_reports)):
        mismatches.append({"field": "batch_reports",
                           "incremental": incremental.batch_reports,
                           "rebuilt": rebuilt.batch_reports})
    return {"identical": not mismatches, "mismatches": mismatches}


def write_comparison_report(path: str, report: dict) -> None:
    write_json(path, report)
