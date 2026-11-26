"""End-to-end deterministic evidence pipeline with incremental reuse."""

import json
import os
import threading
import time

from . import aggregation, calibration, error_track, evidence, features
from . import fixedpoint as fp
from .concurrency import BoundedThreadPool
from .fingerprint import (config_fingerprint, file_fingerprint,
                          label_fingerprint, structure_fingerprint)
from .ledger import (Ledger, calibration_key, segment_key,
                     semantic_input_fp, window_key, word_key)
from .loading import load_config, load_sample
from .scheduler import BatchScheduler
from .snapshot import SnapshotStore


class ShutdownFlag:
    """Thread-safe cooperative shutdown flag set by signal handlers."""

    def __init__(self):
        self._event = threading.Event()

    def request(self, reason="signal"):
        self._event.reason = reason
        self._event.set()

    @property
    def requested(self):
        return self._event.is_set()


class Pipeline:
    def __init__(self, sample_dir, config_path, snapshot_dir,
                 tier_name="coarse", max_batches=None, shutdown=None):
        sample = load_sample(sample_dir)
        self.config = load_config(config_path)
        self.transcript = sample["transcript"]
        self.wav_path = sample["wav_path"]
        self.transcript_path = sample["transcript_path"]
        self.tier = fp.get_tier(tier_name)
        self.store = SnapshotStore(snapshot_dir)
        prior = self.store.read_json("ledger.json")
        self.ledger_path = os.path.join(
            self.store.root, "pending_ledger.json")
        self.ledger = Ledger(self.ledger_path)
        if prior is not None:
            self.ledger.entries = prior.get("entries", {})
            self.ledger.audit = prior.get("audit", [])
        self.scheduler = BatchScheduler(self.config, self.ledger)
        self.max_batches = max_batches
        self.shutdown = shutdown or ShutdownFlag()
        self.audit = {"reused": [], "invalidated": [], "computed": []}

    # ------------------------------------------------------------------
    def fingerprints(self):
        cfg_fp = config_fingerprint(self.config)
        audio_fp = file_fingerprint(self.wav_path)
        label_fp = label_fingerprint(self.transcript)
        struct_fp = structure_fingerprint(self.transcript)
        return {"config": cfg_fp, "audio": audio_fp,
                "label": label_fp, "structure": struct_fp}

    def compute_windows(self, pool):
        from .pcmio import read_pcm
        cfg_fp = self._fps["config"]
        pcm = read_pcm(self.wav_path)
        skeleton = features.build_windows(
            pcm, self.transcript, self.config, self.tier)
        reusable_windows = []
        fresh_windows = []
        for window in skeleton:
            key = window_key(window, self._fps["audio"], cfg_fp,
                             self.tier.name)
            window["cache_key"] = key
            if self.ledger.reusable(key):
                reusable_windows.append(window)
            else:
                fresh_windows.append(window)

        if fresh_windows:
            def _compute_one(target):
                rebuilt = features.build_windows(
                    pcm, self.transcript, self.config, self.tier)
                match = next(item for item in rebuilt
                             if item["window_id"] == target["window_id"])
                return match["features"]

            jobs = [(_compute_one, (window,), {})
                    for window in fresh_windows]
            results = pool.map_ordered(jobs)
            for (_, value, error), window in zip(results, fresh_windows):
                if error is not None:
                    raise error
                window["features"] = value

        for window in fresh_windows:
            self.ledger.record(window["cache_key"], "window",
                               window["segment_refs"], "valid",
                               "freshly computed",
                               {"features": window["features"]})
            self.audit["computed"].append(window["cache_key"])
        for window in reusable_windows:
            entry = self.ledger.reuse(window["cache_key"])
            window["features"] = entry["artifact"]["features"]
            self.audit["reused"].append(window["cache_key"])
        return skeleton

    def _pcm_for(self, window):
        from .pcmio import read_pcm
        if not hasattr(self, "_pcm"):
            self._pcm = read_pcm(self.wav_path)
        return self._pcm

    # ------------------------------------------------------------------
    def compute_words(self, windows, runnable_segments):
        sequence = evidence.global_word_sequence(self.transcript)
        position_map = {(item["segment_id"], item["word_index"]): index
                        for index, item in enumerate(sequence)}
        window_key_by_id = {window["window_id"]: window["cache_key"]
                            for window in windows}
        records = []
        for seg in self.transcript["segments"]:
            if seg["segment_id"] not in runnable_segments:
                continue
            for index, word in enumerate(seg["words"]):
                word_ref = {"segment_id": seg["segment_id"],
                            "word_index": index}
                position = position_map[(seg["segment_id"], index)]
                sem_fp = semantic_input_fp(sequence, position, self.config)
                intermediate = self.tier if self.tier.name == \
                    fp.REFERENCE.name else fp.FINE
                record = evidence.build_word_evidence(
                    word_ref, word, windows, sequence, position,
                    self.config, intermediate)
                evidence.quantize_word_evidence(
                    record, intermediate, self.tier)
                win_keys = [window_key_by_id[win_id]
                            for win_id in record["window_ids"]]
                key = word_key(
                    self._fps["structure"], sem_fp, win_keys,
                    record["token"], record["emotion"],
                    self._fps["config"], self.tier.name, self._fps["label"])
                record["cache_key"] = key
                record["semantic_fp"] = sem_fp
                records.append(record)
        return records

    def run_word_batch(self, pool, windows, runnable_segments):
        if not runnable_segments:
            return []
        # Budget point = one word of evidence; stable segment ordering.
        points = {}
        for seg in self.transcript["segments"]:
            points[seg["segment_id"]] = len(seg["words"])
        self.scheduler.enqueue(runnable_segments, points)
        meta = {seg["segment_id"]: seg
                for seg in self.transcript["segments"]}
        active, report = self.scheduler.next_batch(meta)

        jobs = []
        for seg_id in active:
            jobs.append((self.compute_words,
                         (windows, [seg_id]), {}))
        results = pool.map_ordered(jobs)
        records = []
        for _, value, error in results:
            if error is not None:
                raise error
            records.extend(value)
        self.scheduler.mark_done(active)
        self._last_report = report
        return records, report

    # ------------------------------------------------------------------
    def run_calibration(self, all_word_records):
        rounds = calibration.ordered_rounds(self.transcript)
        cals = calibration.calibrate(
            rounds, all_word_records, self.config, self.tier)
        word_keys = sorted({record["cache_key"]
                            for record in all_word_records})
        key = calibration_key(word_keys, self._fps["config"], self.tier.name,
                              self._fps["label"])
        for item in cals:
            item["cache_key"] = key
        if self.ledger.reusable(key):
            entry = self.ledger.reuse(key)
            self.audit["reused"].append(key)
            return entry["artifact"]["calibrations"], key
        self.ledger.record(key, "calibration", {"rounds": len(rounds)},
                           "valid", "freshly computed",
                           {"calibrations": cals})
        self.audit["computed"].append(key)
        return cals, key

    def run_aggregation(self, cals, all_word_records, cal_key):
        summaries = aggregation.aggregate_segments(
            cals, all_word_records, self.transcript, self.config, self.tier)
        for summary in summaries:
            seg_cal_keys = [cal_key]
            w_keys = sorted({
                record["cache_key"] for record in all_word_records
                if record["segment_id"] == summary["segment_id"]
            })
            key = segment_key(
                summary["segment_id"], seg_cal_keys, w_keys,
                summary["window_ids"], self._fps["config"], self.tier.name,
                self._fps["label"])
            summary["cache_key"] = key
            if self.ledger.reusable(key):
                entry = self.ledger.reuse(key)
                self.audit["reused"].append(key)
                summary.update(entry["artifact"]["summary"])
            else:
                self.ledger.record(key, "segment",
                                   {"segment_id": summary["segment_id"]},
                                   "valid", "freshly computed",
                                   {"summary": summary})
                self.audit["computed"].append(key)
        return summaries

    # ------------------------------------------------------------------
    def reference_path(self):
        ref_tier = fp.REFERENCE
        ref_windows = features.load_windows(
            self.wav_path, self.transcript, self.config, ref_tier)
        ref_words = evidence.build_all_word_evidence(
            self.transcript, ref_windows, self.config, ref_tier)
        rounds = calibration.ordered_rounds(self.transcript)
        ref_cals = calibration.calibrate(
            rounds, ref_words, self.config, ref_tier)
        ref_summaries = aggregation.aggregate_segments(
            ref_cals, ref_words, self.transcript, self.config, ref_tier)
        return ref_windows, ref_words, ref_cals, ref_summaries

    def error_tracking(self, windows, words, cals, summaries):
        ref_windows, ref_words, ref_cals, ref_summaries = \
            self.reference_path()
        tolerance = self.config["tolerance"]
        reports = [
            error_track.track_window(
                windows, ref_windows, self.tier, tolerance["window"]),
            error_track.track_words(
                words, ref_words, self.tier, tolerance["word"]),
            error_track.track_calibration(
                cals, ref_cals, tolerance["calibration"]),
            error_track.track_segments(
                summaries, ref_summaries, tolerance["segment"]),
        ]
        return {"reports": reports,
                "first_breach": error_track.first_overall_breach(reports)}

    # ------------------------------------------------------------------
    def _persist(self, manifest, state, windows, words, cals=None,
                 summaries=None, errors=None, complete=False,
                 exit_report=None):
        self.store.begin()
        self.store.write_json("windows.json", windows)
        self.store.write_json("words.json", words)
        self.store.write_json("state.json", state)
        if cals is not None:
            self.store.write_json("calibrations.json", cals)
        if summaries is not None:
            self.store.write_json("segments.json", summaries)
        if errors is not None:
            self.store.write_json("error_tracking.json", errors)
        if exit_report is not None:
            self.store.write_json("exit_report.json", exit_report)
        self.store.write_json("audit.json", self.audit)
        self.store.write_json(
            "ledger.json",
            {"entries": self.ledger.entries, "audit": self.ledger.audit})
        if os.path.exists(self.ledger_path):
            os.remove(self.ledger_path)
        manifest["complete"] = complete
        self.store.commit(manifest)

    def run(self):
        self._fps = self.fingerprints()
        mode, prior_manifest = self.store.detect_mode(
            self.config, self.transcript)
        self.mode = mode
        prior_state = self.store.read_json("state.json")
        resuming = (mode == "resume" and prior_state is not None
                    and not prior_state.get("complete", False))
        base_version = (prior_manifest or {}).get("version", 0)
        manifest = {
            "version": base_version + 1,
            "config_fp": self._fps["config"],
            "audio_fp": self._fps["audio"],
            "label_fp": self._fps["label"],
            "structure_fp": self._fps["structure"],
            "tier": self.tier.name,
            "shutdown_requested": self.shutdown.requested,
        }
        pool = BoundedThreadPool(self.config["scheduler"]["max_workers"])
        state = {"mode": mode, "batches": [], "consumed_budget": 0,
                 "todo": [], "tier": self.tier.name, "complete": False}
        try:
            windows = self.compute_windows(pool)
            all_segments = [seg["segment_id"]
                            for seg in self.transcript["segments"]]
            all_words = []
            if resuming and prior_state.get("todo"):
                pending = list(prior_state["todo"])
                state["consumed_budget"] = prior_state.get(
                    "consumed_budget", 0)
            else:
                pending = list(all_segments)
            batch_no = 0
            while pending:
                if self.shutdown.requested:
                    break
                delay = self.config["scheduler"].get("test_step_delay", 0.0)
                if delay:
                    self.shutdown._event.wait(delay)
                if self.shutdown.requested:
                    state["todo"] = pending
                    exit_report = {
                        "reason": getattr(
                            self.shutdown._event, "reason", "shutdown"),
                        "todo": pending,
                        "consumed_budget": state["consumed_budget"],
                    }
                    self._persist(manifest, state, windows, all_words,
                                  exit_report=exit_report)
                    return self._result(manifest, state, windows,
                                        all_words, None, None, None, mode)
                batch_no += 1
                records, report = self.run_word_batch(
                    pool, windows, pending)
                all_words.extend(records)
                state["consumed_budget"] += sum(
                    len(next(seg["words"] for seg in
                             self.transcript["segments"]
                             if seg["segment_id"] == seg_id))
                    for seg_id in report["ran"])
                state["batches"].append(report)
                done = set(report["ran"])
                pending = [seg_id for seg_id in pending
                           if seg_id not in done]
                if self.scheduler.has_todo():
                    queued = [item["segment_id"]
                              for item in self.scheduler.todo]
                    pending = list(dict.fromkeys(queued + pending))
                state["todo"] = pending
                self._persist(manifest, state, windows, all_words)
                if self.max_batches is not None and \
                        batch_no >= self.max_batches:
                    break
                if not report["ran"] and not self.scheduler.has_todo():
                    break
            if self.shutdown.requested or pending:
                state["todo"] = pending
                exit_report = None
                if self.shutdown.requested:
                    exit_report = {
                        "reason": getattr(
                            self.shutdown._event, "reason", "shutdown"),
                        "todo": pending,
                        "consumed_budget": state["consumed_budget"],
                    }
                self._persist(manifest, state, windows, all_words,
                              exit_report=exit_report)
                return self._result(manifest, state, windows, all_words,
                                    None, None, None, mode)
            cals, cal_key = self.run_calibration(all_words)
            summaries = self.run_aggregation(cals, all_words, cal_key)
            errors = self.error_tracking(
                windows, all_words, cals, summaries)
            state["todo"] = []
            state["complete"] = True
            self._persist(manifest, state, windows, all_words, cals,
                          summaries, errors, complete=True)
            return self._result(manifest, state, windows, all_words, cals,
                                summaries, errors, mode)
        finally:
            pool.shutdown()

    def _result(self, manifest, state, windows, words, cals, summaries,
                errors, mode):
        return {
            "manifest": manifest,
            "windows": windows,
            "words": words,
            "calibrations": cals or [],
            "segments": summaries or [],
            "error_tracking": errors,
            "state": state,
            "audit": self.audit,
            "mode": mode,
        }
