"""Pipeline orchestration: evidence -> boundary -> trajectory -> summary."""

import json
import os
import signal

import numpy as np

from .blockcache import BlockCache
from .candidates import detect_candidates
from .config import config_hash, load_event_params, load_schedule_config
from .errors import (E_BUDGET_EXHAUSTED, E_CONFIG_MISMATCH, E_DECODE_FAIL,
                     E_INTERNAL, BusinessError)
from .evidence import (SequenceError, compute_frame_evidence, decode_frame,
                       resolve_timestamps, scan_sequence)
from .report import write_report_atomic
from .scheduler import BudgetLedger, build_schedule, run_frame_stage
from .segments import build_segments, group_primitives
from .snapshot import STAGES, SnapshotManager, build_manifest
from .tracks import build_tracks

EXIT_OK = 0
EXIT_CONFIG_MISMATCH = 2
EXIT_ERROR = 1


def _atomic_write_json(path, payload):
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


class Pipeline:
    def __init__(self, samples_dir, configs_dir, workdir, max_concurrency=None):
        self.samples_dir = samples_dir
        self.configs_dir = configs_dir
        self.workdir = workdir
        self.event_params = load_event_params(
            os.path.join(configs_dir, "event_params.json"))
        self.schedule = load_schedule_config(
            os.path.join(configs_dir, "schedule.json"))
        if max_concurrency is not None:
            object.__setattr__(self.schedule, "max_concurrent_sequences",
                               int(max_concurrency))
        self.config_hash = config_hash(self.event_params, self.schedule)
        self.snapshots = SnapshotManager(workdir)
        self.artifacts_dir = os.path.join(workdir, "artifacts")
        self.stop_requested = False
        self.failures = []
        self.sequences = {}          # name -> scan info
        self.excluded = {}           # name -> reason (decode failed)
        self.cursors = {}            # name -> next frame index to process
        self.consumed_total = 0      # cumulative across runs
        self.version = 0
        self.stage_versions = {stage: 0 for stage in STAGES}
        self.recovered_from = None
        self.cache = None
        self.ledger = None
        self.thread_peak = 0
        self.segments_by_seq = {}
        self.dropped_by_seq = {}
        self.suppressed_by_seq = {}
        self.tracks_by_seq = {}
        self.track_decisions = []
        self.truncated = {}

    # ------------------------------------------------------------------ util

    def _record_failure(self, sequence, frame, stage, code, exception):
        self.failures.append({
            "sequence": sequence,
            "frame": int(frame),
            "stage": stage,
            "code": code,
            "exception": exception,
        })

    def _request_stop(self, signum, frame):
        self.stop_requested = True

    def _install_signal_handlers(self):
        signal.signal(signal.SIGINT, self._request_stop)
        signal.signal(signal.SIGTERM, self._request_stop)

    def _commit_snapshot(self, stage):
        self.version += 1
        self.stage_versions[stage] += 1
        segment_summaries = {
            name: [
                {"segment_id": s["segment_id"], "category": s["category"],
                 "start_ms": s["start_ms"], "end_ms": s["end_ms"]}
                for s in segs
            ]
            for name, segs in sorted(self.segments_by_seq.items())
        }
        manifest = build_manifest(
            version=self.version,
            config_hash=self.config_hash,
            stage=stage,
            stage_versions=self.stage_versions,
            sequences={
                name: {"cursor": self.cursors.get(name, 0),
                       "completed": (name in self.sequences
                                     and self.cursors.get(name, 0)
                                     >= self.sequences[name]["frame_count"])}
                for name in sorted(set(self.sequences) | set(self.excluded))
            },
            consumed_budget=self.consumed_total,
            segment_summaries=segment_summaries,
            resident_block_keys=(self.cache.resident_keys()
                                 if self.cache is not None else []),
        )
        self.snapshots.commit(manifest)

    # ---------------------------------------------------------------- resume

    def _try_resume(self):
        manifest = self.snapshots.load_manifest()
        if manifest is None:
            return True
        if (manifest.get("schema_version") != 1
                or manifest.get("config_hash") != self.config_hash):
            self._record_failure("*", -1, "summary", E_CONFIG_MISMATCH,
                                 "ConfigMismatch")
            _atomic_write_json(
                os.path.join(self.workdir, "failures.json"),
                {"failures": self.failures})
            print(f"{E_CONFIG_MISMATCH}: snapshot config hash "
                  f"{manifest.get('config_hash')} != {self.config_hash}; "
                  "refusing to resume")
            return False
        self.version = manifest["version"]
        self.stage_versions.update(manifest.get("stage_versions", {}))
        self.consumed_total = manifest.get("consumed_global_frame_budget", 0)
        self.recovered_from = {
            "manifest_version": manifest["version"],
            "stage": manifest["stage"],
            "cursors": {k: v["cursor"]
                        for k, v in manifest.get("sequences", {}).items()},
        }
        print(f"resume: recovered from manifest v{manifest['version']} "
              f"stage={manifest['stage']} cursors={self.recovered_from['cursors']}")
        return True

    # ------------------------------------------------------------------ scan

    def _scan_sequences(self):
        names = sorted(
            d for d in os.listdir(self.samples_dir)
            if os.path.isdir(os.path.join(self.samples_dir, d)))
        for name in names:
            seq_dir = os.path.join(self.samples_dir, name)
            try:
                frames, stamps, size = scan_sequence(seq_dir)
            except SequenceError as exc:
                self._record_failure(name, -1, "evidence", exc.code,
                                     type(exc).__name__)
                self.excluded[name] = exc.message
                continue
            resolved, sources = resolve_timestamps(stamps, 40)
            self.sequences[name] = {
                "dir": seq_dir,
                "frames": frames,
                "timestamps": resolved,
                "timestamp_sources": sources,
                "size": size,
                "frame_count": len(frames),
            }
            self.cursors.setdefault(name, 0)
        if self.recovered_from:
            for name, cursor in self.recovered_from["cursors"].items():
                if name in self.cursors:
                    self.cursors[name] = cursor

    # -------------------------------------------------------------- evidence

    def _block_dir(self, seq_name):
        path = os.path.join(self.artifacts_dir, "evidence", seq_name)
        os.makedirs(path, exist_ok=True)
        return path

    @staticmethod
    def _block_key(seq_name, block_index):
        return f"{seq_name}:blk{block_index:04d}"

    def _frame_work(self, seq_name, frame_index):
        info = self.sequences[seq_name]
        curr = decode_frame(info["frames"][frame_index])
        prev = (decode_frame(info["frames"][frame_index - 1])
                if frame_index > 0 else curr)
        evidence = compute_frame_evidence(prev, curr, self.event_params)
        survivors, suppressed = detect_candidates(
            seq_name, frame_index, prev, curr, evidence, self.event_params)
        return {"frame": frame_index, "evidence": evidence,
                "candidates": survivors, "suppressed": suppressed}

    def _flush_block(self, seq_name, block_index, frames_payload):
        key = self._block_key(seq_name, block_index)
        payload = {"block_key": key, "frames": frames_payload}
        _atomic_write_json(
            os.path.join(self._block_dir(seq_name),
                         f"blk{block_index:04d}.json"), payload)

    def _evidence_stage(self):
        names = sorted(self.sequences)
        frame_counts = {n: self.sequences[n]["frame_count"] for n in names}
        schedule = build_schedule(names, self.cursors, frame_counts)
        self.ledger = BudgetLedger(self.schedule.global_frame_budget,
                                   self.schedule.per_sequence_frame_quota, names)
        block_size = self.event_params.block_size_frames
        pending = {}   # (seq, block_index) -> {frame_index: payload}
        flushed_frames = [0]

        def on_done(seq_name, frame_index, result):
            block_index = frame_index // block_size
            pending.setdefault((seq_name, block_index), {})[frame_index] = result
            self.cursors[seq_name] = max(self.cursors[seq_name], frame_index + 1)
            self.consumed_total += 1
            flushed_frames[0] += 1
            # Flush any block whose frames are all present (or seq ended).
            for (seq, blk), frames in sorted(list(pending.items())):
                start = blk * block_size
                end = min(start + block_size, frame_counts[seq])
                if all(i in frames for i in range(start, end)) or \
                        all(i in frames or i >= frame_counts[seq]
                            for i in range(start, end)):
                    ordered = [frames[i] for i in sorted(frames)]
                    self._flush_block(seq, blk, ordered)
                    del pending[(seq, blk)]
            if flushed_frames[0] % max(
                    1, self.schedule.snapshot_commit_interval_frames) == 0:
                self._commit_snapshot("evidence")

        results, frame_failures, truncated, peak = run_frame_stage(
            schedule, self.ledger, self._frame_work,
            max_workers=self.schedule.max_concurrent_sequences,
            should_stop=lambda: self.stop_requested,
            on_frame_done=on_done)
        self.thread_peak = max(self.thread_peak, peak)
        for failure in frame_failures:
            code = E_DECODE_FAIL if "Png" in failure["exception"] or \
                "SequenceError" in failure["exception"] else E_INTERNAL
            self._record_failure(failure["sequence"], failure["frame"],
                                 "evidence", code, failure["exception"])
            if code == E_DECODE_FAIL:
                self.excluded[failure["sequence"]] = failure["message"]
        # Flush remaining partial blocks.
        for (seq, blk), frames in sorted(list(pending.items())):
            ordered = [frames[i] for i in sorted(frames)]
            self._flush_block(seq, blk, ordered)
        pending.clear()
        for seq_name, frame_index in sorted(truncated.items()):
            self.truncated[seq_name] = frame_index
            self._record_failure(seq_name, frame_index, "schedule",
                                 E_BUDGET_EXHAUSTED, "BudgetExhausted")
        self._commit_snapshot("evidence")
        return not self.stop_requested

    # -------------------------------------------------------------- boundary

    def _recompute_block(self, block_key):
        seq_name, blk = block_key.rsplit(":blk", 1)
        block_index = int(blk)
        info = self.sequences[seq_name]
        block_size = self.event_params.block_size_frames
        start = block_index * block_size
        end = min(start + block_size, self.cursors[seq_name])
        frames_payload = []
        for frame_index in range(start, end):
            frames_payload.append(self._frame_work(seq_name, frame_index))
        recomputed = {"block_key": block_key, "frames": frames_payload}
        artifact = os.path.join(self._block_dir(seq_name),
                                f"blk{block_index:04d}.json")
        if os.path.isfile(artifact):
            with open(artifact, "r", encoding="utf-8") as fh:
                stored = json.load(fh)
            if stored != recomputed:
                self._record_failure(seq_name, start, "boundary", E_INTERNAL,
                                     "RecomputeMismatch")
        return recomputed

    def _boundary_stage(self):
        self.cache = BlockCache(self.schedule.frame_block_residency_budget,
                                self._recompute_block)
        block_size = self.event_params.block_size_frames
        for seq_name in sorted(self.sequences):
            if seq_name in self.excluded:
                continue
            info = self.sequences[seq_name]
            done = self.cursors[seq_name]
            num_blocks = (done + block_size - 1) // block_size
            candidates = []
            suppressed = []
            for block_index in range(num_blocks):
                block = self.cache.access(self._block_key(seq_name, block_index))
                for frame_payload in block["frames"]:
                    candidates.extend(frame_payload["candidates"])
                    suppressed.extend(frame_payload["suppressed"])
            try:
                primitives = group_primitives(candidates, self.event_params)
                # Boundary confirmation re-references the blocks holding each
                # primitive's first/last candidate frame; evicted blocks are
                # recomputed from the original frame files by the cache loader.
                confirm_blocks = sorted({
                    frame // block_size
                    for prim in primitives
                    for frame in (prim["candidates"][0]["frame_index"],
                                  prim["candidates"][-1]["frame_index"])
                })
                for block_index in confirm_blocks:
                    self.cache.access(self._block_key(seq_name, block_index))
                timestamps = [(info["timestamps"][i],
                               info["timestamp_sources"][i])
                              for i in range(info["frame_count"])]
                segments, dropped = build_segments(
                    seq_name, primitives, timestamps, self.event_params)
            except Exception as exc:
                self._record_failure(seq_name, -1, "boundary", E_INTERNAL,
                                     type(exc).__name__)
                continue
            self.segments_by_seq[seq_name] = segments
            self.dropped_by_seq[seq_name] = dropped
            self.suppressed_by_seq[seq_name] = suppressed
        self._commit_snapshot("boundary")

    # ------------------------------------------------------------ trajectory

    def _trajectory_stage(self):
        for seq_name in sorted(self.segments_by_seq):
            try:
                tracks, decisions = build_tracks(
                    seq_name, self.segments_by_seq[seq_name], self.event_params)
            except Exception as exc:
                self._record_failure(seq_name, -1, "trajectory", E_INTERNAL,
                                     type(exc).__name__)
                continue
            self.tracks_by_seq[seq_name] = tracks
            self.track_decisions.extend(decisions)
        self._commit_snapshot("trajectory")

    # ---------------------------------------------------------------- summary

    def _summary_stage(self):
        all_segments = []
        for seq_name, segments in sorted(self.segments_by_seq.items()):
            for seg in segments:
                all_segments.append({"sequence": seq_name, **seg})
        all_segments.sort(key=lambda s: (s["start_ms"], s["category"],
                                         s["segment_id"]))
        all_tracks = []
        for seq_name, tracks in sorted(self.tracks_by_seq.items()):
            all_tracks.extend(tracks)
        all_tracks.sort(key=lambda t: (-t["confidence"], t["first_ms"],
                                       t["track_id"]))
        os.makedirs(self.workdir, exist_ok=True)
        _atomic_write_json(os.path.join(self.workdir, "events.json"), {
            "config_hash": self.config_hash,
            "segments": all_segments,
            "dropped": {k: v for k, v in sorted(self.dropped_by_seq.items())},
            "suppressed_candidates":
                {k: v for k, v in sorted(self.suppressed_by_seq.items())},
        })
        _atomic_write_json(os.path.join(self.workdir, "tracks.json"), {
            "config_hash": self.config_hash,
            "tracks": all_tracks,
        })
        _atomic_write_json(os.path.join(self.workdir, "failures.json"), {
            "failures": self.failures,
        })
        codes = sorted({f["code"] for f in self.failures})
        report = {
            "run_id": f"run-v{self.version}",
            "config_hash": self.config_hash,
            "concurrency": self.schedule.max_concurrent_sequences,
            "recovered_from": self.recovered_from,
            "sequences": {
                name: {
                    "processed_frames": self.cursors.get(name, 0),
                    "truncated_at_frame": self.truncated.get(name),
                    "event_segments": len(self.segments_by_seq.get(name, [])),
                    "tracks": len(self.tracks_by_seq.get(name, [])),
                    "quota_deductions": [
                        d["frame"] for d in self.ledger.deduction_order
                        if d["sequence"] == name
                    ] if self.ledger else [],
                    "excluded": name in self.excluded,
                }
                for name in sorted(set(self.sequences) | set(self.excluded))
            },
            "cache": self.cache.stats() if self.cache else
                     {"hits": 0, "misses": 0, "evictions": 0, "recomputes": 0},
            "budget": {
                "global_budget": self.schedule.global_frame_budget,
                "global_consumed_this_run":
                    len(self.ledger.deduction_order) if self.ledger else 0,
                "global_consumed_total": self.consumed_total,
                "quota_deduction_order":
                    self.ledger.deduction_order if self.ledger else [],
            },
            "track_decisions": self.track_decisions,
            "thread_peak": self.thread_peak,
            "failures": self.failures,
            "business_codes": codes,
        }
        write_report_atomic(report, os.path.join(self.workdir, "report.json"))
        self._commit_snapshot("summary")

    # -------------------------------------------------------------------- run

    def run(self):
        os.makedirs(self.workdir, exist_ok=True)
        self._install_signal_handlers()
        if not self._try_resume():
            return EXIT_CONFIG_MISMATCH
        self._scan_sequences()
        if not self._evidence_stage():
            # Graceful stop: current frame artifacts and progress are flushed.
            self._commit_snapshot("evidence")
            print("stop requested: progress committed, exiting 0")
            return EXIT_OK
        self._boundary_stage()
        self._trajectory_stage()
        self._summary_stage()
        return EXIT_OK
