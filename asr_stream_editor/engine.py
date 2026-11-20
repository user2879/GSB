"""Incremental streaming engine: edits, budgets, concurrency and snapshots."""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .audio import (AudioClip, ChunkPlan, plan_chunks, plan_frames,
                    replace_chunk_samples)
from .config import AcousticConfig, PhonemeMap, StreamConfig, config_hash
from .errors import (E_DECODE_FAILED, E_FEATURE_FAILED, E_REDECODE_BUDGET_PENDING,
                     FailureRecord)
from .features import FrameFeatures, compute_features, features_artifact
from .snapshot import (STAGE_DECODE, STAGE_FEATURES, STAGE_MERGE, STAGE_PREFIX,
                       SnapshotStore)
from .stability import evaluate_stability, split_tokens
from .streaming import (ChunkResult, decode_one_chunk, merge_chunks,
                        re_score_tokens)


TASK_FEATURE = "feature"
TASK_DECODE = "decode"


def stable_sort_key(chunk_index: int, stage: str) -> Tuple[int, int]:
    """Deterministic ordering key derived from chunk index and stage name."""
    stage_rank = {TASK_FEATURE: 0, TASK_DECODE: 1}[stage]
    return chunk_index * 10 + stage_rank


@dataclass
class EngineResponse:
    status: str
    code: Optional[str]
    command: str
    stable_text: str
    tail_frames: int
    affected_chunks: List[int]
    progress: Dict[str, Any]
    reused_chunks: List[int]
    redecoded_chunks: List[int]
    reused_sample_ranges: List[Dict[str, int]]
    redecoded_sample_ranges: List[Dict[str, int]]
    transcript: List[Dict[str, Any]]
    conflicts: List[Dict[str, Any]]
    failures: List[Dict[str, Any]]
    budget_consumed_ms: int
    within_budget: bool
    version: int
    recovery_chunk_index: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "code": self.code,
            "command": self.command,
            "stable_prefix_text": self.stable_text,
            "mutable_tail_frames": self.tail_frames,
            "affected_chunks": self.affected_chunks,
            "progress": self.progress,
            "reused_chunks": self.reused_chunks,
            "redecoded_chunks": self.redecoded_chunks,
            "reused_sample_ranges": self.reused_sample_ranges,
            "redecoded_sample_ranges": self.redecoded_sample_ranges,
            "transcript": self.transcript,
            "alignment_conflicts": self.conflicts,
            "failures": self.failures,
            "budget_consumed_ms": self.budget_consumed_ms,
            "within_budget": self.within_budget,
            "global_version": self.version,
            "recovery_chunk_index": self.recovery_chunk_index,
        }


class StreamingEngine:
    """Stateful incremental engine for one audio clip."""

    def __init__(self, clip: AudioClip, phoneme_map: PhonemeMap,
                 acoustic: AcousticConfig, stream: StreamConfig,
                 snapshot_store: SnapshotStore,
                 fault_injection: Optional[Dict[str, Any]] = None,
                 should_exit=None):
        self.clip = clip
        self.phoneme_map = phoneme_map
        self.acoustic = acoustic
        self.stream = stream
        self.snapshot_store = snapshot_store
        self.fault_injection = fault_injection or {}
        self.samples = clip.samples.copy()
        self.backoff_scale = 1.0
        self.overlap_ms = stream.overlap_ms
        self.min_consistent_frames = stream.min_consistent_frames
        self.chunks: List[ChunkPlan] = plan_chunks(
            self.samples.shape[0], stream.chunk_ms, self.overlap_ms)
        self.feature_cache: Dict[int, FrameFeatures] = {}
        self.chunk_results: Dict[int, ChunkResult] = {}
        self.failures: List[FailureRecord] = []
        self.merged: Any = None
        self.stability: Any = None
        self.version = 0
        self.recovery_chunk = -1
        self.within_budget_responses = 0
        self.over_budget_responses = 0
        self.budget_consumed_total = 0
        self.stage_timings: Dict[str, List[float]] = {
            STAGE_FEATURES: [], STAGE_DECODE: [], STAGE_MERGE: [], STAGE_PREFIX: []}
        self._lock = threading.RLock()
        self._pending: List[Tuple[int, str]] = []
        self._last_committed_chunk = -1
        self._stop_after_chunk = self.fault_injection.get("exit_after_chunk")
        self._should_exit = should_exit
        self._crashed_chunks = {
            int(key): value for key, value in
            self.fault_injection.get("crash_chunks", {}).items()}
        self._failed_chunks = {
            int(key): value for key, value in
            self.fault_injection.get("fail_chunks", {}).items()}
        self._recover()

    # ------------------------------------------------------------------ state
    def _effective_stream(self) -> StreamConfig:
        return StreamConfig(
            chunk_ms=self.stream.chunk_ms,
            overlap_ms=self.overlap_ms,
            min_consistent_frames=self.min_consistent_frames,
            redecode_budget_ms=self.stream.redecode_budget_ms,
            concurrency=self.stream.concurrency,
            concurrency_limit=self.stream.concurrency_limit,
            snapshot_commit_interval_chunks=self.stream.snapshot_commit_interval_chunks,
            precision=self.stream.precision,
            simulated_task_cost_ms=self.stream.simulated_task_cost_ms,
        )

    def _recover(self) -> None:
        point = self.snapshot_store.recovery_point()
        if point is None:
            self.recovery_chunk = -1
            return
        current_fp = self.clip.fingerprint
        if point["audio_fingerprint"] != current_fp:
            self.recovery_chunk = -1
            return
        self.version = int(point["global_version"])
        self.recovery_chunk = int(point["recovery_chunk_index"])

    # ------------------------------------------------------------- task runner
    def _maybe_inject_failure(self, chunk_index: int, stage: str) -> None:
        stage_alias = {TASK_FEATURE: STAGE_FEATURES, TASK_DECODE: STAGE_DECODE}[stage]
        crash = self._crashed_chunks.get(chunk_index)
        if crash and crash.get("stage", STAGE_FEATURES) in (stage, stage_alias):
            # Simulates SIGKILL/power loss: no Python cleanup runs, so only
            # previously swapped-in snapshots can survive.
            os._exit(int(crash.get("exit_code", 9)))
        rule = self._failed_chunks.get(chunk_index)
        if rule and rule.get("stage", stage_alias) in (stage, stage_alias):
            raise RuntimeError(rule.get("message", f"injected {stage} failure"))

    def _run_task(self, chunk_index: int, stage: str) -> None:
        chunk = self.chunks[chunk_index]
        if stage == TASK_FEATURE:
            self._maybe_inject_failure(chunk_index, TASK_FEATURE)
            part = self.samples[chunk.start_sample:chunk.end_sample]
            features = compute_features(part, self.acoustic, self.phoneme_map,
                                        self.stream.significant_digits)
            self.feature_cache[chunk_index] = features
        else:
            self._maybe_inject_failure(chunk_index, TASK_DECODE)
            result = decode_one_chunk(self.samples, chunk, self.acoustic,
                                      self.phoneme_map, self._effective_stream(),
                                      self.backoff_scale)
            self.chunk_results[chunk_index] = result
            if chunk_index not in self.feature_cache:
                self.feature_cache[chunk_index] = result.features

    def _run_tasks(self, tasks: Sequence[Tuple[int, str]], budget_ms: int
                   ) -> Tuple[List[Tuple[int, str]], int]:
        """Run the deterministic budget prefix concurrently; return pending + cost.

        The budget decision is made on stable ordering keys before any thread
        starts, so concurrency and scheduling cannot change which tasks ran or
        the accumulated deterministic budget consumption.
        """
        ordered = sorted(tasks, key=lambda task: stable_sort_key(task[0], task[1]))
        cost_each = self.stream.simulated_task_cost_ms
        runnable: List[Tuple[int, str]] = []
        consumed = 0
        for task in ordered:
            if consumed + cost_each > budget_ms:
                break
            runnable.append(task)
            consumed += cost_each
        pending = list(ordered[len(runnable):])
        max_workers = min(self.stream.concurrency, self.stream.concurrency_limit)

        def worker(task: Tuple[int, str]) -> Tuple[Tuple[int, str], Optional[Exception]]:
            started_at = time.perf_counter()
            try:
                self._run_task(task[0], task[1])
            except Exception as exc:  # converted to an ordered failure record
                stage_name = STAGE_FEATURES if task[1] == TASK_FEATURE else STAGE_DECODE
                self.stage_timings[stage_name].append(
                    (time.perf_counter() - started_at) * 1000.0)
                return task, exc
            stage_name = STAGE_FEATURES if task[1] == TASK_FEATURE else STAGE_DECODE
            self.stage_timings[stage_name].append(
                (time.perf_counter() - started_at) * 1000.0)
            return task, None

        if max_workers > 1 and runnable:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                outcomes = [future.result()
                            for future in [pool.submit(worker, task) for task in runnable]]
        else:
            outcomes = [worker(task) for task in runnable]
        for task, error in outcomes:
            if error is not None:
                stage_name = STAGE_FEATURES if task[1] == TASK_FEATURE else STAGE_DECODE
                code = E_FEATURE_FAILED if task[1] == TASK_FEATURE else E_DECODE_FAILED
                self._record_failure(task[0], stage_name, code, error)
                if task[1] == TASK_FEATURE:
                    self.chunk_results[task[0]] = ChunkResult(
                        index=task[0], chunk=self.chunks[task[0]], failed=True,
                        failure_code=code, failure_message=str(error))
                else:
                    previous = self.chunk_results.get(task[0])
                    self.chunk_results[task[0]] = ChunkResult(
                        index=task[0], chunk=self.chunks[task[0]],
                        features=previous.features if previous else None,
                        failed=True, failure_code=code,
                        failure_message=str(error))
        return pending, consumed

    # -------------------------------------------------------------- failures
    def _record_failure(self, chunk_index: int, stage: str, code: str,
                        exc: Exception) -> None:
        record = FailureRecord(
            source=self.clip.source, stage=stage, code=code,
            exception_type=type(exc).__name__, message=str(exc),
            chunk_index=chunk_index)
        if record.to_dict() not in [item.to_dict() for item in self.failures]:
            self.failures.append(record)
        self.failures.sort(key=lambda item: (item.chunk_index, item.stage, item.code))

    # ------------------------------------------------------------- planning
    def _needed_tasks(self, affected: Sequence[int]) -> List[Tuple[int, str]]:
        tasks: List[Tuple[int, str]] = []
        for chunk_index in sorted(affected):
            existing = self.chunk_results.get(chunk_index)
            if existing is not None and existing.failed:
                # Failed chunks stay in the failure list; they do not block
                # convergence of the remaining chunks.
                continue
            feature_ok = chunk_index in self.feature_cache
            if not feature_ok:
                tasks.append((chunk_index, TASK_FEATURE))
            decode_ok = bool(existing and not existing.failed)
            if not decode_ok:
                tasks.append((chunk_index, TASK_DECODE))
        return sorted(tasks, key=lambda task: stable_sort_key(task[0], task[1]))

    def _chunk_failed(self, chunk_index: int) -> bool:
        result = self.chunk_results.get(chunk_index)
        return bool(result and result.failed)

    def _invalidate_features(self, chunk_indices: Sequence[int]) -> None:
        for index in chunk_indices:
            self.feature_cache.pop(index, None)
            self.chunk_results.pop(index, None)

    def _invalidate_decodes(self, chunk_indices: Sequence[int]) -> None:
        for index in chunk_indices:
            self.chunk_results.pop(index, None)

    # --------------------------------------------------------------- edits
    def edit_overlap(self, new_overlap_ms: int) -> List[int]:
        if new_overlap_ms <= 0 or new_overlap_ms >= self.stream.chunk_ms:
            raise ValueError("overlap_ms must be positive and smaller than chunk_ms")
        self.overlap_ms = new_overlap_ms
        self.chunks = plan_chunks(self.samples.shape[0], self.stream.chunk_ms,
                                  self.overlap_ms)
        # frame-level features stay reusable; chunk geometry invalidates decodes
        self.feature_cache.clear()
        self.chunk_results.clear()
        return list(range(len(self.chunks)))

    def edit_min_consistent_frames(self, new_value: int) -> List[int]:
        if new_value <= 0:
            raise ValueError("min_consistent_frames must be positive")
        self.min_consistent_frames = new_value
        # prefix threshold does not invalidate any acoustic or decode artifact
        return []

    def edit_backoff(self, scale: float) -> List[int]:
        if scale < 0.0:
            raise ValueError("backoff scale must be >= 0")
        self.backoff_scale = float(scale)
        affected = [chunk.index for chunk in self.chunks]
        self._invalidate_decodes(affected)
        return affected

    def replace_chunk_audio(self, chunk_index: int, replacement: np.ndarray) -> List[int]:
        if chunk_index < 0 or chunk_index >= len(self.chunks):
            raise IndexError(f"chunk {chunk_index} out of range")
        chunk = self.chunks[chunk_index]
        self.samples = replace_chunk_samples(self.samples, chunk, replacement)
        affected = [item.index for item in self.chunks
                    if item.start_sample < chunk.own_end_sample
                    and item.end_sample > chunk.own_start_sample]
        self._invalidate_features(affected)
        return sorted(affected)

    # ------------------------------------------------------------- progress
    def _progress(self, pending: Sequence[Tuple[int, str]]) -> Dict[str, Any]:
        total_feature = len(self.chunks)
        done_feature = len(self.feature_cache)
        total_decode = len(self.chunks)
        done_decode = sum(1 for result in self.chunk_results.values() if not result.failed)
        return {
            "total_chunks": len(self.chunks),
            "features_done": done_feature,
            "features_total": total_feature,
            "decode_done": done_decode,
            "decode_total": total_decode,
            "pending_tasks": [
                {"chunk_index": chunk_index, "stage": stage}
                for chunk_index, stage in pending
            ],
            "percent_complete": round(
                100.0 * (done_feature + done_decode)
                / (2 * max(1, total_feature)), 2),
        }

    # --------------------------------------------------------------- merge
    def _rebuild_transcript(self) -> None:
        frame_plan = plan_frames(self.samples.shape[0], self.acoustic.frame_length_ms,
                                 self.acoustic.frame_shift_ms)
        frame_shift = frame_plan.frame_shift
        ordered_results = [self.chunk_results[index]
                           for index in sorted(self.chunk_results)]
        merged = merge_chunks(ordered_results, self.phoneme_map.words, frame_shift,
                              self.samples.shape[0], self.stream.significant_digits)
        merged.tokens = re_score_tokens(merged.tokens, ordered_results,
                                        self.phoneme_map.words,
                                        self.stream.significant_digits)
        self.merged = merged
        self.stability = evaluate_stability(
            ordered_results, merged.n_frames, self.min_consistent_frames)

    def _range_dicts(self, indices: Sequence[int]) -> List[Dict[str, int]]:
        ranges = []
        for index in indices:
            chunk = self.chunks[index]
            ranges.append({
                "chunk_index": index,
                "start_sample": chunk.start_sample,
                "end_sample": chunk.end_sample,
                "own_start_sample": chunk.own_start_sample,
                "own_end_sample": chunk.own_end_sample,
            })
        return ranges

    def _snapshot_commit(self, affected: Sequence[int], command: str) -> None:
        self.version += 1
        frame_shift_ms = self.acoustic.frame_shift_ms
        tmp_dir = tempfile.mkdtemp(prefix="stage-", dir=self.snapshot_store.tmp_root)
        stage_files: Dict[str, List[str]] = {
            STAGE_FEATURES: [], STAGE_DECODE: [], STAGE_MERGE: [], STAGE_PREFIX: []}
        for index in sorted(self.chunk_results):
            result = self.chunk_results[index]
            if result.failed or result.features is None:
                continue
            feature_path = os.path.join(tmp_dir, f"features_chunk_{index:04d}.json")
            with open(feature_path, "w", encoding="utf-8") as handle:
                import json
                json.dump(features_artifact(result.features,
                                            f"{self.clip.source}#chunk{index}",
                                            frame_shift_ms), handle, sort_keys=True)
            stage_files[STAGE_FEATURES].append(feature_path)
            decode_path = os.path.join(tmp_dir, f"decode_chunk_{index:04d}.json")
            with open(decode_path, "w", encoding="utf-8") as handle:
                import json
                json.dump({
                    "chunk_index": index,
                    "frame_span": list(result.frame_span),
                    "tokens": [token.to_dict(frame_shift_ms) for token in result.tokens],
                }, handle, sort_keys=True)
            stage_files[STAGE_DECODE].append(decode_path)
        merge_path = os.path.join(tmp_dir, "merged.json")
        prefix_path = os.path.join(tmp_dir, "prefix.json")
        with open(merge_path, "w", encoding="utf-8") as handle:
            import json
            json.dump(self._merge_artifact(frame_shift_ms), handle, sort_keys=True)
        with open(prefix_path, "w", encoding="utf-8") as handle:
            import json
            json.dump(self._prefix_artifact(frame_shift_ms), handle, sort_keys=True)
        stage_files[STAGE_MERGE].append(merge_path)
        stage_files[STAGE_PREFIX].append(prefix_path)
        fingerprint = hashlib.sha256(
            np.ascontiguousarray(self.samples).tobytes()).hexdigest()[:16]
        cfg_hash = config_hash(self.phoneme_map, self.acoustic,
                               self._effective_stream(),
                               extra={"backoff_scale": self.backoff_scale})
        max_done = max((index for index in self.chunk_results), default=-1)
        self.snapshot_store.commit(
            version=self.version, config_hash_value=cfg_hash,
            audio_fingerprint=self.clip.fingerprint, stage_artifacts=stage_files,
            recovery_chunk=max_done,
            extra={"command": command,
                   "current_sample_fingerprint": fingerprint})
        self.recovery_chunk = max_done
        shutil.rmtree(tmp_dir, ignore_errors=True)

    def _merge_artifact(self, frame_shift_ms: int) -> Dict[str, Any]:
        return {
            "source": self.clip.source,
            "frame_labels": list(self.merged.frame_labels),
            "frame_owner": list(self.merged.frame_owner),
            "tokens": [token.to_dict(frame_shift_ms) for token in self.merged.tokens],
            "conflicts": list(self.merged.conflicts),
        }

    def _prefix_artifact(self, frame_shift_ms: int) -> Dict[str, Any]:
        prefix_tokens, tail_tokens = split_tokens(
            self.merged.tokens, self.stability.stable_until_frame)
        return {
            "stable_until_frame": self.stability.stable_until_frame,
            "boundary_runs": self.stability.boundary_runs,
            "stable_prefix": [token.to_dict(frame_shift_ms) for token in prefix_tokens],
            "mutable_tail": [token.to_dict(frame_shift_ms) for token in tail_tokens],
        }

    # ------------------------------------------------------------ public API
    def process_command(self, command: Dict[str, Any],
                        replacement_samples: Optional[np.ndarray] = None
                        ) -> EngineResponse:
        """Apply one interactive command and return a budget-bounded response.

        Pending work is kept in ``self._pending`` and resumed on the next
        command (or an explicit ``resume``), so an over-budget intermediate
        state always advances to the final result.
        """
        with self._lock:
            name = command.get("command", "resume")
            affected: List[int] = []
            if name == "initial_decode":
                affected = list(range(len(self.chunks)))
            elif name == "set_overlap":
                affected = self.edit_overlap(int(command["overlap_ms"]))
            elif name == "set_stable_frames":
                affected = self.edit_min_consistent_frames(
                    int(command["min_consistent_frames"]))
            elif name == "set_backoff":
                affected = self.edit_backoff(float(command["backoff_scale"]))
            elif name == "replace_chunk":
                if replacement_samples is None:
                    raise ValueError("replace_chunk requires replacement samples")
                affected = self.replace_chunk_audio(
                    int(command["chunk_index"]), replacement_samples)
            elif name in ("resume", "finalize"):
                affected = self._outstanding_chunks()
            else:
                raise ValueError(f"unknown command: {name}")

            planned = self._needed_tasks(affected)
            pending_existing = list(getattr(self, "_pending", []))
            all_tasks = self._dedupe_tasks(pending_existing + planned)
            self._pending = all_tasks
            started = time.perf_counter()
            pending, consumed = self._run_tasks(
                all_tasks, self.stream.redecode_budget_ms)
            wall_ms = (time.perf_counter() - started) * 1000.0
            self.budget_consumed_total += consumed
            self._pending = pending
            within = not pending
            if within:
                self.within_budget_responses += 1
            else:
                self.over_budget_responses += 1
            if self.chunk_results:
                # Rebuild the partial transcript from every chunk finished so
                # far, then commit at the configured chunk interval. A hard
                # kill therefore always leaves a complete swapped snapshot at
                # the most recent interval boundary.
                merge_started = time.perf_counter()
                self._rebuild_transcript()
                self.stage_timings[STAGE_MERGE].append(
                    (time.perf_counter() - merge_started) * 1000.0)
                prefix_started = time.perf_counter()
                max_done = max(self.chunk_results.keys())
                _ = split_tokens(self.merged.tokens,
                                 self.stability.stable_until_frame)
                self.stage_timings[STAGE_PREFIX].append(
                    (time.perf_counter() - prefix_started) * 1000.0)
                interval = self.stream.snapshot_commit_interval_chunks
                completed_count = max_done + 1
                last_committed_count = self._last_committed_chunk + 1
                due = (completed_count // interval >
                       last_committed_count // interval)
                final_commit = self._decode_complete() and (
                    self._last_committed_chunk < max_done)
                if due or final_commit or name in ("finalize",):
                    self._snapshot_commit(affected, name)
                    self._last_committed_chunk = max_done
                self._maybe_exit(name)
            response = self._build_response(name, affected, pending, consumed,
                                            wall_ms, within)
            return response

    def _dedupe_tasks(self, tasks: Sequence[Tuple[int, str]]
                      ) -> List[Tuple[int, str]]:
        seen = set()
        unique: List[Tuple[int, str]] = []
        for task in sorted(tasks, key=lambda item: stable_sort_key(item[0], item[1])):
            if task not in seen:
                seen.add(task)
                unique.append(task)
        return unique

    def _outstanding_chunks(self) -> List[int]:
        return [chunk.index for chunk in self.chunks
                if chunk.index not in self.chunk_results
                or self.chunk_results[chunk.index].failed
                or chunk.index not in self.feature_cache]

    def _decode_complete(self) -> bool:
        return all(index in self.chunk_results for index in range(len(self.chunks)))

    def _maybe_exit(self, name: str) -> None:
        if self._stop_after_chunk is None:
            if self._should_exit is not None and self._should_exit():
                # graceful termination: the snapshot above was already
                # atomically committed, so exit 0 leaves a consistent state
                raise SystemExit(0)
            return
        max_done = max(self.chunk_results.keys()) if self.chunk_results else -1
        if max_done >= int(self._stop_after_chunk):
            raise SystemExit(0)

    def _build_response(self, name: str, affected: Sequence[int],
                        pending: Sequence[Tuple[int, str]], consumed: int,
                        wall_ms: float, within: bool) -> EngineResponse:
        frame_shift_ms = self.acoustic.frame_shift_ms
        if self.merged is not None:
            prefix_tokens, tail_tokens = split_tokens(
                self.merged.tokens, self.stability.stable_until_frame)
            stable_text = "".join(token.word for token in prefix_tokens)
            transcript = [token.to_dict(frame_shift_ms)
                          for token in self.merged.tokens]
            conflicts = list(self.merged.conflicts)
            tail_frames = (self.merged.n_frames - self.stability.stable_until_frame
                           if self.stability.stable_until_frame >= 0 else 0)
        else:
            stable_text = ""
            transcript = []
            conflicts = []
            tail_frames = 0
        done_indices = sorted(index for index, result in self.chunk_results.items()
                              if not result.failed)
        reused = [index for index in done_indices if index not in affected]
        redecoded = [index for index in done_indices if index in affected]
        code = None if within else E_REDECODE_BUDGET_PENDING
        status = "completed" if within else "pending"
        return EngineResponse(
            status=status, code=code, command=name,
            stable_text=stable_text, tail_frames=max(0, tail_frames),
            affected_chunks=list(affected),
            progress=self._progress(pending),
            reused_chunks=reused, redecoded_chunks=redecoded,
            reused_sample_ranges=self._range_dicts(reused),
            redecoded_sample_ranges=self._range_dicts(redecoded),
            transcript=transcript, conflicts=conflicts,
            failures=[record.to_dict() for record in self.failures],
            budget_consumed_ms=consumed, within_budget=within,
            version=self.version, recovery_chunk_index=self.recovery_chunk)
