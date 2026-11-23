"""Parallel shard scheduler, budget ledger, cache reuse and snapshot control."""

import os
import time

from . import audio as audio_mod
from . import decoder as decoder_mod
from . import features as features_mod
from . import fingerprints as fp
from . import stitching as stitching_mod
from .config import SNAPSHOT_VERSION
from .errors import (
    E_AUDIO_FORMAT,
    E_POINT_BUDGET_PENDING,
    E_SHARD_BUDGET_EXHAUSTED,
    PipelineError,
    failure_record,
)
from .storage import WorkStore


class ShardTask:
    def __init__(self, audio_name, shard_index, sample_start, sample_end,
                 frame_offset, injection):
        self.audio = audio_name
        self.shard_index = shard_index
        self.sample_start = sample_start
        self.sample_end = sample_end
        self.frame_offset = frame_offset
        self.injection = injection
        self.sort_key = (audio_name, shard_index)

    def identity(self):
        return {"audio": self.audio, "shard_index": self.shard_index}


class Orchestrator:
    def __init__(self, sample_dir, score_params, schedule_config, phoneme_map,
                 injections, workdir, serial=False, jitter_ms=0.0):
        self.sample_dir = sample_dir
        self.score_params = score_params
        self.schedule_config = schedule_config
        self.phoneme_map = phoneme_map
        self.injections = injections
        self.serial = serial
        self.jitter_ms = jitter_ms
        self.store = WorkStore(workdir)
        self.config_hash = fp.config_fingerprint(score_params, schedule_config)
        self.acoustic_hash = fp.acoustic_fingerprint(score_params)
        self.lm_hash = fp.lm_fingerprint(score_params)
        self.audio_items = {}
        self.tasks = []
        self.failures = []
        self.reuse_log = []
        self.ledger = []
        self.pending = []
        self.checkpoints = {}
        self.completed = {}
        self.snapshot_events = []
        self.shard_timings = {}
        self.run_consumed = 0

    # ------------------------------------------------------------------ setup
    def prepare(self):
        manifest = self.store.read_manifest()
        if manifest is not None and manifest.get("version") != SNAPSHOT_VERSION:
            self.snapshot_events.append({
                "code": "E_SNAPSHOT_STALE",
                "message": "snapshot version mismatch; ignoring prior manifest",
            })
            manifest = None
        if manifest is not None and manifest.get("config_hash") != self.config_hash:
            # Cache artifacts are content-addressed, so the on-disk cache is
            # retained and only key-derived invalidation triggers recompute.
            self.snapshot_events.append({
                "code": "E_SNAPSHOT_STALE",
                "message": "config hash changed; resuming with key-based invalidation",
            })
        if manifest is not None:
            self.checkpoints = dict(manifest.get("checkpoints", {}))
            self.completed = {
                f"{item['audio']}#{item['shard_index']}": item
                for item in manifest.get("completed_shards", [])
            }
        self._load_audio()
        self._build_tasks()
        return manifest

    def _load_audio(self):
        # Container-level validation is intentionally injection-free so that a
        # shard-targeted fault never cascades into the whole audio file.
        for path in audio_mod.list_wav_files(self.sample_dir):
            name = os.path.basename(path)
            try:
                item = audio_mod.validate_wav(path, None)
                self.audio_items[name] = item
                self._record_audio_stage(item)
            except PipelineError as exc:
                self.failures.append(failure_record(
                    name, None, "audio", exc.code, str(exc), exc.injection_id))

    def _record_audio_stage(self, item):
        key = fp.audio_cache_key(item["raw_fingerprint"])
        payload = {
            "audio": item["audio"],
            "n_samples": item["n_samples"],
            "raw_fingerprint": item["raw_fingerprint"],
        }
        if self.store.has("audio", key):
            self.store.load("audio", key)
            action = "reused"
        else:
            self.store.store("audio", key, payload)
            action = "computed"
        self.reuse_log.append({
            "audio": item["audio"], "shard_index": None, "stage": "audio",
            "action": action, "cache_key": key,
            "injection_id": None,
        })

    def _build_tasks(self):
        shift_ms = self.score_params["frame_shift_ms"]
        shift_samples = audio_mod.SAMPLE_RATE * shift_ms // 1000
        for name in sorted(self.audio_items):
            item = self.audio_items[name]
            spans = audio_mod.plan_shards(
                item["n_samples"],
                self.schedule_config["shard_duration_ms"],
                self.schedule_config["overlap_ms"],
                shift_ms,
            )
            for span in spans:
                injection = next((
                    spec for spec in self.injections.values()
                    if spec["audio"] == name
                    and spec["shard_index"] == span["shard_index"]
                ), None)
                frame_offset = span["sample_start"] // shift_samples
                self.tasks.append(ShardTask(
                    name, span["shard_index"], span["sample_start"],
                    span["sample_end"], frame_offset, injection,
                ))
        self.tasks.sort(key=lambda task: task.sort_key)

    # ------------------------------------------------------------------ worker
    def _jitter(self, task):
        if self.jitter_ms <= 0:
            return
        # Deterministic jitter derived from the stable sort key, not wall time.
        seed = sum(ord(ch) for ch in task.audio) + task.shard_index * 31
        time.sleep((seed % 7) * self.jitter_ms / 1000.0)

    def _apply_audio_injection(self, task, samples, frame_start):
        if not task.injection:
            return samples
        kind = task.injection["type"]
        if kind == "non_finite_samples":
            mutated = samples.copy()
            target = int(task.injection.get("sample_index", 0)) % max(1, mutated.shape[0])
            mutated[target] = float("nan")
            if not np_all_finite(mutated):
                raise PipelineError(
                    E_AUDIO_FORMAT,
                    f"{task.audio} shard {task.shard_index}: non-finite samples detected",
                    stage="audio", injection_id=task.injection["id"],
                )
        if kind == "truncate_wav":
            keep_ratio = float(task.injection.get("keep_ratio", 0.6))
            keep = max(1, int(samples.shape[0] * keep_ratio))
            truncated = samples[:keep]
            # A truncated shard cannot supply its full planned frame span.
            expected_frames = frame_start + features_mod.frame_signal(
                samples, self.score_params["frame_length_ms"],
                self.score_params["frame_shift_ms"],
            )[0].shape[0]
            actual_frames = frame_start + features_mod.frame_signal(
                truncated, self.score_params["frame_length_ms"],
                self.score_params["frame_shift_ms"],
            )[0].shape[0]
            if actual_frames < expected_frames:
                raise PipelineError(
                    E_AUDIO_FORMAT,
                    f"{task.audio} shard {task.shard_index}: truncated WAV bytes "
                    f"({actual_frames} frames < {expected_frames} planned)",
                    stage="audio", injection_id=task.injection["id"],
                )
            return truncated
        return samples

    def _emission_artifact(self, task, item):
        cfg = self.schedule_config
        frame_shift = audio_mod.SAMPLE_RATE * self.score_params["frame_shift_ms"] // 1000
        frame_start = task.sample_start // frame_shift
        samples = item["samples"][task.sample_start:task.sample_end]
        frame_end = frame_start + features_mod.frame_signal(
            samples, self.score_params["frame_length_ms"],
            self.score_params["frame_shift_ms"],
        )[0].shape[0]
        key = fp.emission_cache_key(
            item["raw_fingerprint"], self.acoustic_hash, cfg["quant_mode"],
            cfg["quant_frac_bits"], task.shard_index, frame_start, frame_end,
        )
        return key, samples, frame_start, frame_end

    def process_shard(self, task, allowance):
        """Execute feature + decode for one shard under a point allowance."""
        self._jitter(task)
        started = time.perf_counter()
        item = self.audio_items[task.audio]
        emission_key, samples, frame_start, frame_end = self._emission_artifact(task, item)
        samples = self._apply_audio_injection(task, samples, frame_start)

        if self.store.has("emission", emission_key):
            artifact = self.store.load("emission", emission_key)
            self._log_reuse(task, "emission", emission_key, "reused")
            emissions = np_array(artifact["emissions"])
            point_costs = np_array(artifact["point_costs"], dtype="int64")
        else:
            feature = features_mod.compute_emissions(
                samples, self.score_params, self.schedule_config, task.injection
            )
            emissions = feature["emissions"]
            point_costs = feature["point_costs"]
            artifact = {
                "audio": task.audio,
                "shard_index": task.shard_index,
                "frame_start": frame_start,
                "frame_end": frame_end,
                "emissions": emissions.tolist(),
                "point_costs": point_costs.tolist(),
                "last_padded": feature["last_padded"].tolist(),
            }
            self.store.store("emission", emission_key, artifact)
            self._log_reuse(task, "emission", emission_key, "computed")

        decode_key = fp.decode_cache_key(
            emission_key, self.lm_hash, self.schedule_config["quant_mode"],
            self.schedule_config["quant_frac_bits"], task.shard_index)
        checkpoint = self.checkpoints.get(self._task_id(task))
        full_cost = int(point_costs.sum())
        already = checkpoint["consumed"] if checkpoint else 0
        can_finish = (allowance + already) >= full_cost
        if (self.store.has("decode", decode_key) and checkpoint is None
                and can_finish):
            decoded_payload = self.store.load("decode", decode_key)
            self._log_reuse(task, "decode", decode_key, "reused")
            decoded = decoded_payload["decoded"]
            incremental = full_cost
            cumulative = full_cost
            resume = None
        else:
            effective_allowance = allowance
            if task.injection and task.injection["type"] == "quota_below_min_points":
                effective_allowance = int(task.injection.get("forced_allowance", 1))
            decoded, incremental, cumulative, resume = decoder_mod.run_viterbi(
                emissions, point_costs, self.score_params, effective_allowance,
                checkpoint=checkpoint, injection=task.injection,
                global_limited=getattr(self, "_current_global_limited", False),
            )
            if resume is None:
                payload = {
                    "audio": task.audio,
                    "shard_index": task.shard_index,
                    "frame_offset": task.frame_offset,
                    "decoded": decoded,
                    "point_cost": cumulative,
                }
                self.store.store("decode", decode_key, payload)
                self._log_reuse(task, "decode", decode_key, "computed")
            else:
                self.checkpoints[self._task_id(task)] = resume

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.shard_timings[self._task_id(task)] = round(elapsed_ms, 4)
        return {
            "task": task,
            "decoded": decoded,
            "consumed": incremental,
            "cumulative": cumulative,
            "resume": resume,
            "decode_key": decode_key,
            "emission_key": emission_key,
        }

    @staticmethod
    def _task_id(task):
        return f"{task.audio}#{task.shard_index}"

    def _log_reuse(self, task, stage, key, action):
        self.reuse_log.append({
            "audio": task.audio,
            "shard_index": task.shard_index,
            "stage": stage,
            "action": action,
            "cache_key": key,
            "injection_id": task.injection["id"] if task.injection else None,
        })

    def _reconcile_completed(self):
        """Drop restored completed entries whose derived cache keys are stale.

        A changed audio byte, overlap, backoff weight or quant mode changes the
        emission/decode cache key; such shards must be recomputed even though a
        previous snapshot marked them complete.
        """
        if not self.completed:
            return
        stale_ids = []
        for task in self.tasks:
            task_id = self._task_id(task)
            entry = self.completed.get(task_id)
            if entry is None:
                continue
            item = self.audio_items.get(task.audio)
            if item is None:
                stale_ids.append(task_id)
                continue
            emission_key, _, frame_start, frame_end = self._emission_artifact(task, item)
            decode_key = fp.decode_cache_key(
            emission_key, self.lm_hash, self.schedule_config["quant_mode"],
            self.schedule_config["quant_frac_bits"], task.shard_index)
            if (decode_key != entry.get("decode_key")
                    or emission_key != entry.get("emission_key")
                    or not self.store.has("decode", decode_key)):
                stale_ids.append(task_id)
        for task_id in stale_ids:
            self.completed.pop(task_id, None)
            self.checkpoints.pop(task_id, None)

    # --------------------------------------------------------------- scheduler
    def run(self):
        from concurrent.futures import ThreadPoolExecutor

        max_workers = 1 if self.serial else self.schedule_config["max_concurrency"]
        batch_size = self.schedule_config["shards_per_batch"]
        commit_every = self.schedule_config["snapshot_commit_shards"]
        since_commit = 0

        remaining_global = self.schedule_config["global_point_budget"]
        self._reconcile_completed()
        todo = [task for task in self.tasks if self._task_id(task) not in self.completed]
        # Pending checkpoints from a previous process always go first.
        todo.sort(key=lambda task: (
            0 if self._task_id(task) in self.checkpoints else 1,
            task.sort_key,
        ))

        while todo:
            batch = todo[:batch_size]
            todo = todo[batch_size:]
            plans = self._plan_batch(batch, remaining_global)
            results = []
            if self.serial:
                for plan in plans:
                    results.append(self._execute_plan(plan))
            else:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [executor.submit(self._execute_plan, plan) for plan in plans]
                    for future in futures:
                        results.append(future.result())
            # Apply outcomes in the stable shard order, independent of finish time.
            for plan, outcome in zip(plans, results):
                remaining_global = self._apply_outcome(plan, outcome, remaining_global)
                since_commit += 1
                if since_commit >= commit_every:
                    self._commit_snapshot(remaining_global)
                    since_commit = 0

            failed_ids = {
                (f["audio"], f["shard_index"]) for f in self.failures
            }
            still_todo = [task for task in todo
                          if self._task_id(task) not in self.completed]
            # Pending items jump to the front of the next batch; failures drop out.
            pending_tasks = [
                plan[0] for plan in plans
                if self._task_id(plan[0]) not in self.completed
                and (plan[0].audio, plan[0].shard_index) not in failed_ids
            ]
            todo = pending_tasks + still_todo
            if remaining_global <= 0:
                # Global budget exhausted within this run; the pending queue is
                # persisted and resumed on the next run with a fresh budget.
                break
            if not pending_tasks:
                # No recoverable work remains from this batch.
                todo = still_todo

        self._commit_snapshot(remaining_global)
        return remaining_global

    def _plan_batch(self, batch, remaining_global):
        # Reserve budget in stable shard order; each shard sees the residual
        # global budget left by every earlier shard, exactly like serial order.
        plans = []
        residual = remaining_global
        for task in batch:
            checkpoint = self.checkpoints.get(self._task_id(task))
            already = checkpoint["consumed"] if checkpoint else 0
            shard_cap = self.schedule_config["shard_point_quota"]
            headroom = shard_cap - already
            allowance = min(headroom, max(0, residual))
            # The global budget binds this shard whenever it is the tighter
            # constraint; such a shard must remain PENDING if it cannot finish.
            global_limited = residual <= headroom
            plans.append((task, allowance, global_limited))
            residual = max(0, residual - allowance)
        return plans

    def _execute_plan(self, plan):
        task, allowance, global_limited = plan
        try:
            self._current_global_limited = global_limited
            return ("ok", self.process_shard(task, allowance))
        except PipelineError as exc:
            return ("error", exc)
        except Exception as exc:  # defensive isolation per shard
            from .errors import E_DECODE_FAILED
            return ("error", PipelineError(E_DECODE_FAILED, repr(exc), stage="decode"))

    def _apply_outcome(self, plan, outcome, remaining_global):
        task, allowance, global_limited = plan
        kind, payload = outcome
        if kind == "error":
            exc = payload
            self.failures.append(failure_record(
                task.audio, task.shard_index, exc.stage or "decode",
                exc.code, str(exc), exc.injection_id))
            self.checkpoints.pop(self._task_id(task), None)
            self.ledger.append({
                "audio": task.audio,
                "shard_index": task.shard_index,
                "allocated": allowance,
                "consumed": 0,
                "remaining_after": remaining_global,
                "status": exc.code,
            })
            return remaining_global

        result = payload
        if result["resume"] is not None:
            exhausted_global = global_limited
            code = (E_POINT_BUDGET_PENDING if exhausted_global
                    else E_SHARD_BUDGET_EXHAUSTED)
            self.pending.append({
                "audio": task.audio,
                "shard_index": task.shard_index,
                "consumed": result["cumulative"],
                "code": code,
            })
            self.ledger.append({
                "audio": task.audio,
                "shard_index": task.shard_index,
                "allocated": allowance,
                "consumed": result["consumed"],
                "remaining_after": remaining_global - result["consumed"],
                "status": code,
            })
            self.run_consumed += result["consumed"]
            return remaining_global - result["consumed"]

        self.completed[self._task_id(task)] = {
            "audio": task.audio,
            "shard_index": task.shard_index,
            "decode_key": result["decode_key"],
            "emission_key": result["emission_key"],
            "frame_offset": task.frame_offset,
            "point_cost": result["cumulative"],
        }
        self.checkpoints.pop(self._task_id(task), None)
        self.ledger.append({
            "audio": task.audio,
            "shard_index": task.shard_index,
            "allocated": allowance,
            "consumed": result["consumed"],
            "remaining_after": remaining_global - result["consumed"],
            "status": "COMPLETED",
        })
        self.run_consumed += result["consumed"]
        return remaining_global - result["consumed"]

    # --------------------------------------------------------------- snapshot
    def _commit_snapshot(self, remaining_global):
        completed_shards = [
            self.completed[key] for key in sorted(self.completed)
        ]
        manifest = {
            "version": SNAPSHOT_VERSION,
            "config_hash": self.config_hash,
            "audio_fingerprints": {
                name: item["raw_fingerprint"]
                for name, item in sorted(self.audio_items.items())
            },
            "completed_shards": completed_shards,
            "consumed_points": self.schedule_config["global_point_budget"] - remaining_global,
            "remaining_points": remaining_global,
            "pending_queue": [
                {"audio": task.audio, "shard_index": task.shard_index}
                for task in self._pending_tasks()
            ],
            "checkpoints": self.checkpoints,
        }
        self.store.write_manifest(manifest)

    def _pending_tasks(self):
        ids = set(self.checkpoints.keys())
        return [task for task in self.tasks if self._task_id(task) in ids]

    # --------------------------------------------------------------- stitching
    def stitch(self):
        shift_samples = (
            audio_mod.SAMPLE_RATE * self.score_params["frame_shift_ms"] // 1000
        )
        reports = {}
        for name in sorted(self.audio_items):
            shard_tasks = [task for task in self.tasks if task.audio == name]
            shard_tasks.sort(key=lambda task: task.shard_index)
            failed_shards = {f["shard_index"] for f in self.failures if f["audio"] == name}
            if failed_shards:
                # A failed shard disqualifies the whole audio transcript.
                continue
            decoded_shards = []
            decode_keys = []
            for task in shard_tasks:
                entry = self.completed.get(self._task_id(task))
                if entry is None:
                    return None  # pending shard blocks final stitching
                payload = self.store.load("decode", entry["decode_key"])
                decoded_shards.append({
                    "shard_index": task.shard_index,
                    "frame_offset": task.frame_offset,
                    "words": payload["decoded"]["words"],
                })
                decode_keys.append(entry["decode_key"])
            if not decoded_shards:
                continue
            stitch_key = fp.stitch_cache_key(
                decode_keys, self.config_hash, self.schedule_config["overlap_ms"]
            )
            if self.store.has("stitch", stitch_key):
                result = self.store.load("stitch", stitch_key)
                action = "reused"
            else:
                result = stitching_mod.stitch_audio(decoded_shards, shift_samples)
                self.store.store("stitch", stitch_key, result)
                action = "computed"
            self.reuse_log.append({
                "audio": name, "shard_index": None, "stage": "stitch",
                "action": action, "cache_key": stitch_key, "injection_id": None,
            })
            reports[name] = result
        return reports

    # ----------------------------------------------------------------- report
    def build_report(self, transcripts, remaining_global, injected_hits):
        failures_sorted = sorted(self.failures, key=lambda item: item["sort_key"])
        for item in failures_sorted:
            item.pop("sort_key", None)
        reuse_sorted = sorted(
            self.reuse_log,
            key=lambda item: (item["audio"],
                              -1 if item["shard_index"] is None else item["shard_index"],
                              item["stage"]),
        )
        pending_sorted = sorted(self.pending, key=lambda item: (item["audio"], item["shard_index"]))
        return {
            "config_hash": self.config_hash,
            "mode": "serial" if self.serial else "parallel",
            "max_concurrency": self.schedule_config["max_concurrency"],
            "global_point_budget": self.schedule_config["global_point_budget"],
            "remaining_points": remaining_global,
            "consumed_points": self.run_consumed,
            "shards_total": len(self.tasks),
            "shards_completed": len(self.completed),
            "shards_pending": len(pending_sorted),
            "shards_failed": len(failures_sorted),
            "transcripts": transcripts or {},
            "budget_ledger": self.ledger,
            "pending_queue": pending_sorted,
            "failures": failures_sorted,
            "reuse_recompute_log": reuse_sorted,
            "shard_timings_ms": {
                key: self.shard_timings[key] for key in sorted(self.shard_timings)
            },
            "snapshot_events": self.snapshot_events,
            "injections_hit": injected_hits,
        }


def np_array(values, dtype="float64"):
    import numpy as np
    return np.asarray(values, dtype=dtype)


def np_all_finite(values):
    import numpy as np
    return bool(np.all(np.isfinite(values)))
