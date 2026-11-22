from __future__ import annotations

import json
import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .cache import CacheStore, package_entry, package_from_entry
from .config import RemixConfig
from .conversion import ConversionResult, conversion_cache_key, convert_package
from .errors import DiagnosticError, sort_errors
from .hashing import hash_directory
from .model import ParsedPackage, Track
from .package import (
    package_source_fingerprint,
    parse_cache_key,
    parse_package,
)
from .render import (
    RenderResult,
    TaskPool,
    build_segments,
    render,
)
from .snapshot import SnapshotStore
from .state import ClipRuntime, WorkbenchState
from .wavio import read_pcm16_wav


@dataclass
class CommandJob:
    command_id: str
    command: dict[str, Any]
    requested_version: int
    budget_ms: float
    event: threading.Event = field(default_factory=threading.Event)
    result: RenderResult | None = None
    error: DiagnosticError | None = None
    started: float = 0.0
    completed: float = 0.0


class Workbench:
    def __init__(
        self,
        data_root: str | Path,
        runtime_root: str | Path,
        config: RemixConfig | str | Path,
        *,
        resume: bool = False,
        expected_global_version: str | None = None,
        start_worker: bool = True,
    ) -> None:
        self.data_root = Path(data_root).resolve()
        self.runtime_root = Path(runtime_root).resolve()
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        self.config = config if isinstance(config, RemixConfig) else RemixConfig.from_path(config)
        self.config.validate()
        self.snapshot_config_hash = self.config.config_hash()
        self.sample_manifest = self._load_sample_manifest()
        self.clip_ids = tuple(self.sample_manifest.get("clip_ids", []))
        if not self.clip_ids:
            self.clip_ids = tuple(
                sorted(path.name for path in (self.data_root / "segments").iterdir() if path.is_dir())
            )
        self.input_fingerprint = str(
            self.sample_manifest.get("content_sha256")
            or hash_directory(self.data_root / "segments")
        )
        self.snapshot = SnapshotStore(self.runtime_root / "snapshot")
        self.global_version = self.snapshot.compute_global_version(
            self.input_fingerprint, self.snapshot_config_hash
        )
        audit_expected_version = expected_global_version or self.global_version
        self.snapshot_audit = self.snapshot.audit(
            expected_global_version=audit_expected_version,
            expected_config_hash=self.snapshot_config_hash,
        )
        self.startup_failures = [
            DiagnosticError(**{
                "code": item["code"],
                "message": item["message"],
                "stage": item.get("stage", "snapshot"),
                "format_version": item.get("format_version", ""),
                "clip_id": item.get("clip_id", ""),
                "track": item.get("track", ""),
                "details": item.get("details", {}),
            })
            for item in self.snapshot_audit.get("errors", [])
        ]
        self.cache = CacheStore()
        self.resume_requested = resume
        if resume and self.snapshot_audit.get("reusable"):
            self.cache = self.snapshot.load_cache(self.snapshot_audit)
        self.parse_stats: dict[str, Any] = {
            "cache_hits": 0,
            "parsed": 0,
            "failures": 0,
            "max_observed_workers": 1,
            "elapsed_ms": 0.0,
        }
        self.state = self._load_initial_state()
        self.latest_result = render(
            self.state,
            self.cache,
            previous_state=None,
            retain_frame_details=True,
            simulated_compute_ms=0.0,
        )
        self.latest_result.stage_times_ms["parse"] = self.parse_stats["elapsed_ms"]
        self._commit_snapshot(self.latest_result, self.state.param_version)
        self.budget_hits = 0
        self.responses: list[dict[str, Any]] = []
        self.completed_jobs: list[CommandJob] = []
        self._queue: queue.Queue[CommandJob | None] = queue.Queue()
        self._latest_lock = threading.RLock()
        self._progress_lock = threading.Lock()
        self._progress: dict[str, Any] = {
            "param_version": self.state.param_version,
            "completed_segments": len(self.latest_result.segments),
            "total_segments": len(self.latest_result.segments),
        }
        self._worker: threading.Thread | None = None
        self._requested_version = self.state.param_version
        self._closed = False
        if start_worker:
            self._worker = threading.Thread(target=self._worker_loop, name="stemscope-interactive", daemon=True)
            self._worker.start()

    def _load_sample_manifest(self) -> dict[str, Any]:
        path = self.data_root / "generator_manifest.json"
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _load_initial_state(self) -> WorkbenchState:
        started = time.perf_counter()
        clips: dict[str, ClipRuntime] = {}
        failures: list[DiagnosticError] = list(self.startup_failures)
        pool = TaskPool(self.config.workers)
        futures: dict[str, Future[Any]] = {}
        for clip_id in self.clip_ids:
            segment_dir = self.data_root / "segments" / clip_id
            source_fp = package_source_fingerprint(segment_dir, self.config.format_version)
            key = parse_cache_key(self.config.format_version, source_fp)
            cached = self.cache.get("parse", key)
            if cached is not None:
                try:
                    package = package_from_entry(cached)
                    clips[clip_id] = ClipRuntime.from_package(clip_id, package)
                    self.parse_stats["cache_hits"] += 1
                    continue
                except Exception as exc:
                    failures.append(
                        DiagnosticError(
                            "E_TRACK_PARSE_FAILED",
                            f"cannot restore parse cache for {clip_id}: {exc}",
                            stage="parse",
                            format_version=self.config.format_version,
                            clip_id=clip_id,
                        )
                    )
            futures[clip_id] = pool.submit(
                parse_package,
                segment_dir,
                self.config.format_version,
                clip_id=clip_id,
            )
        for clip_id in sorted(futures):
            parsed = futures[clip_id].result()
            if parsed.ok and parsed.package is not None:
                package = parsed.package
                clips[clip_id] = ClipRuntime.from_package(clip_id, package)
                entry = package_entry(
                    package,
                    f"parse|{clip_id}|{self.config.format_version}",
                    stage="parse",
                    key=package.parse_key,
                )
                self.cache.put(entry)
                self.parse_stats["parsed"] += 1
            else:
                failures.extend(parsed.failures)
                self.parse_stats["failures"] += len(parsed.failures)
        pool.shutdown()
        self.parse_stats["max_observed_workers"] = pool.max_observed
        self.parse_stats["elapsed_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
        return WorkbenchState(
            config=self.config.clone(),
            clips=clips,
            param_version=0,
            format_history=[self.config.format_version],
            conversion_records=[],
            replacement_records=[],
            failures=[error.to_dict() for error in sort_errors(failures)],
        )

    def _commit_snapshot(self, result: RenderResult, param_version: int) -> None:
        state = self.state
        completed = [item["segment_index"] for item in result.segments]
        self.snapshot.checkpoint(
            self.cache,
            global_version=self.global_version,
            config_hash=self.snapshot_config_hash,
            input_fingerprint=self.input_fingerprint,
            format_versions=state.format_versions,
            completed_segments=completed,
            stages=("parse", "convert", "gain", "mix"),
            extra={
                "param_version": param_version,
                "format_version": state.config.format_version,
                "reused_tasks": len(result.reused),
                "recomputed_tasks": len(result.recomputed),
            },
        )

    def _checkpoint_callback(self, param_version: int, result_holder: dict[str, Any]):
        def callback(cache: CacheStore, completed_segments: list[int]) -> None:
            self.snapshot.checkpoint(
                cache,
                global_version=self.global_version,
                config_hash=self.snapshot_config_hash,
                input_fingerprint=self.input_fingerprint,
                format_versions=self.pending_format_versions,
                completed_segments=completed_segments,
                stages=("gain", "mix"),
                extra={"param_version": param_version, "partial": True},
            )
            result_holder["checkpoints"] = int(result_holder.get("checkpoints", 0)) + 1

        return callback

    def _set_progress(self, param_version: int, completed: int, total: int) -> None:
        with self._progress_lock:
            self._progress = {
                "param_version": param_version,
                "completed_segments": completed,
                "total_segments": total,
                "progress": completed / total if total else 0.0,
            }

    def _progress_info(self) -> dict[str, Any]:
        with self._progress_lock:
            return dict(self._progress)

    def _latest_summary(self) -> dict[str, Any]:
        with self._latest_lock:
            summary = self.latest_result.summary()
        summary["is_latest"] = True
        return summary

    def submit(self, command: dict[str, Any], *, budget_ms: float | None = None) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("workbench is closed")
        started = time.perf_counter()
        budget = float(budget_ms if budget_ms is not None else self.config.interaction_budget_ms)
        self._requested_version += 1
        job = CommandJob(
            command_id=str(command.get("id", f"cmd_{self._requested_version:03d}")),
            command=dict(command),
            requested_version=self._requested_version,
            budget_ms=budget,
            started=started,
        )
        baseline_summary = self._latest_summary()
        self._queue.put(job)
        budget_seconds = budget / 1000.0
        if budget_seconds < 0.5:
            reserve = budget_seconds * 0.35
        elif budget_seconds < 2.0:
            reserve = budget_seconds * 0.2 + 0.02
        else:
            reserve = budget_seconds * 0.1 + 0.03
        reserve = min(reserve, max(0.0, budget_seconds * 0.75))
        remaining = max(0.0, budget_seconds - (time.perf_counter() - started) - reserve)
        completed = job.event.wait(remaining)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if completed and job.error is None and job.result is not None:
            summary = job.result.summary()
            summary["is_latest"] = True
            response = {
                "command_id": job.command_id,
                "requested_param_version": job.requested_version,
                "status": "completed",
                "code": None,
                "is_latest": True,
                "budget_ms": budget,
                "elapsed_ms": round(elapsed_ms, 3),
                "summary": summary,
            }
        elif completed and job.error is not None:
            summary = baseline_summary
            summary["is_latest"] = False
            response = {
                "command_id": job.command_id,
                "requested_param_version": job.requested_version,
                "status": "failed",
                "code": job.error.code,
                "is_latest": False,
                "budget_ms": budget,
                "elapsed_ms": round(elapsed_ms, 3),
                "summary": summary,
                "error": job.error.to_dict(),
            }
        else:
            self.budget_hits += 1
            with self._latest_lock:
                candidate = self.latest_result.summary()
            if candidate["param_version"] < job.requested_version:
                summary = candidate
            else:
                summary = baseline_summary
            summary["is_latest"] = False
            response = {
                "command_id": job.command_id,
                "requested_param_version": job.requested_version,
                "status": "budget_exceeded",
                "code": "E_INTERACT_BUDGET_EXCEEDED",
                "is_latest": False,
                "budget_ms": budget,
                "elapsed_ms": round(elapsed_ms, 3),
                "summary": summary,
                "observed_progress": self._progress_info(),
                "stale_result_param_version": summary["param_version"],
                "message": "latest completed summary returned before the requested parameter version completed",
            }
        self.responses.append(response)
        return response

    def _worker_loop(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                self._queue.task_done()
                return
            try:
                self._process_job(job)
            finally:
                job.event.set()
                self._queue.task_done()

    def _process_job(self, job: CommandJob) -> None:
        previous_state = self.state
        new_state = previous_state.clone()
        new_state.param_version = job.requested_version
        self._set_progress(
            job.requested_version,
            0,
            len(build_segments(new_state.clips, new_state.config.render_frames, new_state.config.sample_rate)),
        )
        try:
            new_state = self.apply_command_to_state(previous_state, new_state, job.command, job.requested_version)
            self.pending_format_versions = new_state.format_versions
            holder: dict[str, Any] = {}
            simulated = float(job.command.get("budget_probe", {}).get("simulated_compute_ms", 0.0))

            def progress(completed: int, total: int) -> None:
                self._set_progress(job.requested_version, completed, total)

            result = render(
                new_state,
                self.cache,
                previous_state=previous_state,
                retain_frame_details=False,
                checkpoint_callback=self._checkpoint_callback(job.requested_version, holder),
                progress_callback=progress,
                simulated_compute_ms=simulated,
            )
            result.stage_times_ms["parse"] = 0.0
            with self._latest_lock:
                self.state = new_state
                self.latest_result = result
            self._commit_snapshot(result, job.requested_version)
            self._set_progress(job.requested_version, len(result.segments), len(result.segments))
            job.result = result
            job.completed = time.perf_counter()
        except DiagnosticError as exc:
            job.error = exc
        except Exception as exc:
            job.error = DiagnosticError(
                "E_TRACK_PARSE_FAILED",
                f"interaction command failed: {exc}",
                stage="interaction",
                details={"command": job.command},
            )
        finally:
            self.completed_jobs.append(job)

    def apply_command_to_state(
        self,
        previous_state: WorkbenchState,
        new_state: WorkbenchState,
        command: dict[str, Any],
        param_version: int,
    ) -> WorkbenchState:
        new_state.param_version = param_version
        command_type = command.get("type")
        new_state.config = previous_state.config.apply_command(command)
        new_state.format_history = list(previous_state.format_history)
        new_state.conversion_records = [dict(item) for item in previous_state.conversion_records]
        new_state.replacement_records = [dict(item) for item in previous_state.replacement_records]
        new_state.clips = {clip_id: clip.clone() for clip_id, clip in previous_state.clips.items()}
        new_state.failures = [dict(item) for item in previous_state.failures]
        if command_type == "switch_format":
            self._switch_format(new_state, previous_state, str(command["format"]), param_version)
        elif command_type == "replace_track":
            self._replace_track(new_state, command, param_version)
        return new_state

    def _switch_format(
        self,
        new_state: WorkbenchState,
        previous_state: WorkbenchState,
        target_format: str,
        param_version: int,
    ) -> None:
        pool = TaskPool(new_state.config.workers)

        def timed_convert(*args: Any, **kwargs: Any) -> ConversionResult:
            started = time.perf_counter()
            result = convert_package(*args, **kwargs)
            result.record["elapsed_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
            return result

        futures: dict[str, Future[ConversionResult]] = {}
        cached_results: dict[str, ParsedPackage] = {}
        keys: dict[str, str] = {}
        for clip_id, clip in sorted(previous_state.clips.items()):
            key = conversion_cache_key(clip.package, target_format)
            keys[clip_id] = key
            cached = self.cache.get("convert", key)
            if cached is not None:
                try:
                    cached_results[clip_id] = package_from_entry(cached)
                    continue
                except Exception:
                    pass
            destination = (
                self.runtime_root
                / "conversions"
                / f"p{param_version:06d}"
                / clip_id
            )
            futures[clip_id] = pool.submit(
                timed_convert,
                clip.package,
                target_format,
                destination,
                clip_id=clip_id,
            )
        converted: dict[str, ParsedPackage] = dict(cached_results)
        records: list[dict[str, Any]] = []
        failures: list[DiagnosticError] = []
        for clip_id in sorted(cached_results):
            source = previous_state.clips[clip_id].package
            from .conversion import lost_fields_for_target

            records.append(
                {
                    "conversion_key": keys[clip_id],
                    "source_format": source.format_version,
                    "target_format": target_format,
                    "source_package_fingerprint": source.package_fingerprint,
                    "target_package_fingerprint": cached_results[clip_id].package_fingerprint,
                    "status": "cache_reused",
                    "elapsed_ms": 0.0,
                    "losses": lost_fields_for_target(source, target_format),
                }
            )
        for clip_id in sorted(futures):
            result = futures[clip_id].result()
            records.append(result.record)
            if result.ok and result.target_package is not None:
                converted[clip_id] = result.target_package
                entry = package_entry(
                    result.target_package,
                    f"convert|{clip_id}|{target_format}",
                    stage="convert",
                    key=result.conversion_key,
                )
                self.cache.put(entry)
            else:
                failures.extend(result.failures)
        pool.shutdown()
        if failures:
            raise DiagnosticError(
                "E_CONVERSION_FAILED",
                "one or more clips failed format conversion",
                stage="convert",
                details={"failures": [error.to_dict() for error in sort_errors(failures)]},
            )
        new_state.clips = {
            clip_id: ClipRuntime.from_package(clip_id, package)
            for clip_id, package in sorted(converted.items())
        }
        new_state.conversion_records.extend(records)
        if target_format not in new_state.format_history:
            new_state.format_history.append(target_format)

    def _replace_track(
        self,
        new_state: WorkbenchState,
        command: dict[str, Any],
        param_version: int,
    ) -> None:
        track_name = str(command["track"])
        clip_filter = command.get("clip")
        relative_file = Path(str(command["file"]))
        replacement_path = relative_file if relative_file.is_absolute() else self.data_root / relative_file
        replacement = read_pcm16_wav(replacement_path)
        if replacement.channels != 1:
            raise DiagnosticError(
                "E_TRACK_PARSE_FAILED",
                f"replacement track {track_name} must be mono",
                stage="replace",
                track=track_name,
                details={"path": str(replacement_path), "channels": replacement.channels},
            )
        target_clips = [clip_filter] if clip_filter else sorted(new_state.clips)
        for clip_id in target_clips:
            if clip_id not in new_state.clips:
                raise DiagnosticError(
                    "E_CONFIG_INVALID",
                    f"replacement references unknown clip {clip_id}",
                    stage="interaction",
                    clip_id=str(clip_id),
                )
            clip = new_state.clips[clip_id]
            old_track = clip.tracks.get(track_name)
            if old_track is None:
                raise DiagnosticError(
                    "E_CONFIG_INVALID",
                    f"replacement references unknown track {track_name}",
                    stage="interaction",
                    clip_id=clip_id,
                    track=track_name,
                )
            if replacement.sample_rate != clip.sample_rate:
                raise DiagnosticError(
                    "E_TRACK_PARSE_FAILED",
                    f"replacement sample rate {replacement.sample_rate} != {clip.sample_rate}",
                    stage="replace",
                    clip_id=clip_id,
                    track=track_name,
                )
            if replacement.frames != clip.frames:
                raise DiagnosticError(
                    "E_TRACK_PARSE_FAILED",
                    f"replacement frame count {replacement.frames} != {clip.frames}",
                    stage="replace",
                    clip_id=clip_id,
                    track=track_name,
                    details={"replacement_frames": replacement.frames, "clip_frames": clip.frames},
                )
            old_fingerprint = old_track.content_fingerprint()
            new_track = Track(
                name=track_name,
                pcm=replacement.pcm[0].copy(),
                sample_rate=clip.sample_rate,
                source_format=clip.package.format_version,
                metadata=dict(old_track.metadata),
            )
            new_state.clips[clip_id] = clip.with_track(new_track)
            new_state.replacement_records.append(
                {
                    "clip_id": clip_id,
                    "track": track_name,
                    "file": str(relative_file),
                    "old_content_fingerprint": old_fingerprint,
                    "new_content_fingerprint": new_track.content_fingerprint(),
                    "frames": new_track.frames,
                }
            )

    def wait_for_idle(self) -> None:
        self._queue.join()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._worker is not None:
            self._queue.put(None)
            self._queue.join()
            self._worker.join(timeout=10)

    def render_full_from_blank(self, state: WorkbenchState | None = None) -> RenderResult:
        target = state or self.state
        return render(
            target,
            CacheStore(),
            previous_state=None,
            retain_frame_details=False,
            simulated_compute_ms=0.0,
        )


def compare_render_results(incremental: RenderResult, full: RenderResult) -> dict[str, Any]:
    incremental_keys = sorted(
        {(item["stage"], item["cache_key"]) for item in incremental.reused + incremental.recomputed}
    )
    full_keys = sorted(
        {(item["stage"], item["cache_key"]) for item in full.reused + full.recomputed}
    )
    return {
        "canonical_equal": incremental.canonical_payload() == full.canonical_payload(),
        "pcm_equal": incremental.pcm_sha256 == full.pcm_sha256,
        "segments_equal": incremental.segments == full.segments,
        "frame_details_equal": incremental.frame_detail_sha256 == full.frame_detail_sha256,
        "failures_equal": [item.to_dict() for item in incremental.failures]
        == [item.to_dict() for item in full.failures],
        "task_coverage_equal": incremental_keys == full_keys,
        "incremental_task_count": len(incremental_keys),
        "full_task_count": len(full_keys),
        "incremental_reused": len(incremental.reused),
        "incremental_recomputed": len(incremental.recomputed),
        "full_recomputed": len(full.recomputed),
        }
