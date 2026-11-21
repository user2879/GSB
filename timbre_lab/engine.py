from __future__ import annotations

import copy
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .commands import CommandImpact, apply_command
from .config import AnalysisConfig
from .errors import E_INTERNAL, E_PREVIEW_BUDGET_PENDING, BusinessError
from .model import AudioDocument
from .model import find_segment
from .pipeline import CachedPipeline, compare_incremental_with_full
from .snapshot import RestoredSnapshot, SnapshotManager


@dataclass
class CommandJob:
    index: int
    command: dict[str, Any]
    done: threading.Event = field(default_factory=threading.Event)
    applied: threading.Event = field(default_factory=threading.Event)
    result: dict[str, Any] | None = None
    error: BusinessError | None = None
    impact: CommandImpact | None = None
    forecast_affected: set[str] = field(default_factory=set)
    progress: dict[str, Any] = field(
        default_factory=lambda: {"phase": "queued", "completed": 0, "total": 0, "percent": 0.0}
    )

    @property
    def command_id(self) -> str:
        return str(self.command.get("id", f"command-{self.index}"))

    def update_progress(self, phase: str, completed: int, total: int) -> None:
        self.progress = {
            "phase": phase,
            "completed": completed,
            "total": max(total, 0),
            "percent": round(completed / total * 100.0, 2) if total else 100.0,
        }


class IncrementalEngine:
    def __init__(
        self,
        config: AnalysisConfig,
        document: AudioDocument,
        commands: list[dict[str, Any]],
        output_dir: str | Path,
        *,
        overrides: dict[str, int] | None = None,
        failures: list[dict[str, Any]] | None = None,
        command_cursor: int = 0,
        global_version: int = 0,
        restored: RestoredSnapshot | None = None,
        commands_base_dir: str | Path = ".",
        auto_initialize: bool = True,
    ) -> None:
        self.config = config
        self.document = document
        self.commands = commands
        self.output_dir = Path(output_dir)
        self.commands_base_dir = Path(commands_base_dir)
        self.overrides = dict(overrides or {})
        self.failures = list(failures or [])
        self.command_cursor = command_cursor
        self.global_version = global_version
        self.snapshot_manager = SnapshotManager(self.output_dir)
        self.feature_executor = ThreadPoolExecutor(max_workers=config.max_workers, thread_name_prefix="timbre-feature")
        self.command_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="timbre-command")
        self.pipeline = CachedPipeline(config, executor=self.feature_executor)
        if restored is not None:
            self.pipeline.feature_cache = restored.feature_cache
            self.pipeline.description_cache = restored.description_cache
            self.pipeline.assignment_cache = restored.assignment_cache
            self.pipeline.speaker_cache = restored.speaker_cache
        self.lock = threading.RLock()
        self.jobs: list[CommandJob] = []
        self.futures: list[Future[dict[str, Any]]] = []
        self.last_semantic: dict[str, Any] | None = None
        self.last_summary: dict[str, Any] = {
            "completed_segments": 0,
            "speaker_count": 0,
            "vector_dimensions": len(config.dimensions),
            "affected_segments": [],
            "progress": {"phase": "initializing", "completed": 0, "total": 0, "percent": 0.0},
        }
        self.last_commit_cursor = restored.command_cursor if restored is not None else -1
        self.last_snapshot: dict[str, Any] | None = (
            {
                "snapshot": restored.path.name,
                "path": str(restored.path),
                "manifest": dict(restored.manifest),
            }
            if restored is not None
            else None
        )
        self._closed = False
        self._active_job: CommandJob | None = None
        if auto_initialize:
            self.initialize()

    def initialize(self, *, commit: bool = True) -> dict[str, Any]:
        started = time.perf_counter()
        semantic = self.pipeline.compute(self.document, self.overrides, self.failures)
        semantic["initialization"] = {
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
            "incremental_full_equal": self._verify_if_enabled(semantic),
        }
        self.last_semantic = semantic
        self.last_summary = self.summarize(semantic, [], {"phase": "complete", "completed": 1, "total": 1, "percent": 100.0})
        if commit:
            self.commit_snapshot()
        return semantic

    def _verify_if_enabled(self, semantic: dict[str, Any]) -> bool | None:
        if not self.config.verify_incremental_vs_full:
            return None
        comparison = compare_incremental_with_full(
            semantic,
            self.document,
            self.config,
            self.overrides,
            self.failures,
        )
        semantic["full_rebuild_comparison"] = comparison
        return bool(comparison["equal"])

    def summarize(
        self,
        semantic: dict[str, Any],
        affected_segments: Iterable[str],
        progress: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "completed_segments": len(semantic.get("segments", [])),
            "speaker_count": len(semantic.get("speaker_vectors", [])),
            "vector_dimensions": len(self.config.dimensions),
            "affected_segments": sorted(set(affected_segments)),
            "progress": progress,
            "command_cursor": self.command_cursor,
        }

    def submit(self, command: dict[str, Any]) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("engine is closed")
        with self.lock:
            index = self.command_cursor + sum(1 for item in self.jobs if not item.done.is_set()) + 1
            job = CommandJob(index=index, command=command, forecast_affected=self._forecast_affected(command))
            self.jobs.append(job)
            future = self.command_executor.submit(self._process_job, job)
            self.futures.append(future)
        deadline = time.perf_counter() + max(0.0, self.config.preview_budget_ms / 1000.0)
        remaining = max(0.0, deadline - time.perf_counter())
        job.done.wait(timeout=remaining)
        if job.done.is_set():
            assert job.result is not None
            return job.result
        return self.pending_preview(job)

    def pending_preview(self, job: CommandJob) -> dict[str, Any]:
        with self.lock:
            latest = copy.deepcopy(self.last_summary)
            if job.impact:
                affected = sorted(job.impact.affected_segments)
                affected_source = "applied_plan"
            elif job.forecast_affected:
                affected = sorted(job.forecast_affected)
                affected_source = "forecast"
            else:
                affected = list(latest.get("affected_segments", []))
                affected_source = "latest_completed"
            progress = dict(job.progress)
            queue_depth = sum(1 for item in self.jobs if not item.done.is_set())
        return {
            "status": "pending",
            "code": E_PREVIEW_BUDGET_PENDING,
            "command_id": job.command_id,
            "command_index": job.index,
            "latest_completed": latest,
            "preview": {
                "completed_segments": latest.get("completed_segments", 0),
                "speaker_count": latest.get("speaker_count", 0),
                "vector_dimensions": len(self.config.dimensions),
                "affected_segments": affected,
                "affected_segments_source": affected_source,
                "progress": progress,
                "queue_depth": queue_depth,
            },
        }

    def _forecast_affected(self, command: dict[str, Any]) -> set[str]:
        op = str(command.get("op", ""))
        if op in {"set_quantization", "set_quantization_level"}:
            return {segment.segment_id for segment in self.document.segments}
        selector = command.get("target", command.get("selector"))
        if not isinstance(selector, dict):
            return set()
        try:
            target = find_segment(self.document, selector)
        except KeyError:
            return set()
        affected = {target.segment_id}
        if op == "merge":
            same_file = self.document.segments_for_file(target.file)
            index = next(index for index, segment in enumerate(same_file) if segment.segment_id == target.segment_id)
            direction = str(command.get("direction", "next"))
            if direction == "next" and index + 1 < len(same_file):
                affected.add(same_file[index + 1].segment_id)
            elif direction == "previous" and index > 0:
                affected.add(same_file[index - 1].segment_id)
        return affected

    def _process_job(self, job: CommandJob) -> dict[str, Any]:
        with self.lock:
            self._active_job = job
            job.update_progress("applying", 0, 1)
            document_backup = self.document.clone()
            overrides_backup = copy.deepcopy(self.overrides)
            failures_backup = copy.deepcopy(self.failures)
            feature_backup = dict(self.pipeline.feature_cache)
            description_backup = dict(self.pipeline.description_cache)
            assignment_backup = dict(self.pipeline.assignment_cache)
            speaker_backup = dict(self.pipeline.speaker_cache)
        started = time.perf_counter()
        try:
            with self.lock:
                impact = apply_command(
                    self.document,
                    self.config,
                    self.overrides,
                    job.command,
                    self.commands_base_dir,
                )
            job.impact = impact
            job.applied.set()
            segment_count = len(self.document.segments)
            pair_count = max(0, segment_count - len(self.document.files))
            total = segment_count * 3 + pair_count
            job.update_progress("feature", 0, total)

            def progress_callback(phase: str, completed: int, total_tasks: int) -> None:
                job.update_progress(phase, completed, max(total, total_tasks))

            self.pipeline.progress = progress_callback
            semantic = self.pipeline.compute(
                self.document,
                self.overrides,
                self.failures,
                affected_segments=impact.affected_segments,
                affected_stages=impact.affected_stages,
            )
            comparison_bool = self._verify_if_enabled(semantic)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            command_result = {
                "command": copy.deepcopy(job.command),
                "impact": impact.as_dict(),
                "elapsed_ms": elapsed_ms,
                "incremental_vs_full_equal": comparison_bool,
                "full_rebuild_comparison": copy.deepcopy(semantic.get("full_rebuild_comparison")),
            }
            semantic["command_history"] = [command_result]
            with self.lock:
                self.command_cursor = job.index
                self.global_version = job.index
                self.last_semantic = semantic
                self.last_summary = self.summarize(
                    semantic,
                    impact.affected_segments,
                    {"phase": "complete", "completed": total, "total": total, "percent": 100.0},
                )
                self._maybe_commit(semantic)
            result = {
                "status": "complete",
                "code": None,
                "command_id": job.command_id,
                "command_index": job.index,
                "command": copy.deepcopy(job.command),
                "impact": impact.as_dict(),
                "preview": copy.deepcopy(self.last_summary),
                "incremental_vs_full_equal": comparison_bool,
                "full_rebuild_comparison": copy.deepcopy(semantic.get("full_rebuild_comparison")),
                "elapsed_ms": elapsed_ms,
                "semantic": semantic,
            }
            job.result = result
            job.update_progress("complete", total, total)
            return result
        except BusinessError as exc:
            self._restore_after_error(
                document_backup,
                overrides_backup,
                failures_backup,
                feature_backup,
                description_backup,
                assignment_backup,
                speaker_backup,
            )
            with self.lock:
                self.command_cursor = job.index
                self.global_version = job.index
                command_failure = {
                    **exc.as_dict(),
                    "command_id": job.command_id,
                    "command": copy.deepcopy(job.command),
                    "isolated": True,
                }
                self.failures.append(command_failure)
                if self.last_semantic is None:
                    self.initialize(commit=False)
                assert self.last_semantic is not None
                self.last_semantic = copy.deepcopy(self.last_semantic)
                self.last_semantic["failures"] = copy.deepcopy(self.failures)
                self.last_semantic.setdefault("command_failures", []).append(command_failure)
                self.last_summary = self.summarize(
                    self.last_semantic,
                    job.impact.affected_segments if job.impact else [],
                    {"phase": "failed", "completed": 1, "total": 1, "percent": 100.0},
                )
                self._maybe_commit(self.last_semantic, force=True)
            result = {
                "status": "failed",
                "code": exc.code,
                "command_id": job.command_id,
                "command_index": job.index,
                "command": copy.deepcopy(job.command),
                "message": exc.message,
                "details": exc.details,
                "preview": copy.deepcopy(self.last_summary),
                "semantic": self.last_semantic,
            }
            job.error = exc
            job.result = result
            job.update_progress("failed", 1, 1)
            return result
        except Exception as exc:  # unexpected failures remain isolated per command
            self._restore_after_error(
                document_backup,
                overrides_backup,
                failures_backup,
                feature_backup,
                description_backup,
                assignment_backup,
                speaker_backup,
            )
            business = BusinessError(E_INTERNAL, str(exc))
            with self.lock:
                self.command_cursor = job.index
                self.global_version = job.index
                self.failures.append({**business.as_dict(), "command_id": job.command_id, "isolated": True})
                if self.last_semantic is None:
                    self.initialize(commit=False)
                self.last_semantic = copy.deepcopy(self.last_semantic)
                self.last_semantic["failures"] = copy.deepcopy(self.failures)
                self.last_summary = self.summarize(self.last_semantic, [], {"phase": "failed", "completed": 1, "total": 1, "percent": 100.0})
                self._maybe_commit(self.last_semantic, force=True)
            result = {
                "status": "failed",
                "code": business.code,
                "command_id": job.command_id,
                "command_index": job.index,
                "message": business.message,
                "preview": copy.deepcopy(self.last_summary),
                "semantic": self.last_semantic,
            }
            job.error = business
            job.result = result
            job.update_progress("failed", 1, 1)
            return result
        finally:
            with self.lock:
                self._active_job = None
            job.done.set()

    def _restore_after_error(
        self,
        document: AudioDocument,
        overrides: dict[str, int],
        failures: list[dict[str, Any]],
        features: dict[str, dict[str, Any]],
        descriptions: dict[str, dict[str, Any]],
        assignments: dict[str, dict[str, Any]],
        speakers: dict[str, dict[str, Any]],
    ) -> None:
        with self.lock:
            self.document = document
            self.overrides = overrides
            self.failures = failures
            self.pipeline.feature_cache = features
            self.pipeline.description_cache = descriptions
            self.pipeline.assignment_cache = assignments
            self.pipeline.speaker_cache = speakers

    def _maybe_commit(self, semantic: dict[str, Any], force: bool = False) -> None:
        due = self.command_cursor % self.config.snapshot_commit_every == 0
        final_command = self.command_cursor >= len(self.commands)
        if not force and not due and not final_command:
            return
        if self.last_commit_cursor == self.command_cursor and not force:
            return
        self.commit_snapshot()

    def commit_snapshot(self) -> dict[str, Any]:
        result = self.snapshot_manager.commit(
            config=self.config,
            document=self.document,
            overrides=self.overrides,
            failures=self.failures,
            command_cursor=self.command_cursor,
            commands=self.commands,
            global_version=self.global_version,
            feature_cache=self.pipeline.feature_cache,
            description_cache=self.pipeline.description_cache,
            assignment_cache=self.pipeline.assignment_cache,
            speaker_cache=self.pipeline.speaker_cache,
        )
        self.last_commit_cursor = self.command_cursor
        self.last_snapshot = result
        return result

    def join(self, timeout: float | None = None) -> list[dict[str, Any]]:
        deadline = None if timeout is None else time.perf_counter() + timeout
        results: list[dict[str, Any]] = []
        for future in self.futures:
            remaining = None if deadline is None else max(0.0, deadline - time.perf_counter())
            result = future.result(timeout=remaining)
            results.append(result)
        return results

    def wait_for_idle(self, timeout: float | None = None) -> None:
        self.join(timeout=timeout)

    def close(self) -> None:
        if self._closed:
            return
        self.join()
        self._closed = True
        self.command_executor.shutdown(wait=True)
        self.feature_executor.shutdown(wait=True)

    def __enter__(self) -> "IncrementalEngine":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
