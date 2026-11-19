"""Extraction engine: concurrency, budgeted interaction, snapshots and reports."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable

from .cache import IncrementalCache
from .commands import CommandEffect
from .config import RuleConfig
from .documents import Document
from .errors import ErrorCode, Failure
from .pipeline import DocumentResult
from .snapshots import SnapshotStore

WorkHook = Callable[[str, str], None]


@dataclass
class InteractionResponse:
    command_id: str
    status: str
    code: str | None
    summary: dict
    affected_docs: list[str]
    progress: dict
    latest: bool
    elapsed_ms: float

    def as_dict(self) -> dict:
        return {
            "command_id": self.command_id,
            "status": self.status,
            "code": self.code,
            "summary": self.summary,
            "affected_docs": list(self.affected_docs),
            "progress": dict(self.progress),
            "latest": self.latest,
            "elapsed_ms": round(self.elapsed_ms, 6),
        }


class ExtractionEngine:
    def __init__(
        self,
        config: RuleConfig,
        documents: dict[str, Document],
        failures: list[Failure] | None = None,
        snapshot_store: SnapshotStore | None = None,
        work_hook: WorkHook | None = None,
    ) -> None:
        self.config = config
        self.documents: dict[str, Document] = dict(documents)
        self.failures: list[Failure] = list(failures or [])
        self.cache = IncrementalCache()
        self.snapshot_store = snapshot_store
        self.work_hook = work_hook
        self._lock = threading.Lock()
        self._results: dict[str, DocumentResult] = {}
        self._dirty: set[str] = set()
        self._commands_since_commit = 0
        self._run_counter = 0
        self.diagnostic: dict = {
            "runs": [],
            "band_distribution": {},
            "boundary_samples": [],
            "reuse": {"reused": 0, "recomputed": 0, "invalidated": 0},
            "invalidation_log": [],
            "timing_ms": {},
        }

    def sort_key(self, doc_id: str, stage: str) -> tuple[str, str]:
        return (doc_id, stage)

    def _trigger_refs(self) -> set[str]:
        refs = {entry.entry_id for entry in self.config.relation_triggers}
        refs.update(entry.entry_id for entry in self.config.event_triggers)
        return refs

    def process_documents(
        self,
        doc_ids: list[str],
        deadline: float | None = None,
    ) -> tuple[list[str], list[str]]:
        ordered_ids = sorted(doc_ids)
        completed: list[str] = []
        pending: list[str] = []
        refs = self._trigger_refs()
        workers = max(1, self.config.thread_pool_workers)

        def worker(doc_id: str) -> tuple[str, DocumentResult | None]:
            if deadline is not None and time.perf_counter() >= deadline:
                return doc_id, None
            document = self.documents[doc_id]
            if self.work_hook is not None:
                self.work_hook(doc_id, "process")
            payload = self.cache.process_document(document, self.config, refs)
            result = DocumentResult(
                doc_id=doc_id,
                entities=payload["entities"],
                relations=payload["relations"],
                excluded_pairs=payload["excluded_pairs"],
                events=payload["events"],
                boundaries=payload["boundaries"],
                stage_keys=dict(payload["stage_keys"]),
            )
            return doc_id, result

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(worker, doc_id): doc_id for doc_id in ordered_ids}
            for future in futures:
                doc_id, result = future.result()
                if result is None:
                    pending.append(doc_id)
                    continue
                self._results[doc_id] = result
                self._dirty.discard(doc_id)
                completed.append(doc_id)

        for missing in set(ordered_ids) - set(completed):
            if missing not in pending:
                pending.append(missing)
        pending.sort()
        return completed, pending

    def handle_effect(self, effect: CommandEffect) -> None:
        if effect.deleted_doc is not None:
            self.cache.invalidate_document(effect.deleted_doc, effect.command_type)
            self._results.pop(effect.deleted_doc, None)
            self._dirty.discard(effect.deleted_doc)
            self.documents.pop(effect.deleted_doc, None)
        if effect.replaced_doc is not None:
            self.cache.invalidate_document(effect.replaced_doc.doc_id, effect.command_type)
            self._results.pop(effect.replaced_doc.doc_id, None)
            self._dirty.add(effect.replaced_doc.doc_id)
            self.documents[effect.replaced_doc.doc_id] = effect.replaced_doc
        if effect.failures:
            self.failures.extend(effect.failures)
        for doc_id, stages in effect.invalidated_stages.items():
            if effect.replaced_doc or effect.deleted_doc:
                continue
            for stage in stages:
                self.cache.invalidate_stage_and_downstream(doc_id, stage, effect.command_type)
            self._dirty.add(doc_id)
        if effect.invalidated_stages:
            self.diagnostic["invalidation_log"].append(
                {
                    "command_id": effect.command_id,
                    "command_type": effect.command_type,
                    "affected_docs": list(effect.affected_docs),
                    "stages": {k: list(v) for k, v in effect.invalidated_stages.items()},
                }
            )

    def apply_interaction(
        self,
        command_id: str,
        effect: CommandEffect,
        config: RuleConfig,
        documents: dict[str, Document],
    ) -> InteractionResponse:
        with self._lock:
            started = time.perf_counter()
            self.config = config
            self.documents = dict(documents)
            self.handle_effect(effect)
            budget_seconds = self.config.interaction_budget_ms / 1000.0
            deadline = started + budget_seconds

            target_ids = sorted(effect.affected_docs)
            stale = [doc_id for doc_id in target_ids if doc_id not in self.documents]
            target_ids = [doc_id for doc_id in target_ids if doc_id in self.documents]
            runnable = [doc_id for doc_id in target_ids if doc_id in self._dirty]
            cached_complete = [doc_id for doc_id in target_ids if doc_id not in self._dirty]

            completed: list[str] = []
            pending: list[str] = []
            if time.perf_counter() < deadline:
                completed, pending = self.process_documents(runnable, deadline)
            else:
                pending = list(runnable)

            latest = not pending
            now = time.perf_counter()
            if latest:
                self._run_counter += 1
                self._commands_since_commit += 1
                self._refresh_diagnostic(command_id, (now - started) * 1000.0)
                self._maybe_commit_snapshot()
            summary = self.build_summary()
            progress = {
                "completed": len(self._results),
                "total": len(self.documents),
                "cached_complete": len(cached_complete),
                "newly_completed": len(completed),
                "pending": len(pending),
            }
            response = InteractionResponse(
                command_id=command_id,
                status="latest" if latest else "pending",
                code=None if latest else ErrorCode.E_INTERACT_BUDGET_PENDING,
                summary=summary,
                affected_docs=target_ids + stale,
                progress=progress,
                latest=latest,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
            )
            self.diagnostic["runs"].append(
                {
                    "command_id": command_id,
                    "latest": latest,
                    "elapsed_ms": round(response.elapsed_ms, 6),
                    "pending": list(pending),
                }
            )
            return response

    def drain_pending(self, doc_ids: list[str]) -> list[str]:
        with self._lock:
            runnable = sorted(doc_id for doc_id in doc_ids if doc_id in self.documents)
            completed, pending = self.process_documents(runnable, None)
            if not pending:
                self._run_counter += 1
                self._commands_since_commit += 1
                self._refresh_diagnostic("drain", 0.0)
                self._maybe_commit_snapshot()
            return pending

    def run_full(self) -> dict:
        started = time.perf_counter()
        with self._lock:
            target_ids = sorted(self.documents)
            completed, pending = self.process_documents(target_ids, None)
            self._run_counter += 1
            self._commands_since_commit += 1
            elapsed = (time.perf_counter() - started) * 1000.0
            self._refresh_diagnostic("full", elapsed)
            self._maybe_commit_snapshot(force=True)
            return {
                "completed": completed,
                "pending": pending,
                "elapsed_ms": elapsed,
                "results": self.export_results(),
            }

    def _maybe_commit_snapshot(self, force: bool = False) -> None:
        if self.snapshot_store is None:
            return
        if not force and self._commands_since_commit < self.config.snapshot_commit_interval:
            return
        self.snapshot_store.commit(self.export_snapshot())
        self._commands_since_commit = 0

    def commit_final_snapshot(self) -> dict | None:
        if self.snapshot_store is None:
            return None
        return self.snapshot_store.commit(self.export_snapshot())

    def build_summary(self) -> dict:
        ordered = [self._results[key] for key in sorted(self._results)]
        triple_count = sum(len(result.relations) for result in ordered)
        event_count = sum(len(result.events) for result in ordered)
        return {
            "completed_documents": len(ordered),
            "total_documents": len(self.documents),
            "triple_count": triple_count,
            "event_count": event_count,
            "affected_documents": sorted(self._results),
        }

    def _refresh_diagnostic(self, label: str, elapsed_ms: float) -> None:
        band_distribution: dict[str, int] = {}
        boundary_samples: list[dict] = []
        for doc_id in sorted(self._results):
            result = self._results[doc_id]
            for relation in result.relations:
                band_distribution[f"relation:{relation.band}"] = band_distribution.get(f"relation:{relation.band}", 0) + 1
            for event in result.events:
                for arg in event.args:
                    band_distribution[f"event_arg:{arg.band}"] = band_distribution.get(f"event_arg:{arg.band}", 0) + 1
            boundary_samples.extend(item.as_dict() for item in result.boundaries)
        self.diagnostic["band_distribution"] = dict(sorted(band_distribution.items()))
        self.diagnostic["boundary_samples"] = boundary_samples
        self.diagnostic["reuse"] = {
            "reused": self.cache.reused_count,
            "recomputed": self.cache.recomputed_count,
            "invalidated": self.cache.invalidated_count,
        }
        self.diagnostic["timing_ms"][label] = round(elapsed_ms, 6)

    def export_results(self) -> list[dict]:
        return [self._results[doc_id].as_dict() for doc_id in sorted(self._results)]

    def export_snapshot(self) -> dict:
        return {
            "version_inputs": {
                "config_fp": self._config_fp(),
                "doc_ids": sorted(self.documents),
            },
            "results": self.export_results(),
            "failures": [failure.as_dict() for failure in self._stable_failures()],
            "cache": self.cache.snapshot(),
        }

    def _config_fp(self) -> str:
        from .fingerprint import fingerprint
        from .pipeline import config_fingerprints
        return fingerprint(config_fingerprints(self.config))

    def _stable_failures(self) -> list[Failure]:
        return sorted(self.failures, key=lambda item: (item.doc_id, item.stage, item.code, item.detail))

    def build_report(self) -> dict:
        missing_arg_failures = self._argument_missing_failures()
        all_failures = self._stable_failures() + missing_arg_failures
        return {
            "band_distribution": dict(self.diagnostic["band_distribution"]),
            "boundary_samples": list(self.diagnostic["boundary_samples"]),
            "reuse_and_invalidation": {
                "reused": self.cache.reused_count,
                "recomputed": self.cache.recomputed_count,
                "invalidated": self.cache.invalidated_count,
                "audit_tail": [entry.as_dict() for entry in self.cache.audit[-25:]],
                "invalidation_log": list(self.diagnostic["invalidation_log"]),
            },
            "failures": [failure.as_dict() for failure in all_failures],
            "timing_ms": dict(self.diagnostic["timing_ms"]),
            "runs": list(self.diagnostic["runs"]),
        }

    def _argument_missing_failures(self) -> list[Failure]:
        failures: list[Failure] = []
        for doc_id in sorted(self._results):
            result = self._results[doc_id]
            for event in result.events:
                if event.missing_roles:
                    failures.append(
                        Failure(
                            doc_id,
                            "event",
                            ErrorCode.E_ARGT_MISSING,
                            f"{event.event_id} missing roles {','.join(event.missing_roles)}",
                        )
                    )
        return sorted(failures, key=lambda item: (item.doc_id, item.detail))
