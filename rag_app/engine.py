"""Incremental conversational retrieval engine with budgeted responses."""

from __future__ import annotations

import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any

from .engine_pipeline import PipelineStats, TurnArtifacts, TurnPipeline
from .engine_state import EngineState, apply_command
from .errors import (
    E_ANSWER_BUDGET_PENDING,
    E_CITATION_MISSING,
    E_CONFIG_MISMATCH,
    OK,
)
from .models import (
    Command,
    Document,
    Failure,
    QueryTurn,
    RetrievalConfig,
    turn_sort_key,
)
from .retrieval import BM25Index
from .rules import RuleTable
from .snapshot import SnapshotStore


@dataclass
class TurnRecord:
    turn: QueryTurn
    effective_query: str
    artifacts: TurnArtifacts | None = None
    failures: list[Failure] = field(default_factory=list)
    current: bool = True
    stale_stages: set[str] = field(default_factory=set)
    epoch: int = 0

    @property
    def key(self) -> tuple[str, int]:
        return (self.turn.session_id, self.turn.turn_no)

    @property
    def sort_key(self) -> str:
        return turn_sort_key(self.turn.session_id, self.turn.turn_no)


@dataclass(frozen=True)
class TurnContext:
    """Immutable stage inputs captured when a turn task is submitted."""

    index: BM25Index
    config: RetrievalConfig
    rules: RuleTable
    template_name: str


class Engine:
    def __init__(
        self,
        docs: list[Document],
        turns: list[QueryTurn],
        config: RetrievalConfig,
        rules: RuleTable,
        store: SnapshotStore,
        template_name: str | None = None,
        faults: dict[str, Any] | None = None,
        base_failures: list[Failure] | None = None,
    ):
        self.state = EngineState(
            docs=list(docs),
            turns={(t.session_id, t.turn_no): t for t in turns},
            config=config,
            rules=rules,
            template_name=template_name or rules.template_names[0],
        )
        self.store = store
        self.faults = faults or {}
        self.stats = PipelineStats()
        self.lock = threading.RLock()
        self.records: dict[tuple[str, int], TurnRecord] = {
            (t.session_id, t.turn_no): TurnRecord(
                turn=t, effective_query=self.state.effective_query(t)
            )
            for t in self.state.ordered_turns()
        }
        for record in self.records.values():
            record.stale_stages.update(
                ("retrieval", "extraction", "citation")
            )
        self.global_version = 0
        self.last_cursor = 0
        self.command_log: list[dict[str, Any]] = []
        self.base_failures: list[Failure] = list(base_failures or [])
        self.index = self._build_index()
        self.pipeline = self._new_pipeline()
        self.executor: ThreadPoolExecutor | None = None
        self.pending: dict[Any, tuple[str, int, int]] = {}
        self.store.ensure_dirs()

    # ------------------------------------------------------------------ setup
    def _build_index(self) -> BM25Index:
        return BM25Index(
            self.state.docs, self.state.config, self.state.rules
        )

    def _new_pipeline(self) -> TurnPipeline:
        return TurnPipeline(
            store=self.store,
            stats=self.stats,
            lock=self.lock,
        )

    def _fault_delay(self, stage: str, sort_key: str) -> None:
        injected = self.faults.get("retrieval_delay_ms", 0)
        if not injected:
            return
        targets = self.faults.get("retrieval_delay_targets", [])
        if not targets or sort_key in targets:
            time.sleep(injected / 1000.0)

    # ---------------------------------------------------------------- snapshot
    def current_config_hash(self) -> str:
        return self.state.config.fingerprint()

    def current_rules_fingerprint(self) -> str:
        return self.state.rules.fingerprint()

    def manifest(self) -> dict[str, Any]:
        return {
            "global_version": self.global_version,
            "config_hash": self.current_config_hash(),
            "rules_fingerprint": self.current_rules_fingerprint(),
            "template_name": self.state.template_name,
            "completed_cursor": self.last_cursor,
            "stage_versions": self.store.stage_versions(),
            "turn_order": [record.sort_key for record in self.sorted_records()],
        }

    def commit_snapshot(self) -> None:
        self.store.write_manifest(self.manifest())

    def resume_from_manifest(self) -> dict[str, Any] | None:
        manifest = self.store.check_resume(
            self.current_config_hash(), self.current_rules_fingerprint()
        )
        if manifest is None:
            return None
        self.global_version = int(manifest.get("global_version", 0))
        self.last_cursor = int(manifest.get("completed_cursor", 0))
        return manifest

    @classmethod
    def resume_error_code(cls) -> str:
        return E_CONFIG_MISMATCH

    # --------------------------------------------------------------- utilities
    def sorted_records(self) -> list[TurnRecord]:
        return [self.records[key] for key in sorted(self.records)]

    def completed_count(self) -> int:
        return sum(
            1
            for record in self.sorted_records()
            if record.artifacts is not None and record.current
        )

    def _ensure_executor(self) -> ThreadPoolExecutor:
        if self.executor is None:
            limit = max(int(self.state.config.max_concurrency), 1)
            self.executor = ThreadPoolExecutor(
                max_workers=limit,
                thread_name_prefix="rag-turn",
            )
        return self.executor

    def shutdown(self) -> None:
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.executor = None

    def failure_list(self) -> list[Failure]:
        failures = list(self.base_failures)
        for record in self.sorted_records():
            failures.extend(record.failures)
        failures.sort(key=lambda item: item.sort_key)
        return failures

    def citation_summary(self) -> list[dict[str, Any]]:
        summary: list[dict[str, Any]] = []
        for record in self.sorted_records():
            if record.artifacts is None or record.artifacts.citation is None:
                continue
            citation = record.artifacts.citation
            summary.append(
                {
                    "turn": record.sort_key,
                    "answer_available": citation.answer_available,
                    "citation_count": len(citation.citations),
                    "doc_ids": list(citation.referenced_doc_set),
                }
            )
        return summary

    # ----------------------------------------------------------------- tasks
    def _run_turn_task(
        self,
        record: TurnRecord,
        effective_query: str,
        context: "TurnContext",
        epoch: int,
    ) -> tuple[tuple[str, int], TurnArtifacts, list[Failure]]:
        record.effective_query = effective_query
        prior = record.artifacts
        stale_stages = set(record.stale_stages)
        self._fault_delay("retrieval", record.sort_key)
        forced_missing = self.faults.get("force_no_citation_turn")
        if forced_missing == record.sort_key:
            from .engine_keys import retrieval_key
            from .extraction import CitationResult, ExtractionResult

            r_key = retrieval_key(
                effective_query=effective_query,
                config=context.config,
                rules=context.rules,
                index_fingerprint=context.index.content_fingerprint,
            )
            empty_retrieval = context.index.search(
                r_key, effective_query, context.config.recall_top_k
            )
            empty_extraction = ExtractionResult(
                key=f"forced:{record.sort_key}",
                snippets=(),
                referenced_doc_set=(),
            )
            failure = Failure(
                code=E_CITATION_MISSING,
                scope=record.sort_key,
                message=(
                    "citation disabled by fault injection "
                    "force_no_citation_turn"
                ),
                sort_key=record.sort_key,
            )
            unavailable = CitationResult(
                key=f"forced:{record.sort_key}",
                answer="",
                answer_available=False,
                citations=(),
                backfilled=(),
                referenced_doc_set=(),
                failure=failure,
            )
            forced_artifacts = TurnArtifacts(
                retrieval=empty_retrieval,
                extraction=empty_extraction,
                citation=unavailable,
            )
            record.failures = [failure]
            record.stale_stages.clear()
            return record.key, forced_artifacts, [failure]
        artifacts, failures = self.pipeline.run(
            context.index,
            context.config,
            context.rules,
            context.template_name,
            record.turn,
            effective_query,
            prior,
            stale_stages,
        )
        return record.key, artifacts, failures

    def _rebuild_if_needed(self) -> None:
        new_index = self._build_index()
        if new_index.content_fingerprint != self.index.content_fingerprint:
            self.index = new_index
        self.pipeline = self._new_pipeline()

    def _mark_affected(
        self, affected: list[tuple[str, int]], level: str
    ) -> list[str]:
        affected_keys: list[str] = []
        stages = (
            ("retrieval", "extraction", "citation")
            if level == "retrieval"
            else ("citation",)
        )
        for key in affected:
            record = self.records[key]
            record.current = False
            record.failures = []
            record.stale_stages.update(stages)
            record.epoch += 1
            # Cancel any in-flight task that was scheduled against the
            # previous epoch; running tasks will be discarded on completion.
            for future, future_key in list(self.pending.items()):
                if future_key[0] == key:
                    future.cancel()
            affected_keys.append(record.sort_key)
        return affected_keys

    def _pump(self, budget_ms: int) -> dict[str, Any]:
        """Drive recomputation until the budget expires or work completes."""
        start = time.perf_counter()
        deadline = start + max(budget_ms, 0) / 1000.0
        executor = self._ensure_executor()
        pending_records = [
            record
            for record in self.sorted_records()
            if record.stale_stages or record.artifacts is None
        ]
        for record in pending_records:
            if record.key in [key for key, _epoch in self.pending.values()]:
                continue
            # Snapshot immutable stage inputs at submit time so a later
            # command mutating config/rules cannot change an in-flight task.
            context = TurnContext(
                index=self.index,
                config=self.state.config,
                rules=self.state.rules,
                template_name=self.state.template_name,
            )
            effective_query = self.state.effective_query(record.turn)
            epoch = record.epoch
            future = executor.submit(
                self._run_turn_task,
                record,
                effective_query,
                context,
                epoch,
            )
            self.pending[future] = (record.key, epoch)
        completed_now = 0
        timed_out = False
        while self.pending:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                timed_out = True
                break
            done, _ = wait(
                tuple(self.pending.keys()),
                timeout=remaining,
                return_when=FIRST_COMPLETED,
            )
            if not done:
                timed_out = True
                break
            for future in done:
                turn_key, task_epoch = self.pending.pop(future)
                if future.cancelled():
                    continue
                _turn_key, artifacts, failures = future.result()
                record = self.records[turn_key]
                if task_epoch != record.epoch:
                    # Superseded by a newer command: discard stale result.
                    continue
                record.artifacts = artifacts
                record.failures = failures
                record.current = True
                record.stale_stages.clear()
                completed_now += 1
        self.last_cursor = self.completed_count()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        over_budget = timed_out and bool(self.pending)
        return {
            "elapsed_ms": round(elapsed_ms, 3),
            "completed_now": completed_now,
            "over_budget": over_budget,
            "pending_turns": [
                self.records[key].sort_key
                for key, _epoch in sorted(
                    self.pending.values(),
                    key=lambda item: turn_sort_key(*item[0]),
                )
            ],
        }

    def _latest_summary(
        self,
        *,
        status: str,
        code: str,
        command: Command | None,
        affected_keys: list[str],
        pump: dict[str, Any] | None,
        elapsed_ms: float,
    ) -> dict[str, Any]:
        pending_turns = pump["pending_turns"] if pump else []
        return {
            "status": status,
            "code": code,
            "command_seq": None if command is None else command.seq,
            "completed_turns": self.completed_count(),
            "total_turns": len(self.records),
            "current_citations": self.citation_summary(),
            "affected_turns": affected_keys,
            "pending_turns": pending_turns,
            "progress": {
                "done": self.completed_count(),
                "total": len(self.records),
                "pct": round(
                    100.0 * self.completed_count() / max(len(self.records), 1), 2
                ),
            },
            "elapsed_ms": round(elapsed_ms, 3),
            "budget_ms": self.state.config.first_response_budget_ms,
        }

    def run_full(self) -> dict[str, Any]:
        """One-shot full execution with no budget restriction."""
        start = time.perf_counter()
        pump = self._pump(budget_ms=10**12)
        self.global_version += 1
        self.commit_snapshot()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return self._latest_summary(
            status=OK,
            code=OK,
            command=None,
            affected_keys=[],
            pump=pump,
            elapsed_ms=elapsed_ms,
        )

    def handle_command(self, command: Command) -> dict[str, Any]:
        start = time.perf_counter()
        plan = apply_command(self.state, command)
        affected_keys = self._mark_affected(plan.affected, plan.level)
        self._rebuild_if_needed()
        pump = self._pump(self.state.config.first_response_budget_ms)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if pump["over_budget"]:
            status = "PENDING"
            code = E_ANSWER_BUDGET_PENDING
        else:
            status = OK
            code = OK
            self.global_version += 1
        self.commit_snapshot()
        summary = self._latest_summary(
            status=status,
            code=code,
            command=command,
            affected_keys=affected_keys,
            pump=pump,
            elapsed_ms=elapsed_ms,
        )
        summary["note"] = plan.note
        self.command_log.append(
            {
                "seq": command.seq,
                "op": command.op,
                "args": command.args,
                "status": status,
                "code": code,
                "elapsed_ms": summary["elapsed_ms"],
                "budget_ms": summary["budget_ms"],
                "over_budget": pump["over_budget"],
                "affected_turns": affected_keys,
                "completed_now": pump["completed_now"],
                "pending_turns": pump["pending_turns"],
            }
        )
        return summary

    def drain_pending(self) -> dict[str, Any]:
        """Finish all background work without a budget cap."""
        start = time.perf_counter()
        pump = self._pump(budget_ms=10**12)
        self.global_version += 1
        self.commit_snapshot()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return self._latest_summary(
            status=OK,
            code=OK,
            command=None,
            affected_keys=[],
            pump=pump,
            elapsed_ms=elapsed_ms,
        )

    def turn_results(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for record in self.sorted_records():
            artifacts = record.artifacts
            rows.append(
                {
                    "turn": record.sort_key,
                    "session_id": record.turn.session_id,
                    "turn_no": record.turn.turn_no,
                    "is_followup": record.turn.is_followup,
                    "context_ref": list(record.turn.context_ref),
                    "raw_query": record.turn.query,
                    "effective_query": record.effective_query,
                    "current": record.current,
                    "retrieval": None
                    if artifacts is None or artifacts.retrieval is None
                    else artifacts.retrieval.to_dict(),
                    "extraction": None
                    if artifacts is None or artifacts.extraction is None
                    else artifacts.extraction.to_dict(),
                    "citation": None
                    if artifacts is None or artifacts.citation is None
                    else artifacts.citation.to_dict(),
                }
            )
        return rows
