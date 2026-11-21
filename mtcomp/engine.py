from __future__ import annotations

import copy
import dataclasses
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from .alignment import AlignmentResult, build_alignment
from .cache import SentenceKeys, sentence_keys
from .canonical import sha256_value
from .commands import CommandOutcome, apply_command
from .confidence import (
    ConfidenceResult,
    boundary_detail,
    score_confidence,
)
from .decoder import DecodeResult, NoAlignmentError, viterbi_decode
from .model import (
    CompileConfig,
    Failure,
    SourceSentence,
    TranslationState,
    stable_failures,
)


@dataclass
class EngineOptions:
    drain: bool = True
    max_cycles: int = 10_000
    snapshot_callback: Callable[["Engine"], None] | None = None


@dataclass
class ExecutionOutcome:
    sentence_id: str
    keys: SentenceKeys
    alignment: AlignmentResult | None
    decode: DecodeResult | None
    confidence: ConfidenceResult | None
    boundary: dict[str, Any] | None
    stage_misses: tuple[str, ...]
    error_code: str | None = None
    error_message: str | None = None
    charged_points: int = 0


@dataclass
class SentenceRecord:
    sentence_id: str
    keys: SentenceKeys | None = None
    decode: DecodeResult | None = None
    confidence: ConfidenceResult | None = None
    boundary: dict[str, Any] | None = None
    dirty: bool = True
    active_error_code: str | None = None
    last_action: str = "NEW"
    charged_points_last: int = 0
    decode_points_static: int = 0
    recompute_count: int = 0
    reuse_count: int = 0
    cache_hit_count: int = 0
    cumulative_points: int = 0
    last_run_id: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "sentence_id": self.sentence_id,
            "keys": self.keys.to_dict() if self.keys else None,
            "boundary": self.boundary,
            "dirty": self.dirty,
            "active_error_code": self.active_error_code,
            "last_action": self.last_action,
            "charged_points_last": self.charged_points_last,
            "decode_points_static": self.decode_points_static,
            "recompute_count": self.recompute_count,
            "reuse_count": self.reuse_count,
            "cache_hit_count": self.cache_hit_count,
            "cumulative_points": self.cumulative_points,
            "last_run_id": self.last_run_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SentenceRecord":
        return cls(
            sentence_id=data["sentence_id"],
            keys=SentenceKeys.from_dict(data["keys"]) if data.get("keys") else None,
            boundary=data.get("boundary"),
            dirty=bool(data.get("dirty", True)),
            active_error_code=data.get("active_error_code"),
            last_action=data.get("last_action", "NEW"),
            charged_points_last=int(data.get("charged_points_last", 0)),
            decode_points_static=int(data.get("decode_points_static", 0)),
            recompute_count=int(data.get("recompute_count", 0)),
            reuse_count=int(data.get("reuse_count", 0)),
            cache_hit_count=int(data.get("cache_hit_count", 0)),
            cumulative_points=int(data.get("cumulative_points", 0)),
            last_run_id=int(data.get("last_run_id", 0)),
        )


@dataclass
class RunReport:
    run_id: int
    events: list[dict[str, Any]]
    pending_ids: set[str]
    exhausted: list[dict[str, Any]]
    charged_points: int
    max_workers_used: int
    completed_batch_cursor: int
    action_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "events": self.events,
            "pending_ids": sorted(self.pending_ids),
            "exhausted": self.exhausted,
            "charged_points": self.charged_points,
            "max_workers_used": self.max_workers_used,
            "completed_batch_cursor": self.completed_batch_cursor,
            "action_counts": self.action_counts,
        }


class Engine:
    def __init__(
        self,
        state: TranslationState,
        failures: Iterable[Failure] = (),
        options: EngineOptions | None = None,
    ):
        if state.params is None or state.config is None:
            raise ValueError("engine requires complete params and compile config")
        self.state = state
        self.options = options or EngineOptions()
        self.failures: list[Failure] = list(failures)
        self.records: dict[str, SentenceRecord] = {}
        self.alignment_cache: dict[str, AlignmentResult] = {}
        self.decode_cache: dict[str, DecodeResult] = {}
        self.confidence_cache: dict[str, ConfidenceResult] = {}
        self.pending_ids: set[str] = set()
        self.run_id = 0
        self.total_charged_points = 0
        self.batch_cumulative_points: dict[int, int] = {}
        self.ledger_events: list[dict[str, Any]] = []
        self.run_reports: list[RunReport] = []
        self.batches_since_snapshot = 0
        self._batch_cursor = -1
        self._active_runtime_failures: dict[tuple[str, str], Failure] = {}
        self._sync_records()

    @property
    def params(self):
        assert self.state.params is not None
        return self.state.params

    @property
    def config(self) -> CompileConfig:
        assert self.state.config is not None
        return self.state.config

    def clone_fresh(self) -> "Engine":
        return Engine(copy.deepcopy(self.state), failures=[], options=EngineOptions())

    def _runtime_failure_order(
        self, code: str, sentence: SourceSentence, run_id: int
    ) -> tuple[Any, ...]:
        return (6 if code == "E_NO_ALIGNMENT" else 7, sentence.batch, sentence.sentence_id, run_id, code)

    def _activate_runtime_failure(
        self,
        *,
        code: str,
        sentence: SourceSentence,
        message: str,
        run_id: int,
        details: dict[str, Any] | None = None,
    ) -> Failure:
        key = (sentence.sentence_id, code)
        existing = self._active_runtime_failures.get(key)
        if existing is not None:
            existing.active = True
            existing.run_id = run_id
            existing.message = message
            existing.details = dict(details or {})
            existing.order = self._runtime_failure_order(code, sentence, run_id)
            return existing
        failure = Failure(
            code=code,
            phase="decode_schedule" if code == "E_DECODE_BUDGET_PENDING" else "decode",
            message=message,
            file="",
            line=0,
            sentence_id=sentence.sentence_id,
            run_id=run_id,
            active=True,
            details=dict(details or {}),
            order=self._runtime_failure_order(code, sentence, run_id),
        )
        self._active_runtime_failures[key] = failure
        self.failures.append(failure)
        return failure

    def _resolve_runtime_failures(self, sentence_id: str, run_id: int) -> None:
        for (record_id, _code), failure in list(self._active_runtime_failures.items()):
            if record_id == sentence_id and failure.active:
                failure.active = False
                failure.resolved_run_id = run_id

    def _sync_records(self) -> None:
        current_ids = set(self.state.sentences)
        for sentence_id in list(self.records):
            if sentence_id not in current_ids:
                record = self.records.pop(sentence_id)
                record.dirty = True
                self.pending_ids.discard(sentence_id)
                self._resolve_runtime_failures(sentence_id, self.run_id)
        for sentence_id, sentence in self.state.sentences.items():
            keys = sentence_keys(self.state, sentence)
            record = self.records.get(sentence_id)
            if record is None:
                record = SentenceRecord(
                    sentence_id=sentence_id,
                    keys=keys,
                    dirty=True,
                    decode_points_static=self.decode_points(sentence),
                )
                self.records[sentence_id] = record
                continue
            if record.keys != keys:
                record.keys = keys
                record.dirty = True
                record.decode = None
                record.confidence = None
                record.boundary = None
                record.active_error_code = None
            record.decode_points_static = self.decode_points(sentence)

    def decode_points(self, sentence: SourceSentence) -> int:
        return max(1, len(sentence.tokens) * self.config.ngram_order)

    def apply_command(self, command: dict[str, Any], index: int) -> CommandOutcome:
        outcome = apply_command(self.state, command, index, self.config)
        if outcome.failure is not None:
            outcome.failure.active = False
            outcome.failure.resolved_run_id = self.run_id
            outcome.failure.details = {
                **outcome.failure.details,
                "isolated": True,
                "state_mutated": False,
            }
            self.failures.append(outcome.failure)
        self._sync_records()
        return outcome

    def _all_stages_cached(self, keys: SentenceKeys) -> bool:
        return (
            keys.alignment_key in self.alignment_cache
            and keys.decode_key in self.decode_cache
            and keys.confidence_key in self.confidence_cache
        )

    def _reference_confidence(
        self,
        sentence: SourceSentence,
        base_decode: DecodeResult,
        base_confidence: ConfidenceResult,
    ) -> dict[str, Any]:
        reference_bits = self.params.fixed_point_bits + 1
        reference_tier = self.params.quantization_tier
        reference_params = dataclasses.replace(
            self.params, fixed_point_bits=reference_bits
        )
        reference_state = TranslationState(
            sentences={sentence.sentence_id: sentence},
            lexicon=self.state.lexicon,
            params=reference_params,
            config=self.config,
        )
        reference_keys = sentence_keys(
            reference_state,
            sentence,
            bits=reference_bits,
            tier=reference_tier,
        )
        try:
            alignment = self.alignment_cache.get(reference_keys.alignment_key)
            if alignment is None:
                alignment = build_alignment(
                    reference_state, sentence, reference_keys
                )
            decode = self.decode_cache.get(reference_keys.decode_key)
            if decode is None:
                decode = viterbi_decode(
                    alignment,
                    decode_key=reference_keys.decode_key,
                    bits=reference_bits,
                    scale=1 << reference_bits,
                )
            reference_confidence = self.confidence_cache.get(
                reference_keys.confidence_key
            )
            if reference_confidence is None:
                reference_confidence = score_confidence(
                    decode,
                    confidence_key=reference_keys.confidence_key,
                    threshold=self.params.confidence_threshold,
                    bits=reference_bits,
                    tier=reference_tier,
                )
            return boundary_detail(
                base_confidence,
                reference_confidence,
                base_path=base_decode.path_tuple(),
                reference_path=decode.path_tuple(),
            )
        except NoAlignmentError as exc:
            return {
                "reference_fixed_bits": reference_bits,
                "reference_status": "ERROR",
                "decision_flip": False,
                "flip_cause": "NONE",
                "reference_error": str(exc),
            }

    def _execute_sentence(
        self,
        sentence: SourceSentence,
        keys: SentenceKeys,
        charged_points: int,
    ) -> ExecutionOutcome:
        misses: list[str] = []
        alignment = self.alignment_cache.get(keys.alignment_key)
        if alignment is None:
            alignment = build_alignment(self.state, sentence, keys)
            misses.append("alignment")
        if alignment.missing_tokens:
            return ExecutionOutcome(
                sentence_id=sentence.sentence_id,
                keys=keys,
                alignment=alignment,
                decode=None,
                confidence=None,
                boundary=None,
                stage_misses=tuple(misses),
                error_code="E_NO_ALIGNMENT",
                error_message=(
                    "no lexical candidates for tokens: "
                    + ", ".join(alignment.missing_tokens)
                ),
                charged_points=charged_points,
            )
        decode = self.decode_cache.get(keys.decode_key)
        if decode is None:
            try:
                decode = viterbi_decode(
                    alignment,
                    decode_key=keys.decode_key,
                    bits=self.params.fixed_point_bits,
                    scale=self.params.scale,
                )
            except NoAlignmentError as exc:
                misses.append("decode")
                return ExecutionOutcome(
                    sentence_id=sentence.sentence_id,
                    keys=keys,
                    alignment=alignment,
                    decode=None,
                    confidence=None,
                    boundary=None,
                    stage_misses=tuple(misses),
                    error_code="E_NO_ALIGNMENT",
                    error_message=str(exc),
                    charged_points=charged_points,
                )
            misses.append("decode")
        confidence = self.confidence_cache.get(keys.confidence_key)
        if confidence is None:
            confidence = score_confidence(
                decode,
                confidence_key=keys.confidence_key,
                threshold=self.params.confidence_threshold,
                bits=self.params.fixed_point_bits,
                tier=self.params.quantization_tier,
            )
            misses.append("confidence")
        boundary = None
        if confidence.boundary:
            boundary = self._reference_confidence(sentence, decode, confidence)
        return ExecutionOutcome(
            sentence_id=sentence.sentence_id,
            keys=keys,
            alignment=alignment,
            decode=decode,
            confidence=confidence,
            boundary=boundary,
            stage_misses=tuple(misses),
            charged_points=charged_points,
        )

    def _commit_outcome(
        self, outcome: ExecutionOutcome, run_id: int, sentence: SourceSentence
    ) -> None:
        record = self.records[sentence.sentence_id]
        if outcome.alignment is not None:
            self.alignment_cache.setdefault(
                outcome.keys.alignment_key, outcome.alignment
            )
        if outcome.decode is not None:
            self.decode_cache.setdefault(outcome.keys.decode_key, outcome.decode)
        if outcome.confidence is not None:
            self.confidence_cache.setdefault(
                outcome.keys.confidence_key, outcome.confidence
            )
        record.keys = outcome.keys
        record.charged_points_last = outcome.charged_points
        record.cumulative_points += outcome.charged_points
        record.last_run_id = run_id
        record.dirty = False
        if outcome.error_code:
            record.active_error_code = outcome.error_code
            record.decode = None
            record.confidence = None
            record.boundary = None
            record.last_action = "ERROR"
            pending_failure = self._active_runtime_failures.get(
                (sentence.sentence_id, "E_DECODE_BUDGET_PENDING")
            )
            if pending_failure is not None and pending_failure.active:
                pending_failure.active = False
                pending_failure.resolved_run_id = run_id
            self._activate_runtime_failure(
                code=outcome.error_code,
                sentence=sentence,
                message=outcome.error_message or outcome.error_code,
                run_id=run_id,
                details={"tokens": list(sentence.tokens)},
            )
            self.pending_ids.discard(sentence.sentence_id)
            return
        assert outcome.decode is not None and outcome.confidence is not None
        record.active_error_code = None
        record.decode = outcome.decode
        record.confidence = outcome.confidence
        record.boundary = outcome.boundary
        self._resolve_runtime_failures(sentence.sentence_id, run_id)
        self.pending_ids.discard(sentence.sentence_id)
        if outcome.stage_misses:
            record.recompute_count += 1
            record.last_action = "RECOMPUTED"
        else:
            record.cache_hit_count += 1
            record.last_action = "CACHE_STAGE_HIT"

    def _base_event(
        self,
        *,
        run_id: int,
        sentence: SourceSentence,
        record: SentenceRecord,
        action: str,
        charged_points: int,
    ) -> dict[str, Any]:
        return {
            "run_id": run_id,
            "batch": sentence.batch,
            "sentence_id": sentence.sentence_id,
            "stable_key": [sentence.batch, sentence.sentence_id],
            "action": action,
            "decode_points": record.decode_points_static,
            "charged_points": charged_points,
            "cache": record.keys.to_dict() if record.keys else None,
            "recompute_delta": 1
            if action == "RECOMPUTED"
            else 0,
            "reuse_delta": 1
            if action in {"REUSED", "REUSED_ERROR"}
            else 0,
            "cache_hit_delta": 1 if action == "CACHE_STAGE_HIT" else 0,
        }

    def _compute_completed_batch_cursor(self) -> int:
        if not self.state.sentences:
            return -1
        batches = sorted({sentence.batch for sentence in self.state.sentences.values()})
        cursor = -1
        for batch in batches:
            ids = [
                sentence.sentence_id
                for sentence in self.state.sentences.values()
                if sentence.batch == batch
            ]
            complete = all(
                sentence_id in self.records
                and not self.records[sentence_id].dirty
                and sentence_id not in self.pending_ids
                for sentence_id in ids
            )
            if not complete:
                break
            cursor = batch
        return cursor

    def run_cycle(self, *, force_snapshot: bool = False) -> RunReport:
        self.run_id += 1
        run_id = self.run_id
        self._sync_records()
        sentences = [self.state.sentences[key] for key in sorted(self.state.sentences)]
        work = sorted(
            [
                sentence
                for sentence in sentences
                if self.records[sentence.sentence_id].dirty
                or sentence.sentence_id in self.pending_ids
            ],
            key=lambda sentence: sentence.stable_key,
        )
        budget = self.config.decode_points_per_batch
        remaining_by_batch: dict[int, int] = {}
        exhausted_batches: dict[int, dict[str, Any]] = {}
        tasks: list[tuple[SourceSentence, SentenceKeys, int]] = []
        pending_now: set[str] = set()

        for sentence in work:
            record = self.records[sentence.sentence_id]
            assert record.keys is not None
            remaining = remaining_by_batch.setdefault(sentence.batch, budget)
            cache_complete = self._all_stages_cached(record.keys)
            cost = 0 if cache_complete else record.decode_points_static
            if sentence.batch in exhausted_batches:
                pending_now.add(sentence.sentence_id)
                continue
            if cost > remaining:
                exhausted_batches[sentence.batch] = {
                    "run_id": run_id,
                    "batch": sentence.batch,
                    "first_pending_key": [sentence.batch, sentence.sentence_id],
                    "remaining_points": remaining,
                    "required_points": cost,
                }
                pending_now.add(sentence.sentence_id)
                continue
            remaining_by_batch[sentence.batch] = remaining - cost
            tasks.append((sentence, record.keys, cost))

        outcomes: list[ExecutionOutcome] = []
        max_workers_used = 0
        if tasks:
            max_workers = min(
                max(1, self.config.thread_pool_concurrency), len(tasks)
            )
            max_workers_used = max_workers
            if max_workers == 1:
                for sentence, keys, cost in tasks:
                    outcomes.append(self._execute_sentence(sentence, keys, cost))
            else:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [
                        executor.submit(
                            self._execute_sentence, sentence, keys, cost
                        )
                        for sentence, keys, cost in tasks
                    ]
                    for future in futures:
                        outcomes.append(future.result())
        outcome_by_id = {outcome.sentence_id: outcome for outcome in outcomes}
        for sentence in sorted(
            [item[0] for item in tasks], key=lambda sentence: sentence.stable_key
        ):
            self._commit_outcome(outcome_by_id[sentence.sentence_id], run_id, sentence)

        for sentence in work:
            if sentence.sentence_id in pending_now:
                self.pending_ids.add(sentence.sentence_id)
                record = self.records[sentence.sentence_id]
                record.dirty = True
                record.last_action = "PENDING"
                record.last_run_id = run_id
                record.charged_points_last = 0
                self._activate_runtime_failure(
                    code="E_DECODE_BUDGET_PENDING",
                    sentence=sentence,
                    message="decode point budget exhausted; sentence retained in pending queue",
                    run_id=run_id,
                    details={
                        "batch": sentence.batch,
                        "remaining_points": remaining_by_batch.get(
                            sentence.batch, budget
                        ),
                        "required_points": record.decode_points_static,
                    },
                )
            else:
                self._resolve_pending_failure_if_done(sentence.sentence_id, run_id)

        for sentence in sentences:
            if sentence.sentence_id in pending_now:
                continue
            if sentence.sentence_id in outcome_by_id:
                continue
            record = self.records[sentence.sentence_id]
            if record.dirty:
                continue
            if record.active_error_code:
                record.last_action = "REUSED_ERROR"
                record.reuse_count += 1
            else:
                record.last_action = "REUSED"
                record.reuse_count += 1
            record.last_run_id = run_id
            record.charged_points_last = 0

        events: list[dict[str, Any]] = []
        for sentence in sentences:
            record = self.records[sentence.sentence_id]
            if sentence.sentence_id in outcome_by_id:
                outcome = outcome_by_id[sentence.sentence_id]
                if outcome.error_code:
                    action = "ERROR"
                elif outcome.stage_misses:
                    action = "RECOMPUTED"
                else:
                    action = "CACHE_STAGE_HIT"
                event = self._base_event(
                    run_id=run_id,
                    sentence=sentence,
                    record=record,
                    action=action,
                    charged_points=outcome.charged_points,
                )
            elif sentence.sentence_id in pending_now:
                action = "PENDING"
                event = self._base_event(
                    run_id=run_id,
                    sentence=sentence,
                    record=record,
                    action=action,
                    charged_points=0,
                )
            else:
                action = (
                    "REUSED_ERROR" if record.active_error_code else "REUSED"
                )
                event = self._base_event(
                    run_id=run_id,
                    sentence=sentence,
                    record=record,
                    action=action,
                    charged_points=0,
                )
            events.append(event)

        global_cursor = self.total_charged_points
        local_remaining: dict[int, int] = {}
        local_cumulative: dict[int, int] = {}
        for event in events:
            batch = int(event["batch"])
            remaining = local_remaining.setdefault(batch, budget)
            batch_total = local_cumulative.setdefault(
                batch, self.batch_cumulative_points.get(batch, 0)
            )
            charged = int(event["charged_points"])
            event["cumulative_points_before"] = global_cursor
            event["batch_cumulative_points_before"] = batch_total
            event["remaining_budget_before"] = remaining
            if charged:
                remaining = max(0, remaining - charged)
                global_cursor += charged
                batch_total += charged
                local_remaining[batch] = remaining
                local_cumulative[batch] = batch_total
                self.batch_cumulative_points[batch] = batch_total
            event["cumulative_points_after"] = global_cursor
            event["batch_cumulative_points_after"] = batch_total
            event["remaining_budget_after"] = remaining
        self.total_charged_points = global_cursor

        self.ledger_events.extend(events)
        exhausted = [exhausted_batches[key] for key in sorted(exhausted_batches)]
        action_counts: dict[str, int] = {}
        for event in events:
            action_counts[event["action"]] = action_counts.get(event["action"], 0) + 1
        cursor = self._compute_completed_batch_cursor()
        previously_completed = self._batch_cursor
        self._batch_cursor = cursor
        if cursor > previously_completed:
            new_batches = len(
                {
                    sentence.batch
                    for sentence in sentences
                    if previously_completed < sentence.batch <= cursor
                }
            )
            self.batches_since_snapshot += new_batches
        report = RunReport(
            run_id=run_id,
            events=events,
            pending_ids=set(pending_now),
            exhausted=exhausted,
            charged_points=sum(
                int(event["charged_points"]) for event in events
            ),
            max_workers_used=max_workers_used,
            completed_batch_cursor=cursor,
            action_counts=action_counts,
        )
        self.run_reports.append(report)
        if self.options.snapshot_callback is not None:
            due = (
                self.batches_since_snapshot
                >= self.config.snapshot_commit_interval_batches
            )
            if due or force_snapshot:
                self.options.snapshot_callback(self)
                self.batches_since_snapshot = 0
        return report

    def _resolve_pending_failure_if_done(self, sentence_id: str, run_id: int) -> None:
        record = self.records.get(sentence_id)
        if record is None or record.dirty or record.active_error_code:
            return
        failure = self._active_runtime_failures.get(
            (sentence_id, "E_DECODE_BUDGET_PENDING")
        )
        if failure is not None and failure.active:
            failure.active = False
            failure.resolved_run_id = run_id

    def run_to_idle(self, max_cycles: int | None = None) -> list[RunReport]:
        limit = max_cycles or self.options.max_cycles
        reports: list[RunReport] = []
        idle_cycles = 0
        for _ in range(limit):
            any_dirty = any(record.dirty for record in self.records.values())
            if not any_dirty and not self.pending_ids:
                break
            before = (
                self.total_charged_points,
                len(self.pending_ids),
                sum(1 for record in self.records.values() if record.dirty),
            )
            report = self.run_cycle()
            reports.append(report)
            after = (
                self.total_charged_points,
                len(self.pending_ids),
                sum(1 for record in self.records.values() if record.dirty),
            )
            if after == before:
                idle_cycles += 1
                if idle_cycles >= 2:
                    break
            else:
                idle_cycles = 0
        return reports

    def execute_commands(
        self, commands: list[dict[str, Any]], start_index: int = 0
    ) -> tuple[CommandOutcome, ...]:
        outcomes: list[CommandOutcome] = []
        for index in range(start_index, len(commands)):
            outcome = self.apply_command(commands[index], index)
            outcomes.append(outcome)
            self.run_cycle()
        return tuple(outcomes)

    def record_to_row(self, sentence: SourceSentence) -> dict[str, Any]:
        record = self.records[sentence.sentence_id]
        keys = record.keys or sentence_keys(self.state, sentence)
        base: dict[str, Any] = {
            "id": sentence.sentence_id,
            "batch": sentence.batch,
            "tokens": list(sentence.tokens),
            "translation": "",
            "fragments": [],
            "status": "PENDING" if sentence.sentence_id in self.pending_ids else "ERROR",
            "boundary": False,
            "boundary_detail": None,
            "confidence_permille": None,
            "confidence_fixed": None,
            "confidence_fixed_bits": self.params.fixed_point_bits,
            "confidence_fixed_scale": self.params.scale,
            "references": {
                "consulted_entry_keys": [key for key, _ in keys.referenced_entries],
                "used_entry_keys": list(record.decode.used_entry_keys)
                if record.decode
                else [],
                "referenced_sources": list(keys.referenced_sources),
            },
            "cache": {
                "content_hash": keys.content_hash,
                "alignment_key": keys.alignment_key,
                "decode_key": keys.decode_key,
                "confidence_key": keys.confidence_key,
            },
            "accounting": {
                "decode_points_static": record.decode_points_static,
                "charged_points_last_run": record.charged_points_last,
                "cumulative_points": record.cumulative_points,
                "recompute_count": record.recompute_count,
                "reuse_count": record.reuse_count,
                "cache_hit_count": record.cache_hit_count,
                "last_action": record.last_action,
                "last_run_id": record.last_run_id,
            },
            "error": None,
        }
        if record.active_error_code:
            base["status"] = "ERROR"
            base["error"] = {
                "code": record.active_error_code,
                "message": next(
                    (
                        failure.message
                        for failure in self.failures
                        if failure.sentence_id == sentence.sentence_id
                        and failure.code == record.active_error_code
                        and failure.active
                    ),
                    record.active_error_code,
                ),
            }
            return base
        if sentence.sentence_id in self.pending_ids or record.confidence is None:
            return base
        assert record.decode is not None and record.confidence is not None
        fragments = [
            fragment.to_dict(self.params.fixed_point_bits)
            for fragment in record.decode.fragments
        ]
        base.update(
            {
                "translation": " ".join(record.decode.target_tokens),
                "fragments": fragments,
                "status": record.confidence.status,
                "boundary": record.confidence.boundary,
                "boundary_detail": record.boundary,
                "confidence_permille": record.confidence.confidence_permille,
                "confidence_fixed": record.confidence.confidence_fixed,
                "confidence": {
                    "integer_permille": record.confidence.confidence_permille,
                    "fixed": record.confidence.confidence_fixed,
                    "fixed_bits": record.confidence.fixed_point_bits,
                    "fixed_scale": record.confidence.scale,
                    "threshold_permille": record.confidence.threshold_permille,
                    "threshold_fixed": record.confidence.threshold_fixed,
                    "status": record.confidence.status,
                },
                "decode": {
                    "score_fixed": record.decode.score_fixed,
                    "second_score_fixed": record.decode.second_score_fixed,
                    "margin_fixed": record.decode.margin_fixed,
                    "mean_alignment_confidence_fixed": record.decode.mean_alignment_conf_fixed,
                    "path_signature": record.decode.path_signature,
                    "candidate_signature": record.decode.candidate_signature,
                },
            }
        )
        return base

    def sentence_rows(self) -> list[dict[str, Any]]:
        return [
            self.record_to_row(self.state.sentences[key])
            for key in sorted(
                self.state.sentences,
                key=lambda sentence_id: self.state.sentences[sentence_id].stable_key,
            )
        ]

    def ledger_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for sentence in sorted(
            self.state.sentences.values(), key=lambda item: item.stable_key
        ):
            record = self.records[sentence.sentence_id]
            rows.append(
                {
                    "batch": sentence.batch,
                    "sentence_id": sentence.sentence_id,
                    "stable_key": [sentence.batch, sentence.sentence_id],
                    "last_action": record.last_action,
                    "last_run_id": record.last_run_id,
                    "decode_points_static": record.decode_points_static,
                    "charged_points_last_run": record.charged_points_last,
                    "recompute_count": record.recompute_count,
                    "reuse_count": record.reuse_count,
                    "cache_hit_count": record.cache_hit_count,
                    "cumulative_points": record.cumulative_points,
                    "cache_keys": record.keys.to_dict() if record.keys else None,
                    "active_error_code": record.active_error_code,
                }
            )
        return rows

    def canonical_ledger_rows(self) -> list[dict[str, Any]]:
        """Result-derived ledger projection, independent of run history."""
        rows: list[dict[str, Any]] = []
        for sentence in sorted(
            self.state.sentences.values(), key=lambda item: item.stable_key
        ):
            record = self.records[sentence.sentence_id]
            rows.append(
                {
                    "batch": sentence.batch,
                    "sentence_id": sentence.sentence_id,
                    "stable_key": [sentence.batch, sentence.sentence_id],
                    "decode_points_static": record.decode_points_static,
                    "cache_keys": record.keys.to_dict() if record.keys else None,
                    "active_error_code": record.active_error_code,
                }
            )
        return rows

    def active_failures(self) -> list[Failure]:
        return stable_failures(failure for failure in self.failures if failure.active)

    def all_failures(self) -> list[Failure]:
        return stable_failures(self.failures)

    def summary(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for row in self.sentence_rows():
            counts[row["status"]] = counts.get(row["status"], 0) + 1
        return {
            "sentence_count": len(self.state.sentences),
            "status_counts": dict(sorted(counts.items())),
            "failure_count": len(self.failures),
            "active_failure_count": len(self.active_failures()),
            "total_charged_points": self.total_charged_points,
            "pending_count": len(self.pending_ids),
            "completed_batch_cursor": self._batch_cursor,
            "run_count": self.run_id,
            "cache_entries": {
                "alignment": len(self.alignment_cache),
                "decode": len(self.decode_cache),
                "confidence": len(self.confidence_cache),
            },
        }

    def to_snapshot_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.to_dict(),
            "run_id": self.run_id,
            "total_charged_points": self.total_charged_points,
            "batch_cumulative_points": {
                str(key): value for key, value in sorted(self.batch_cumulative_points.items())
            },
            "pending_ids": sorted(self.pending_ids),
            "completed_batch_cursor": self._batch_cursor,
            "records": [
                self.records[key].to_dict() for key in sorted(self.records)
            ],
            "failures": [failure.to_dict() for failure in self.all_failures()],
            "ledger_events": self.ledger_events,
            "run_reports": [report.to_dict() for report in self.run_reports],
            "caches": {
                "alignment": {
                    key: value.to_dict()
                    for key, value in sorted(self.alignment_cache.items())
                },
                "decode": {
                    key: value.to_dict()
                    for key, value in sorted(self.decode_cache.items())
                },
                "confidence": {
                    key: value.to_dict()
                    for key, value in sorted(self.confidence_cache.items())
                },
            },
        }

    @classmethod
    def from_snapshot_dict(
        cls,
        data: dict[str, Any],
        options: EngineOptions | None = None,
    ) -> "Engine":
        state = TranslationState.from_dict(data["state"])
        engine = cls(state=state, failures=[], options=options)
        engine.run_id = int(data.get("run_id", 0))
        engine.total_charged_points = int(data.get("total_charged_points", 0))
        engine.batch_cumulative_points = {
            int(key): int(value)
            for key, value in data.get("batch_cumulative_points", {}).items()
        }
        engine.pending_ids = set(data.get("pending_ids", []))
        engine._batch_cursor = int(
            data.get("completed_batch_cursor", -1)
        )
        engine.records = {
            item["sentence_id"]: SentenceRecord.from_dict(item)
            for item in data.get("records", [])
        }
        engine.failures = [_failure_from_dict(item) for item in data.get("failures", [])]
        engine._active_runtime_failures = {
            (failure.sentence_id or "", failure.code): failure
            for failure in engine.failures
            if failure.active
            and failure.code
            in {"E_NO_ALIGNMENT", "E_DECODE_BUDGET_PENDING"}
        }
        engine.ledger_events = list(data.get("ledger_events", []))
        engine.run_reports = [
            RunReport(
                run_id=int(item["run_id"]),
                events=list(item.get("events", [])),
                pending_ids=set(item.get("pending_ids", [])),
                exhausted=list(item.get("exhausted", [])),
                charged_points=int(item.get("charged_points", 0)),
                max_workers_used=int(item.get("max_workers_used", 0)),
                completed_batch_cursor=int(
                    item.get("completed_batch_cursor", -1)
                ),
                action_counts=dict(item.get("action_counts", {})),
            )
            for item in data.get("run_reports", [])
        ]
        caches = data.get("caches", {})
        engine.alignment_cache = {
            key: AlignmentResult.from_dict(value)
            for key, value in caches.get("alignment", {}).items()
        }
        engine.decode_cache = {
            key: DecodeResult.from_dict(value)
            for key, value in caches.get("decode", {}).items()
        }
        engine.confidence_cache = {
            key: ConfidenceResult.from_dict(value)
            for key, value in caches.get("confidence", {}).items()
        }
        engine._sync_records()
        for record in engine.records.values():
            if record.keys is not None:
                # Sentence outputs are content-addressed and intentionally not
                # duplicated in the record audit payload.
                record.decode = engine.decode_cache.get(record.keys.decode_key)
                record.confidence = engine.confidence_cache.get(
                    record.keys.confidence_key
                )
                if (
                    not record.active_error_code
                    and record.decode is not None
                    and record.confidence is not None
                ):
                    record.dirty = False
        return engine


def _failure_from_dict(data: dict[str, Any]) -> Failure:
    return Failure(
        code=data["code"],
        phase=data.get("phase", ""),
        message=data.get("message", ""),
        file=data.get("file", ""),
        line=int(data.get("line", 0)),
        sentence_id=data.get("sentence_id"),
        entry_key=data.get("entry_key"),
        injection_id=data.get("injection_id"),
        command_id=data.get("command_id"),
        run_id=data.get("run_id"),
        active=bool(data.get("active", True)),
        resolved_run_id=data.get("resolved_run_id"),
        details=dict(data.get("details", {})),
        order=tuple(data.get("order", ())),
    )


def canonical_sentence_payload(row: dict[str, Any]) -> dict[str, Any]:
    """Projection used for incremental/full field-by-field parity checks."""
    return {
        "id": row["id"],
        "batch": row["batch"],
        "tokens": row["tokens"],
        "translation": row["translation"],
        "fragments": row["fragments"],
        "status": row["status"],
        "boundary": row["boundary"],
        "boundary_detail": row["boundary_detail"],
        "confidence_permille": row["confidence_permille"],
        "confidence_fixed": row["confidence_fixed"],
        "confidence_fixed_bits": row["confidence_fixed_bits"],
        "confidence": row.get("confidence"),
        "decode": row.get("decode"),
        "references": row["references"],
        "cache": {
            "content_hash": row["cache"]["content_hash"],
            "alignment_key": row["cache"]["alignment_key"],
            "decode_key": row["cache"]["decode_key"],
            "confidence_key": row["cache"]["confidence_key"],
        },
        "error": row["error"],
    }


def payload_hash(row: dict[str, Any]) -> str:
    return sha256_value(canonical_sentence_payload(row))
