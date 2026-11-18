from __future__ import annotations

import hashlib
import re
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .config import (
    config_hash,
    extraction_config,
    merge_config_view,
    normalize_config,
    rewrite_config_view,
)
from .lexicon import Lexicon
from .loader import LoadedInput, _failure, _sort_failures
from .snapshot import SnapshotManager
from .util import fingerprint, round_weight


SCHEMA_VERSION = 1
STAGES = ("extraction", "merge", "rewrite")


@dataclass
class SessionState:
    session_id: str
    events: list[dict[str, Any]]
    summary: list[dict[str, Any]] = field(default_factory=list)
    rounds: list[dict[str, Any]] = field(default_factory=list)
    blocked_turns: set[int] = field(default_factory=set)
    caches: dict[str, dict[str, dict[str, Any]]] = field(
        default_factory=lambda: {stage: {} for stage in STAGES}
    )
    recomputed_stages: dict[str, list[int]] = field(
        default_factory=lambda: {stage: [] for stage in STAGES}
    )
    reused_stages: dict[str, list[int]] = field(
        default_factory=lambda: {stage: [] for stage in STAGES}
    )
    recomputed_rounds: list[int] = field(default_factory=list)
    reused_rounds: list[int] = field(default_factory=list)
    reuse_flags: dict[str, bool] = field(
        default_factory=lambda: {stage: False for stage in STAGES}
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "events": self.events,
            "summary": self.summary,
            "rounds": self.rounds,
            "blocked_turns": sorted(self.blocked_turns),
            "caches": self.caches,
            "recomputed_stages": self.recomputed_stages,
            "reused_stages": self.reused_stages,
            "recomputed_rounds": self.recomputed_rounds,
            "reused_rounds": self.reused_rounds,
            "reuse_flags": self.reuse_flags,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SessionState":
        return cls(
            session_id=value["session_id"],
            events=list(value.get("events", [])),
            summary=list(value.get("summary", [])),
            rounds=list(value.get("rounds", [])),
            blocked_turns=set(int(item) for item in value.get("blocked_turns", [])),
            caches={
                stage: dict(value.get("caches", {}).get(stage, {}))
                for stage in STAGES
            },
            recomputed_stages={
                stage: list(value.get("recomputed_stages", {}).get(stage, []))
                for stage in STAGES
            },
            reused_stages={
                stage: list(value.get("reused_stages", {}).get(stage, []))
                for stage in STAGES
            },
            recomputed_rounds=list(value.get("recomputed_rounds", [])),
            reused_rounds=list(value.get("reused_rounds", [])),
            reuse_flags={
                stage: bool(value.get("reuse_flags", {}).get(stage, False))
                for stage in STAGES
            },
        )

    def export_seed(self) -> dict[str, Any]:
        return {
            "caches": deepcopy(self.caches),
            "blocked_turns": sorted(self.blocked_turns),
        }


class Engine:
    def __init__(
        self,
        *,
        config: dict[str, Any],
        lexicon: Lexicon,
        loaded: LoadedInput,
        mode: str = "full",
        snapshot_manager: SnapshotManager | None = None,
        seed_sessions: dict[str, dict[str, Any]] | None = None,
        initial_failures: list[dict[str, Any]] | None = None,
        fault_hook: Callable[[str, str, int], None] | None = None,
        stop_requested: Callable[[], bool] | None = None,
        state_extra: Callable[[], dict[str, Any]] | None = None,
        resume_state: dict[str, Any] | None = None,
        resumed_from: int | None = None,
        artifact_versions: dict[str, int] | None = None,
    ) -> None:
        self.lexicon = lexicon
        self.snapshot_manager = snapshot_manager
        self.fault_hook = fault_hook
        self.stop_requested = stop_requested
        self.state_extra = state_extra
        self.resumed_from = resumed_from
        self.stage_work = {stage: 0 for stage in STAGES}
        self.observed_wall_clock_ms = {stage: 0.0 for stage in STAGES}
        self.rounds_completed = 0
        self.artifact_versions = {stage: 0 for stage in STAGES}
        if artifact_versions is not None:
            self.artifact_versions = {
                stage: int(artifact_versions.get(stage, 0)) for stage in STAGES
            }
        self.snapshot_version = (
            snapshot_manager.current_version if snapshot_manager is not None else 0
        )

        if resume_state is not None:
            self.mode = str(resume_state.get("mode", mode))
            self.config = normalize_config(resume_state["config"])
            self.failures = list(resume_state.get("failures", []))
            self.rank = int(resume_state.get("cursor", 0))
            self.phase = str(resume_state.get("phase", "round_start"))
            self.status = str(resume_state.get("status", "running"))
            self.pending_extraction = dict(resume_state.get("pending_extraction", {}))
            self.pending_merge = dict(resume_state.get("pending_merge", {}))
            self.stage_work = {
                stage: int(resume_state.get("stage_work", {}).get(stage, 0))
                for stage in STAGES
            }
            self.observed_wall_clock_ms = {
                stage: float(
                    resume_state.get("observed_wall_clock_ms", {}).get(stage, 0.0)
                )
                for stage in STAGES
            }
            self.rounds_completed = int(resume_state.get("rounds_completed", 0))
            self.artifact_versions = {
                stage: int(
                    resume_state.get("artifact_versions", {}).get(stage, 0)
                )
                for stage in STAGES
            }
            self.sessions = {
                sid: SessionState.from_dict(value)
                for sid, value in resume_state.get("sessions", {}).items()
            }
        else:
            self.mode = mode
            self.config = normalize_config(config)
            self.rank = 0
            self.phase = "round_start"
            self.status = "running"
            self.pending_extraction = {}
            self.pending_merge = {}
            self.failures = _sort_failures(
                [*loaded.failures, *(initial_failures or [])]
            )
            seed_sessions = seed_sessions or {}
            by_session: dict[str, list[dict[str, Any]]] = {}
            for event in loaded.events:
                by_session.setdefault(event["session_id"], []).append(event)
            self.sessions = {}
            for session_id in sorted(by_session):
                seed = seed_sessions.get(session_id, {})
                events = sorted(by_session[session_id], key=lambda item: item["turn"])
                caches = deepcopy(seed.get("caches", {}))
                self.sessions[session_id] = SessionState(
                    session_id=session_id,
                    events=events,
                    blocked_turns=set(seed.get("blocked_turns", [])),
                    caches={
                        stage: dict(caches.get(stage, {})) for stage in STAGES
                    },
                )
        self.max_rank = max(
            (len(session.events) for session in self.sessions.values()), default=0
        )

    def run(
        self,
        *,
        stop_after_rounds: int | None = None,
        crash_after_snapshots: int | None = None,
    ) -> dict[str, Any]:
        guard = 0
        while True:
            guard += 1
            if guard > 10_000_000:
                raise RuntimeError("engine made no progress")
            if self.phase == "round_start":
                if self.rank >= self.max_rank:
                    if self.status != "complete":
                        self.status = "complete"
                        self.phase = "complete"
                        self._commit("complete", status="complete", stage=None)
                    break
                self._run_extraction_stage()
                self.phase = "extraction_done"
                self._commit("extraction_done", status="running", stage="extraction")
            elif self.phase == "extraction_done":
                self._run_merge_stage()
                self.phase = "merge_done"
                self._commit("merge_done", status="running", stage="merge")
            elif self.phase == "merge_done":
                self._run_rewrite_stage()
                self.rank += 1
                self.rounds_completed += 1
                self.phase = "round_start"
                self._commit("round_done", status="running", stage="rewrite")
                if (
                    stop_after_rounds is not None
                    and self.rounds_completed >= stop_after_rounds
                    and self.rank < self.max_rank
                ):
                    self.status = "interrupted"
                    self._commit("interrupted", status="interrupted", stage=None)
                    break
                if (
                    self.stop_requested is not None
                    and self.stop_requested()
                    and self.rank < self.max_rank
                ):
                    self.status = "interrupted"
                    self._commit("interrupted", status="interrupted", stage=None)
                    break
            elif self.phase == "complete":
                break
            else:
                raise RuntimeError(f"unknown engine phase: {self.phase}")

            if (
                crash_after_snapshots is not None
                and self.snapshot_version >= crash_after_snapshots
                and self.rank < self.max_rank
            ):
                return {"__crash_requested__": True}

        return self.result()

    def export_seed(self) -> dict[str, dict[str, Any]]:
        return {
            session_id: session.export_seed()
            for session_id, session in sorted(self.sessions.items())
        }

    def to_state(self) -> dict[str, Any]:
        state = {
            "schema_version": SCHEMA_VERSION,
            "mode": self.mode,
            "config": self.config,
            "failures": _sort_failures(self.failures),
            "cursor": self.rank,
            "phase": self.phase,
            "status": self.status,
            "rounds_completed": self.rounds_completed,
            "stage_work": self.stage_work,
            "observed_wall_clock_ms": self.observed_wall_clock_ms,
            "artifact_versions": self.artifact_versions,
            "pending_extraction": self.pending_extraction,
            "pending_merge": self.pending_merge,
            "sessions": {
                session_id: session.to_dict()
                for session_id, session in sorted(self.sessions.items())
            },
        }
        return state

    def result(self) -> dict[str, Any]:
        sessions: dict[str, Any] = {}
        summary_lengths: list[dict[str, Any]] = []
        merge_count = 0
        truncation_count = 0
        for session_id, session in sorted(self.sessions.items()):
            rounds = session.rounds
            final_summary = rounds[-1]["summary_entries"] if rounds else []
            final_source_map = rounds[-1]["source_map"] if rounds else []
            merge_count += sum(
                int(round_item.get("merge", {}).get("merge_count", 0))
                for round_item in rounds
            )
            truncation_count += sum(
                int(round_item.get("truncation_count", 0)) for round_item in rounds
            )
            for round_item in rounds:
                summary_lengths.append(
                    {
                        "entry_count": len(round_item["summary_entries"]),
                        "session_id": session_id,
                        "summary_length": round_item["summary_length"],
                        "turn": round_item["turn"],
                    }
                )
            sessions[session_id] = {
                "session_id": session_id,
                "rounds": rounds,
                "final_summary": final_summary,
                "final_source_map": final_source_map,
                "incremental": {
                    "recomputed_rounds": sorted(session.recomputed_rounds),
                    "reused_rounds": sorted(session.reused_rounds),
                    "recomputed_stages": {
                        stage: sorted(session.recomputed_stages[stage])
                        for stage in STAGES
                    },
                    "reused_stages": {
                        stage: sorted(session.reused_stages[stage])
                        for stage in STAGES
                    },
                },
            }
        return {
            "schema_version": SCHEMA_VERSION,
            "status": self.status,
            "mode": self.mode,
            "config_hash": config_hash(self.config),
            "vocab_fingerprint": self.lexicon.fingerprint,
            "resumed_from": self.resumed_from,
            "recovery": {
                "cursor": self.rank,
                "phase": self.phase,
                "snapshot_version": self.snapshot_version,
                "status": self.status,
            },
            "counts": {
                "merge": merge_count,
                "truncation": truncation_count,
            },
            "summary_lengths": summary_lengths,
            "stage_timing": {
                stage: {
                    "work_units": self.stage_work[stage],
                    "logical_ms": round_weight(self.stage_work[stage] / 1000.0),
                }
                for stage in STAGES
            },
            "runtime_metrics": {
                "observed_wall_clock_ms": {
                    stage: round(self.observed_wall_clock_ms[stage], 6)
                    for stage in STAGES
                },
                "concurrency": self.config["concurrency"],
            },
            "failures": _sort_failures(self.failures),
            "sessions": sessions,
        }

    def _run_extraction_stage(self) -> None:
        active = self._active_sessions()
        started = time.perf_counter()
        outputs: dict[str, dict[str, Any]] = {}
        failures: list[dict[str, Any]] = []
        work_units = 0
        elapsed_total = 0.0
        with ThreadPoolExecutor(max_workers=self.config["concurrency"]) as pool:
            futures = [
                (
                    session_id,
                    pool.submit(self._extract_session, session_id),
                )
                for session_id in active
            ]
            for session_id, future in futures:
                output, reused, stage_failures, work, elapsed = future.result()
                outputs[session_id] = output
                failures.extend(stage_failures)
                work_units += work
                elapsed_total += elapsed
                self.sessions[session_id].reuse_flags["extraction"] = reused
        self.pending_extraction = outputs
        self.failures.extend(failures)
        self.failures = _sort_failures(self.failures)
        self.stage_work["extraction"] += work_units
        self.observed_wall_clock_ms["extraction"] += elapsed_total
        if self.observed_wall_clock_ms["extraction"] <= 0:
            self.observed_wall_clock_ms["extraction"] = (
                time.perf_counter() - started
            ) * 1000.0

    def _run_merge_stage(self) -> None:
        active = self._active_sessions()
        started = time.perf_counter()
        outputs: dict[str, dict[str, Any]] = {}
        failures: list[dict[str, Any]] = []
        work_units = 0
        elapsed_total = 0.0
        with ThreadPoolExecutor(max_workers=self.config["concurrency"]) as pool:
            futures = [
                (
                    session_id,
                    pool.submit(
                        self._merge_session,
                        session_id,
                        self.pending_extraction.get(session_id, {}),
                    ),
                )
                for session_id in active
            ]
            for session_id, future in futures:
                output, reused, stage_failures, work, elapsed = future.result()
                outputs[session_id] = output
                failures.extend(stage_failures)
                work_units += work
                elapsed_total += elapsed
                self.sessions[session_id].reuse_flags["merge"] = reused
        self.pending_merge = outputs
        self.pending_extraction = {}
        self.failures.extend(failures)
        self.failures = _sort_failures(self.failures)
        self.stage_work["merge"] += work_units
        self.observed_wall_clock_ms["merge"] += elapsed_total
        if self.observed_wall_clock_ms["merge"] <= 0:
            self.observed_wall_clock_ms["merge"] = (
                time.perf_counter() - started
            ) * 1000.0

    def _run_rewrite_stage(self) -> None:
        active = self._active_sessions()
        started = time.perf_counter()
        outputs: dict[str, dict[str, Any]] = {}
        failures: list[dict[str, Any]] = []
        work_units = 0
        elapsed_total = 0.0
        with ThreadPoolExecutor(max_workers=self.config["concurrency"]) as pool:
            futures = [
                (
                    session_id,
                    pool.submit(
                        self._rewrite_session,
                        session_id,
                        self.pending_merge.get(session_id, {}),
                    ),
                )
                for session_id in active
            ]
            for session_id, future in futures:
                output, reused, stage_failures, work, elapsed = future.result()
                outputs[session_id] = output
                failures.extend(stage_failures)
                work_units += work
                elapsed_total += elapsed
                self.sessions[session_id].reuse_flags["rewrite"] = reused
        self.failures.extend(failures)
        self.failures = _sort_failures(self.failures)
        self.stage_work["rewrite"] += work_units
        self.observed_wall_clock_ms["rewrite"] += elapsed_total
        if self.observed_wall_clock_ms["rewrite"] <= 0:
            self.observed_wall_clock_ms["rewrite"] = (
                time.perf_counter() - started
            ) * 1000.0

        for session_id in active:
            session = self.sessions[session_id]
            extraction = self.pending_extraction.get(session_id) or self._round_cache_output(
                session, "extraction", session.events[self.rank]["turn"]
            )
            merge_output = self.pending_merge.get(session_id, {})
            rewrite_output = outputs[session_id]
            turn = session.events[self.rank]["turn"]
            stage_flags = dict(session.reuse_flags)
            for stage in STAGES:
                if stage_flags[stage]:
                    session.reused_stages[stage].append(turn)
                else:
                    session.recomputed_stages[stage].append(turn)
            if all(stage_flags.values()):
                session.reused_rounds.append(turn)
            else:
                session.recomputed_rounds.append(turn)
            round_item = {
                "session_id": session_id,
                "turn": turn,
                "extraction": extraction,
                "merge": merge_output,
                "rewrite": rewrite_output,
                "summary_entries": rewrite_output["entries"],
                "source_map": rewrite_output["source_map"],
                "summary_length": rewrite_output["summary_length"],
                "truncation_count": rewrite_output["truncation_count"],
                "reuse": {
                    "stages": stage_flags,
                    "fully_reused": all(stage_flags.values()),
                },
            }
            session.rounds.append(round_item)
        self.pending_merge = {}
        self.pending_extraction = {}

    def _extract_session(
        self, session_id: str
    ) -> tuple[dict[str, Any], bool, list[dict[str, Any]], int, float]:
        started = time.perf_counter()
        session = self.sessions[session_id]
        event = session.events[self.rank]
        turn = event["turn"]
        config = self.config
        window = config["compression_window"]
        window_start = max(1, turn - window + 1)
        window_end = turn
        previous_turn = (
            session.events[self.rank - 1]["turn"] if self.rank > 0 else None
        )
        previous_window = (
            set(
                range(
                    max(1, previous_turn - window + 1),
                    previous_turn + 1,
                )
            )
            if previous_turn is not None
            else set()
        )
        current_window = set(range(window_start, window_end + 1))
        evicted_turns = sorted(previous_window - current_window)
        considered = [
            item
            for item in session.events
            if window_start <= item["turn"] <= window_end
            and item["turn"] not in session.blocked_turns
        ]
        key_material = {
            "stage": "extraction",
            "session_id": session_id,
            "turn": turn,
            "previous_turn": previous_turn,
            "window": [window_start, window_end],
            "considered": [
                [item["turn"], item["content_fingerprint"]] for item in considered
            ],
            "blocked_turns": sorted(session.blocked_turns),
            "config": extraction_config(config),
            "vocab_fingerprint": self.lexicon.fingerprint,
        }
        cache_key = fingerprint(key_material)
        cached = session.caches["extraction"].get(str(turn))
        if cached is not None and cached.get("key") == cache_key:
            output = deepcopy(cached["output"])
            return output, True, [], 0, time.perf_counter() - started

        failures: list[dict[str, Any]] = []
        work_units = len(considered) + len(current_window)
        candidates: list[dict[str, Any]] = []
        utterance_weights: list[dict[str, Any]] = []
        try:
            if self.fault_hook is not None:
                self.fault_hook("extraction", session_id, turn)
            for item in considered:
                try:
                    utterance_weights.append(
                        {
                            "turn": int(item["turn"]),
                            "weight": self._utterance_weight(item, turn),
                        }
                    )
                    candidates.extend(self._candidates_for_event(item, turn))
                except Exception as exc:
                    session.blocked_turns.add(int(item["turn"]))
                    failures.append(
                        _failure(
                            stage="extraction",
                            code="E_STAGE_EXCEPTION",
                            message=str(exc) or "candidate extraction failed",
                            session_id=session_id,
                            turn=int(item["turn"]),
                            exception_type=type(exc).__name__,
                        )
                    )
                work_units += 1 + len(item["text"])
        except Exception as exc:
            session.blocked_turns.add(turn)
            failures.append(
                _failure(
                    stage="extraction",
                    code="E_STAGE_EXCEPTION",
                    message=str(exc) or "extraction stage failed",
                    session_id=session_id,
                    turn=turn,
                    exception_type=type(exc).__name__,
                )
            )

        candidates = self._select_candidates(candidates)
        output = {
            "candidates": candidates,
            "considered_events": [
                {
                    "content_fingerprint": item["content_fingerprint"],
                    "text": item["text"],
                    "turn": item["turn"],
                    "utterance_weight": next(
                        (
                            weight["weight"]
                            for weight in utterance_weights
                            if weight["turn"] == item["turn"]
                        ),
                        None,
                    ),
                }
                for item in considered
            ],
            "utterance_weights": utterance_weights,
            "evicted_turns": evicted_turns,
            "window_start": window_start,
            "window_end": window_end,
            "window_turns": list(range(window_start, window_end + 1)),
            "cache_key": cache_key,
        }
        output["fingerprint"] = fingerprint(
            {
                "candidates": candidates,
                "considered_events": output["considered_events"],
                "evicted_turns": evicted_turns,
                "utterance_weights": utterance_weights,
                "window": [window_start, window_end],
            }
        )
        session.caches["extraction"][str(turn)] = {
            "key": cache_key,
            "output": output,
        }
        return output, False, failures, work_units, time.perf_counter() - started

    def _merge_session(
        self, session_id: str, extraction: dict[str, Any]
    ) -> tuple[dict[str, Any], bool, list[dict[str, Any]], int, float]:
        started = time.perf_counter()
        session = self.sessions[session_id]
        turn = session.events[self.rank]["turn"]
        prior_fingerprint = fingerprint(session.summary)
        key_material = {
            "stage": "merge",
            "session_id": session_id,
            "turn": turn,
            "extraction_fingerprint": extraction.get("fingerprint"),
            "prior_summary_fingerprint": prior_fingerprint,
            "config": merge_config_view(self.config),
            "vocab_fingerprint": self.lexicon.fingerprint,
        }
        cache_key = fingerprint(key_material)
        cached = session.caches["merge"].get(str(turn))
        if cached is not None and cached.get("key") == cache_key:
            output = deepcopy(cached["output"])
            session.summary = deepcopy(output["entries"])
            return output, True, [], 0, time.perf_counter() - started

        failures: list[dict[str, Any]] = []
        candidates = extraction.get("candidates", [])
        work_units = max(1, len(session.summary) * len(candidates) + len(candidates))
        try:
            if self.fault_hook is not None:
                self.fault_hook("merge", session_id, turn)
            entries, merge_records = self._merge_candidates(
                session.summary, candidates, turn
            )
        except Exception as exc:
            entries = deepcopy(session.summary)
            merge_records = []
            for candidate in candidates:
                session.blocked_turns.add(int(candidate["source_turn"]))
            failures.append(
                _failure(
                    stage="merge",
                    code="E_STAGE_EXCEPTION",
                    message=str(exc) or "merge stage failed",
                    session_id=session_id,
                    turn=turn,
                    exception_type=type(exc).__name__,
                )
            )
        session.summary = deepcopy(entries)
        entries_fingerprint = fingerprint(entries)
        output = {
            "entries": entries,
            "merge_records": merge_records,
            "merge_count": len(merge_records),
            "entries_fingerprint": entries_fingerprint,
            "cache_key": cache_key,
        }
        session.caches["merge"][str(turn)] = {"key": cache_key, "output": output}
        return output, False, failures, work_units, time.perf_counter() - started

    def _rewrite_session(
        self, session_id: str, merge_output: dict[str, Any]
    ) -> tuple[dict[str, Any], bool, list[dict[str, Any]], int, float]:
        started = time.perf_counter()
        session = self.sessions[session_id]
        turn = session.events[self.rank]["turn"]
        key_material = {
            "stage": "rewrite",
            "session_id": session_id,
            "turn": turn,
            "entries_fingerprint": merge_output.get("entries_fingerprint"),
            "config": rewrite_config_view(self.config),
            "vocab_fingerprint": self.lexicon.fingerprint,
        }
        cache_key = fingerprint(key_material)
        cached = session.caches["rewrite"].get(str(turn))
        if cached is not None and cached.get("key") == cache_key:
            return deepcopy(cached["output"]), True, [], 0, time.perf_counter() - started

        failures: list[dict[str, Any]] = []
        entries = merge_output.get("entries", [])
        work_units = max(1, len(entries))
        rewritten: list[dict[str, Any]] = []
        try:
            if self.fault_hook is not None:
                self.fault_hook("rewrite", session_id, turn)
            for entry in entries:
                rewritten.append(self._rewrite_entry(entry))
        except Exception as exc:
            rewritten = []
            for entry in entries:
                text = str(entry.get("text", ""))
                truncated_text, truncation = self._truncate(
                    text, self.config["max_summary_length"]
                )
                rewritten.append(
                    {
                        "entry_seq": entry["entry_seq"],
                        "summary_text": truncated_text,
                        "template_id": "fallback",
                        "truncation": truncation,
                        "weight": entry["weight"],
                        "sources": deepcopy(entry["sources"]),
                    }
                )
            failures.append(
                _failure(
                    stage="rewrite",
                    code="E_STAGE_EXCEPTION",
                    message=str(exc) or "rewrite stage failed",
                    session_id=session_id,
                    turn=turn,
                    exception_type=type(exc).__name__,
                )
            )
        source_map = [
            {
                "entry_seq": entry["entry_seq"],
                "sources": deepcopy(entry["sources"]),
            }
            for entry in rewritten
        ]
        output = {
            "entries": rewritten,
            "source_map": source_map,
            "summary_length": sum(len(entry["summary_text"]) for entry in rewritten),
            "truncation_count": sum(
                1 for entry in rewritten if entry["truncation"]["truncated"]
            ),
            "fingerprint": fingerprint(rewritten),
            "cache_key": cache_key,
        }
        session.caches["rewrite"][str(turn)] = {"key": cache_key, "output": output}
        return output, False, failures, work_units, time.perf_counter() - started

    def _candidates_for_event(
        self, event: dict[str, Any], current_turn: int
    ) -> list[dict[str, Any]]:
        config = self.config
        weights = config["point_weights"]
        decay = float(weights["position_decay"])
        distance = max(0, current_turn - int(event["turn"]))
        decay_factor = decay**distance
        by_start = {int(item["start"]): item for item in event["tokens"]}
        result: list[dict[str, Any]] = []
        for start, end, text in split_sentences(event["text"]):
            tokens = [
                item
                for item in event["tokens"]
                if int(item["start"]) >= start and int(item["end"]) <= end
            ]
            if not tokens:
                continue
            terms = sorted({str(item["token"]) for item in tokens})
            term_score = sum(
                float(
                    weights["term_weights"].get(
                        str(item["token"]), 1.0
                    )
                )
                for item in tokens
            )
            sentence_bonus = (
                float(weights["sentence_bonus"])
                if _ends_sentence(text)
                else 0.0
            )
            weight = round_weight(
                (float(weights["base"]) + term_score + sentence_bonus)
                * decay_factor
            )
            result.append(
                {
                    "candidate_id": (
                        f"p:{event['session_id']}:{event['turn']}:{start}:{end}"
                    ),
                    "end": end,
                    "source_turn": int(event["turn"]),
                    "start": start,
                    "terms": terms,
                    "text": text,
                    "weight": weight,
                }
            )
        return result

    def _utterance_weight(self, event: dict[str, Any], current_turn: int) -> float:
        weights = self.config["point_weights"]
        term_score = sum(
            float(weights["term_weights"].get(str(token["token"]), 1.0))
            for token in event["tokens"]
        )
        distance = max(0, current_turn - int(event["turn"]))
        return round_weight(
            (float(weights["base"]) + term_score)
            * (float(weights["position_decay"]) ** distance)
        )

    def _select_candidates(
        self, candidates: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []
        weights = np.asarray(
            [float(item["weight"]) for item in candidates], dtype=np.float64
        )
        turns = np.asarray(
            [int(item["source_turn"]) for item in candidates], dtype=np.int64
        )
        starts = np.asarray(
            [int(item["start"]) for item in candidates], dtype=np.int64
        )
        order = np.lexsort((starts, turns, -weights))
        limit = self.config["max_candidates_per_round"]
        return [deepcopy(candidates[int(index)]) for index in order[:limit]]

    def _merge_candidates(
        self,
        prior: list[dict[str, Any]],
        candidates: list[dict[str, Any]],
        turn: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        threshold = float(self.config["dedupe_similarity_threshold"])
        groups: list[dict[str, Any]] = []
        next_seq = 1
        for entry in prior:
            groups.append(
                {
                    "anchor_seq": int(entry["entry_seq"]),
                    "representative_seq": int(entry["entry_seq"]),
                    "entry": deepcopy(entry),
                    "sources": deepcopy(entry["sources"]),
                }
            )
            next_seq = max(next_seq, int(entry["entry_seq"]) + 1)
        records: list[dict[str, Any]] = []
        for candidate in candidates:
            provisional_seq = next_seq
            next_seq += 1
            best_group: dict[str, Any] | None = None
            best_similarity = -1.0
            for group in groups:
                similarity = _jaccard(
                    set(group["entry"].get("terms", [])),
                    set(candidate.get("terms", [])),
                )
                if similarity < threshold:
                    continue
                if (
                    similarity > best_similarity
                    or (
                        similarity == best_similarity
                        and best_group is not None
                        and group["anchor_seq"] < best_group["anchor_seq"]
                    )
                ):
                    best_group = group
                    best_similarity = similarity
            if best_group is None:
                groups.append(
                    {
                        "anchor_seq": provisional_seq,
                        "representative_seq": provisional_seq,
                        "entry": {
                            "entry_seq": provisional_seq,
                            "sources": [candidate_source(candidate)],
                            "terms": list(candidate.get("terms", [])),
                            "text": candidate["text"],
                            "weight": candidate["weight"],
                        },
                        "sources": [candidate_source(candidate)],
                    }
                )
                continue

            replaced = False
            current = best_group["entry"]
            if float(candidate["weight"]) > float(current["weight"]) or (
                float(candidate["weight"]) == float(current["weight"])
                and provisional_seq < int(best_group["representative_seq"])
            ):
                current["text"] = candidate["text"]
                current["terms"] = list(candidate.get("terms", []))
                current["weight"] = candidate["weight"]
                best_group["representative_seq"] = provisional_seq
                replaced = True
            best_group["sources"] = _merge_source_lists(
                best_group["sources"], [candidate_source(candidate)]
            )
            current["sources"] = deepcopy(best_group["sources"])
            records.append(
                {
                    "kept_entry_seq": int(best_group["anchor_seq"]),
                    "kept_source": "candidate" if replaced else "accumulated",
                    "kept_weight": round_weight(current["weight"]),
                    "merged_candidate_ids": [candidate["candidate_id"]],
                    "merged_source_entry_seqs": [
                        int(best_group["anchor_seq"]),
                        provisional_seq,
                    ],
                    "round": turn,
                    "similarity": round_weight(best_similarity),
                }
            )

        entries: list[dict[str, Any]] = []
        for position, group in enumerate(groups, start=1):
            entry = deepcopy(group["entry"])
            entry["entry_seq"] = position
            entry["sources"] = _sort_sources(entry["sources"])
            entries.append(entry)
        entries.sort(key=lambda item: item["entry_seq"])
        return entries, records

    def _rewrite_entry(self, entry: dict[str, Any]) -> dict[str, Any]:
        template = self._choose_template(entry)
        rendered = template["format"].replace("{point}", str(entry["text"]))
        summary_text, truncation = self._truncate(
            rendered, self.config["max_summary_length"]
        )
        return {
            "entry_seq": int(entry["entry_seq"]),
            "summary_text": summary_text,
            "template_id": template["id"],
            "truncation": truncation,
            "weight": round_weight(entry["weight"]),
            "sources": _sort_sources(entry["sources"]),
        }

    def _choose_template(self, entry: dict[str, Any]) -> dict[str, Any]:
        templates = self.config["template_weights"]
        total = sum(float(template["weight"]) for template in templates)
        seed = hashlib.sha256(
            (
                str(entry.get("text", ""))
                + "|"
                + "|".join(
                    source["candidate_id"] for source in entry.get("sources", [])
                )
            ).encode("utf-8")
        ).hexdigest()
        bucket = int(seed[:16], 16) % max(1, int(round(total * 1_000_000)))
        cursor = 0
        for template in templates:
            cursor += int(round(float(template["weight"]) * 1_000_000))
            if bucket < cursor:
                return template
        return templates[-1]

    def _truncate(self, text: str, maximum: int) -> tuple[str, dict[str, Any]]:
        original_length = len(text)
        if original_length <= maximum:
            return text, {
                "discarded_chars": 0,
                "kept_length": original_length,
                "original_length": original_length,
                "truncated": False,
                "truncation_position": None,
            }
        boundary = _last_sentence_boundary(text, maximum)
        cut = boundary if boundary is not None and boundary > 0 else maximum
        kept = text[:cut].rstrip()
        if not kept:
            kept = text[:cut]
        return kept, {
            "discarded_chars": original_length - len(kept),
            "kept_length": len(kept),
            "original_length": original_length,
            "truncated": True,
            "truncation_position": cut,
        }

    def _active_sessions(self) -> list[str]:
        return sorted(
            session_id
            for session_id, session in self.sessions.items()
            if self.rank < len(session.events)
        )

    def _round_cache_output(
        self, session: SessionState, stage: str, turn: int
    ) -> dict[str, Any]:
        cached = session.caches.get(stage, {}).get(str(turn))
        if cached is None:
            return {}
        return deepcopy(cached.get("output", {}))

    def _summary_fingerprint(self) -> str:
        return fingerprint(
            {
                session_id: {
                    "summary": session.summary,
                    "rounds": len(session.rounds),
                }
                for session_id, session in sorted(self.sessions.items())
            }
        )

    def _commit(self, phase: str, *, status: str, stage: str | None) -> None:
        next_version = self.snapshot_version + 1
        if stage is not None:
            self.artifact_versions[stage] = next_version
        if self.snapshot_manager is None:
            self.snapshot_version = next_version
            return
        manifest = self.snapshot_manager.commit(
            state=self._snapshot_state(),
            phase=phase,
            cursor=self.rank,
            status=status,
            config_hash=config_hash(self.config),
            vocab_fingerprint=self.lexicon.fingerprint,
            summary_fingerprint=self._summary_fingerprint(),
            artifact_versions=self.artifact_versions,
            extra={
                "rounds_completed": self.rounds_completed,
                "mode": self.mode,
                "snapshot_interval": self.config["snapshot_interval"],
            },
        )
        self.snapshot_version = int(manifest["snapshot_version"])

    def _snapshot_state(self) -> dict[str, Any]:
        state = self.to_state()
        if self.state_extra is not None:
            state["workbench"] = self.state_extra()
        return state


def split_sentences(text: str) -> list[tuple[int, int, str]]:
    boundaries: list[int] = []
    for index, char in enumerate(text):
        if char in "。！？!?；;":
            boundaries.append(index + 1)
        elif char == ".":
            previous_char = text[index - 1] if index > 0 else ""
            next_char = text[index + 1] if index + 1 < len(text) else ""
            if not (previous_char.isdigit() and next_char.isdigit()):
                boundaries.append(index + 1)
        elif char == "\n":
            boundaries.append(index)
    if not boundaries or boundaries[-1] < len(text):
        boundaries.append(len(text))
    result: list[tuple[int, int, str]] = []
    cursor = 0
    for boundary in boundaries:
        if boundary <= cursor:
            continue
        start = cursor
        while start < boundary and text[start].isspace():
            start += 1
        end = boundary
        while end > start and text[end - 1].isspace():
            end -= 1
        if end > start:
            result.append((start, end, text[start:end]))
        cursor = boundary
    return result


def _ends_sentence(text: str) -> bool:
    stripped = text.rstrip()
    return bool(stripped) and stripped[-1] in "。！？!?；;."


def _last_sentence_boundary(text: str, maximum: int) -> int | None:
    positions = [
        index + 1
        for index, char in enumerate(text[:maximum])
        if char in "。！？!?；;."
    ]
    return positions[-1] if positions else None


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 1.0
    return round_weight(len(left & right) / len(union))


def candidate_source(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": candidate["candidate_id"],
        "end": int(candidate["end"]),
        "start": int(candidate["start"]),
        "turn": int(candidate["source_turn"]),
    }


def _merge_source_lists(
    left: list[dict[str, Any]], right: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for source in [*left, *right]:
        by_id[str(source["candidate_id"])] = deepcopy(source)
    return _sort_sources(list(by_id.values()))


def _sort_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        sources,
        key=lambda item: (
            int(item["turn"]),
            int(item["start"]),
            int(item["end"]),
            str(item["candidate_id"]),
        ),
    )


def core_view(
    result: dict[str, Any], *, include_stage_timing: bool = False
) -> dict[str, Any]:
    """Return the deterministic fields required to match incremental and full runs."""

    sessions: dict[str, Any] = {}
    for session_id, session in sorted(result.get("sessions", {}).items()):
        sessions[session_id] = {
            "session_id": session_id,
            "final_summary": session.get("final_summary", []),
            "final_source_map": session.get("final_source_map", []),
            "rounds": [
                {
                    "turn": round_item.get("turn"),
                    "extraction": round_item.get("extraction", {}),
                    "merge": round_item.get("merge", {}),
                    "rewrite": round_item.get("rewrite", {}),
                    "summary_entries": round_item.get("summary_entries", []),
                    "source_map": round_item.get("source_map", []),
                    "summary_length": round_item.get("summary_length"),
                    "truncation_count": round_item.get("truncation_count"),
                }
                for round_item in session.get("rounds", [])
            ],
        }
    view = {
        "counts": result.get("counts"),
        "failures": result.get("failures"),
        "sessions": sessions,
        "summary_lengths": result.get("summary_lengths"),
        "status": result.get("status"),
    }
    if include_stage_timing:
        view["stage_timing"] = result.get("stage_timing")
    return view
