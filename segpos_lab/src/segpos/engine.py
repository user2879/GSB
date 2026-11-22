"""增量交互引擎：精确失效、阶段缓存、首响预算与线程池并发。"""
from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from .errors import (
    E_COMMAND_INVALID,
    E_INTERACT_BUDGET_PENDING,
    Failure,
)
from .fingerprint import combine, fingerprint
from .loader import validate_sentence_text
from .models import Config, Lexicon, LexiconEntry, ProbTable, Sentence, Token
from .oov import process_oov
from .segmenter import FmmSegmenter
from .snapshot import STAGES, SnapshotStore, atomic_write_json
from .tagger import viterbi


def stable_task_key(sid: str, stage: str) -> str:
    return f"{sid}::{stage}"


@dataclass
class SentenceState:
    sid: str
    text: str
    seg_tokens: list[Token] = field(default_factory=list)
    seg_meta: dict = field(default_factory=dict)
    oov_tokens: list[Token] = field(default_factory=list)
    oov_meta: dict = field(default_factory=dict)
    tag_tokens: list[Token] = field(default_factory=list)
    tag_meta: dict = field(default_factory=dict)
    referenced_entries: set[str] = field(default_factory=set)
    stage_keys: dict[str, str] = field(default_factory=dict)
    stages_valid: dict[str, bool] = field(default_factory=dict)


class IncrementalEngine:
    def __init__(
        self,
        config: Config,
        sentences: list[Sentence],
        lexicon: Lexicon,
        prob_table: ProbTable,
        charset: set[str],
        failures: list[Failure],
        snapshot_dir: str | Path,
    ):
        self.config = config
        self.sentences = sentences
        self.lexicon = lexicon
        self.prob_table = prob_table
        self.charset = charset
        self.failures = list(failures)
        self.states: dict[str, SentenceState] = {
            s.sid: SentenceState(sid=s.sid, text=s.text) for s in sentences
        }
        self.snapshot = SnapshotStore(snapshot_dir)
        self.global_version = 0
        self.stage_versions = {stage: 0 for stage in STAGES}
        self.reuse_counts = {stage: 0 for stage in STAGES}
        self.stage_timings_ms = {stage: 0.0 for stage in STAGES}
        self.commands_processed = 0
        self.pending: list[str] = []
        self.recovery_version: int | None = None
        self._restore_snapshot()

    # ------------------------------------------------------------------ keys
    def _seg_cache_key(self, state: SentenceState) -> str:
        text_fp = fingerprint(state.text, prefix="sentence-text")
        return combine(
            "cache-segment",
            text_fp,
            self.lexicon.snapshot_fingerprint(),
        )

    def _oov_cache_key(self, state: SentenceState) -> str:
        text_fp = fingerprint(state.text, prefix="sentence-text")
        return combine(
            "cache-oov",
            text_fp,
            self.lexicon.snapshot_fingerprint(),
            self.config.oov_param_fingerprint(),
        )

    def _tag_cache_key(self, state: SentenceState) -> str:
        text_fp = fingerprint(state.text, prefix="sentence-text")
        return combine(
            "cache-tag",
            text_fp,
            self.lexicon.snapshot_fingerprint(),
            self.config.oov_param_fingerprint(),
            self.prob_table.table_fingerprint(),
            self.config.param_fingerprint(),
        )

    # ------------------------------------------------------------- per-stage
    def _compute_segment(self, state: SentenceState) -> None:
        sentence = Sentence(sid=state.sid, text=state.text, source_line=-1)
        segmenter = FmmSegmenter(self.lexicon)
        tokens, meta = segmenter.segment(sentence)
        state.seg_tokens = tokens
        state.seg_meta = meta
        state.referenced_entries = set(meta["hit_entries"])

    def _compute_oov(self, state: SentenceState) -> None:
        tokens, meta = process_oov(state.seg_tokens, state.seg_meta, self.config)
        state.oov_tokens = tokens
        state.oov_meta = meta

    def _compute_tag(self, state: SentenceState) -> None:
        tokens = [
            Token(
                text=t.text,
                start=t.start,
                end=t.end,
                source=t.source,
                freq=t.freq,
                match_len=t.match_len,
                segment_basis=t.segment_basis,
                fingerprint=t.fingerprint,
            )
            for t in state.oov_tokens
        ]
        tokens, meta = viterbi(tokens, self.prob_table, self.config, state.oov_meta)
        state.tag_tokens = tokens
        state.tag_meta = meta

    def _stage_work(self, sid: str, stage: str) -> tuple[str, str, float]:
        started = time.perf_counter()
        state = self.states[sid]
        if stage == "segment":
            self._compute_segment(state)
        elif stage == "oov":
            self._compute_oov(state)
        elif stage == "tag":
            self._compute_tag(state)
        delay = self.config.simulated_task_delay_ms / 1000.0
        if delay > 0:
            time.sleep(delay)
        return sid, stage, (time.perf_counter() - started) * 1000.0

    # --------------------------------------------------------------- planning
    def _plan_work(self, affected: set[str]) -> list[tuple[str, str]]:
        ordered_ids = sorted(self.states)
        plan: list[tuple[str, str]] = []
        for sid in ordered_ids:
            state = self.states[sid]
            if sid in affected:
                for stage in ("segment", "oov", "tag"):
                    state.stages_valid[stage] = False
            seg_key = self._seg_cache_key(state)
            seg_needed = not state.stages_valid.get("segment") or state.stage_keys.get("segment") != seg_key
            if seg_needed:
                plan.append((sid, "segment"))
            oov_key = self._oov_cache_key(state)
            oov_needed = (
                seg_needed
                or not state.stages_valid.get("oov")
                or state.stage_keys.get("oov") != oov_key
            )
            if oov_needed:
                plan.append((sid, "oov"))
            tag_key = self._tag_cache_key(state)
            tag_needed = (
                oov_needed
                or not state.stages_valid.get("tag")
                or state.stage_keys.get("tag") != tag_key
            )
            if tag_needed:
                plan.append((sid, "tag"))
        plan.sort(key=lambda item: stable_task_key(item[0], item[1]))
        return plan

    def _mark_completed(self, sid: str, stage: str) -> None:
        state = self.states[sid]
        state.stage_keys[stage] = self._stage_cache_key(stage, state)
        state.stages_valid[stage] = True

    # ------------------------------------------------------------ scheduling
    def run_due(self, affected: set[str]) -> dict:
        plan = self._plan_work(affected)
        budget_s = self.config.first_response_budget_ms / 1000.0
        deadline = time.perf_counter() + budget_s
        completed: list[tuple[str, str]] = []
        pending: list[tuple[str, str]] = []
        budget_pending = False

        planned = set(plan)
        if not plan:
            self.pending = []
        else:
            waves: dict[str, list[str]] = {"segment": [], "oov": [], "tag": []}
            for sid, stage in plan:
                waves[stage].append(sid)
            for stage in ("segment", "oov", "tag"):
                sids = sorted(waves[stage])
                if not sids:
                    continue
                max_workers = min(self.config.concurrency, len(sids))
                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    futures = {pool.submit(self._stage_work, sid, stage): sid for sid in sids}
                    ordered = sorted(futures, key=lambda fut: stable_task_key(futures[fut], stage))
                    wave_timed_out = False
                    for future in ordered:
                        remaining = deadline - time.perf_counter()
                        try:
                            sid, done_stage, elapsed = future.result(
                                timeout=max(remaining, 0.0) if remaining > 0 else 0.0001
                            )
                        except TimeoutError:
                            budget_pending = True
                            wave_timed_out = True
                            continue
                        self.stage_timings_ms[done_stage] += elapsed
                        self._mark_completed(sid, done_stage)
                        completed.append((sid, done_stage))
                    for future, sid in futures.items():
                        if not future.done():
                            future.cancel()
                            pending.append((sid, stage))
                    if wave_timed_out:
                        # 预算耗尽：当前波未完成句子与后续波整体保留为待办
                        done_in_wave = {sid for sid, done_stage in completed if done_stage == stage}
                        for sid in sids:
                            if sid not in done_in_wave:
                                pending.append((sid, stage))
                        later = {"segment": ("oov", "tag"), "oov": ("tag",), "tag": ()}[stage]
                        for later_stage in later:
                            for later_sid in waves[later_stage]:
                                pending.append((later_sid, later_stage))
                        break
            deduped = sorted(set(pending), key=lambda item: stable_task_key(item[0], item[1]))
            pending = deduped
            self.pending = sorted({sid for sid, _ in pending})

        for sid in sorted(self.states):
            state = self.states[sid]
            for stage, key in (
                ("segment", self._seg_cache_key(state)),
                ("oov", self._oov_cache_key(state)),
                ("tag", self._tag_cache_key(state)),
            ):
                if (
                    (sid, stage) not in planned
                    and state.stages_valid.get(stage)
                    and state.stage_keys.get(stage) == key
                ):
                    self.reuse_counts[stage] += 1

        response = self.summary(affected, completed, pending)
        if budget_pending and pending:
            response["code"] = E_INTERACT_BUDGET_PENDING
        return response

    # -------------------------------------------------------------- commands
    def apply_command(self, command: dict) -> set[str]:
        cmd_type = command["type"]
        affected: set[str] = set()
        if cmd_type in ("add_entry", "adjust_freq"):
            word = command["word"]
            freq = command["freq"]
            tag = command.get("tag")
            if not isinstance(word, str) or not word or not isinstance(freq, (int, float)) or isinstance(freq, bool) or not math.isfinite(freq) or freq <= 0:
                self.failures.append(Failure(E_COMMAND_INVALID, str(word), "invalid word or freq", "commands", command.get("_index", 0)))
                return affected
            if tag is not None and tag not in set(self.prob_table.tags):
                self.failures.append(Failure(E_COMMAND_INVALID, word, f"tag {tag!r} undeclared", "commands", command.get("_index", 0)))
                return affected
            old = self.lexicon.entries.get(word)
            if cmd_type == "add_entry":
                if old is not None:
                    if tag is not None and tag != old.tag:
                        self.failures.append(Failure(E_COMMAND_INVALID, word, "tag conflicts with existing entry", "commands", command.get("_index", 0)))
                        return affected
                    old.freq += freq
                    old.sources.append(f"command:{command.get('_index')}")
                else:
                    self.lexicon.entries[word] = LexiconEntry(word=word, freq=float(freq), tag=tag or self.prob_table.tags[0], sources=[f"command:{command.get('_index')}"])
            else:
                if old is None:
                    self.failures.append(Failure(E_COMMAND_INVALID, word, "adjust_freq on missing entry", "commands", command.get("_index", 0)))
                    return affected
                old.freq = float(freq)
                old.sources.append(f"command:{command.get('_index')}")
            affected = self._sentences_referencing(word)
        elif cmd_type == "remove_entry":
            word = command["word"]
            if word not in self.lexicon.entries:
                self.failures.append(Failure(E_COMMAND_INVALID, word, "remove_entry on missing entry", "commands", command.get("_index", 0)))
                return affected
            del self.lexicon.entries[word]
            affected = self._sentences_referencing(word)
        elif cmd_type == "replace_prob_table":
            table = command.get("table")
            if not isinstance(table, ProbTable):
                self.failures.append(Failure(E_COMMAND_INVALID, "prob_table", "replacement table invalid", "commands", command.get("_index", 0)))
                return affected
            self.prob_table = table
            affected = set(self.states)
        elif cmd_type == "replace_sentence":
            sid = command["sid"]
            text = command["text"]
            reason = validate_sentence_text(text, self.charset) if isinstance(text, str) else "text must be string"
            if reason is not None or sid not in self.states:
                self.failures.append(Failure(E_COMMAND_INVALID, str(sid), reason or "unknown sentence id", "commands", command.get("_index", 0)))
                return affected
            if self.states[sid].text != text:
                self.states[sid].text = text
                affected.add(sid)
        else:
            self.failures.append(Failure(E_COMMAND_INVALID, cmd_type, "unsupported command", "commands", command.get("_index", 0)))
        return affected

    def _sentences_referencing(self, word: str) -> set[str]:
        return {sid for sid, state in self.states.items() if word in state.referenced_entries}

    # ---------------------------------------------------------------- summary
    def _completed_sids(self) -> list[str]:
        out = []
        for sid in sorted(self.states):
            state = self.states[sid]
            if all(state.stages_valid.get(stage) for stage in ("segment", "oov", "tag")):
                out.append(sid)
        return out

    def summary(self, affected: set[str], completed: list[tuple[str, str]], pending: list[tuple[str, str]]) -> dict:
        completed_sids = self._completed_sids()
        low_conf: list[dict] = []
        for sid in sorted(completed_sids):
            for item in self.states[sid].tag_meta.get("low_confidence", []):
                low_conf.append({"sid": sid, **item})
        low_conf.sort(key=lambda item: (item["sid"], item["word_index"]))
        total = len(self.states)
        return {
            "code": "OK",
            "global_version": self.global_version,
            "total_sentences": total,
            "completed_sentences": len(completed_sids),
            "progress": len(completed_sids) / total if total else 1.0,
            "low_confidence_count": len(low_conf),
            "affected_sentences": sorted(affected),
            "completed_tasks": [stable_task_key(sid, stage) for sid, stage in completed],
            "pending_sentences": sorted({sid for sid, _ in pending}),
            "reuse_counts": dict(self.reuse_counts),
        }

    # -------------------------------------------------------------- snapshots
    def _stage_payload(self, stage: str) -> dict:
        if stage == "load":
            return {
                "lexicon": {word: entry.to_dict() for word, entry in sorted(self.lexicon.entries.items())},
                "prob_table_hash": self.prob_table.table_fingerprint(),
                "sentences": [
                    {"sid": sid, "text": self.states[sid].text} for sid in sorted(self.states)
                ],
                "failures": [failure.to_dict() for failure in self._sorted_failures()],
            }
        sentences_part: dict[str, dict] = {}
        for sid in sorted(self.states):
            state = self.states[sid]
            if stage == "segment":
                if not state.stages_valid.get("segment"):
                    continue
                sentences_part[sid] = {
                    "tokens": [token.to_dict() for token in state.seg_tokens],
                    "meta": state.seg_meta,
                }
            elif stage == "oov":
                if not state.stages_valid.get("oov"):
                    continue
                sentences_part[sid] = {
                    "tokens": [token.to_dict() for token in state.oov_tokens],
                    "meta": state.oov_meta,
                }
            else:
                if not state.stages_valid.get("tag"):
                    continue
                sentences_part[sid] = {
                    "tokens": [token.to_dict() for token in state.tag_tokens],
                    "meta": state.tag_meta,
                }
        return {"global_version": self.global_version, "stage": stage, "sentences": sentences_part}

    def _sorted_failures(self) -> list[Failure]:
        return sorted(self.failures, key=lambda f: (f.source_line, f.code, f.key))

    def commit_snapshot(self, budget_consumed: bool) -> dict:
        lex_hash = self.lexicon.snapshot_fingerprint()
        prob_hash = self.prob_table.table_fingerprint()
        for stage in STAGES:
            self.stage_versions[stage] = self.global_version
        self.snapshot.begin()
        for stage in STAGES:
            self.snapshot.write_stage(stage, self.global_version, self._stage_payload(stage))
        return self.snapshot.complete(
            global_version=self.global_version,
            lexicon_hash=lex_hash,
            prob_hash=prob_hash,
            budget_consumed=budget_consumed,
            pending=self.pending,
            stage_versions=self.stage_versions,
        )

    def _restore_snapshot(self) -> None:
        manifest = self.snapshot.restore()
        if manifest is None:
            return
        self.global_version = int(manifest["global_version"])
        self.stage_versions = dict(manifest["stage_versions"])
        self.pending = list(manifest.get("pending_sentences", []))
        self.recovery_version = self.global_version
        for stage in ("segment", "oov", "tag"):
            wrapped = self.snapshot.read_stage(stage)
            if not wrapped:
                continue
            payload = wrapped["payload"]
            for sid, part in payload.get("sentences", {}).items():
                state = self.states.get(sid)
                if state is None:
                    continue
                tokens = [Token(**fields) for fields in part["tokens"]]
                if stage == "segment":
                    state.seg_tokens, state.seg_meta = tokens, part["meta"]
                    state.referenced_entries = set(part["meta"].get("hit_entries", []))
                elif stage == "oov":
                    state.oov_tokens, state.oov_meta = tokens, part["meta"]
                else:
                    state.tag_tokens, state.tag_meta = tokens, part["meta"]
                state.stages_valid[stage] = True
                state.stage_keys[stage] = self._stage_cache_key(stage, state)

    def _stage_cache_key(self, stage: str, state: SentenceState) -> str:
        if stage == "segment":
            return self._seg_cache_key(state)
        if stage == "oov":
            return self._oov_cache_key(state)
        return self._tag_cache_key(state)

    # ------------------------------------------------------------------ report
    def build_report(self, command_count: int, extra_failures: list[Failure] | None = None) -> dict:
        all_failures = self._sorted_failures()
        if extra_failures:
            all_failures = sorted(all_failures + list(extra_failures), key=lambda f: (f.source_line, f.code, f.key))
        completed_sids = self._completed_sids()
        seg_token_total = sum(len(self.states[sid].seg_tokens) for sid in completed_sids)
        oov_total = sum(self.states[sid].oov_meta.get("oov_token_count", 0) for sid in completed_sids)
        low_conf: list[dict] = []
        for sid in sorted(completed_sids):
            for item in self.states[sid].tag_meta.get("low_confidence", []):
                low_conf.append({"sid": sid, **item})
        low_conf.sort(key=lambda item: (item["sid"], item["word_index"]))
        per_sentence = {
            sid: {
                "segment": [token.to_dict() for token in self.states[sid].seg_tokens],
                "segment_meta": self.states[sid].seg_meta,
                "oov": [token.to_dict() for token in self.states[sid].oov_tokens],
                "oov_meta": self.states[sid].oov_meta,
                "tag": [token.to_dict() for token in self.states[sid].tag_tokens],
                "tag_meta": self.states[sid].tag_meta,
            }
            for sid in completed_sids
        }
        return {
            "total_sentences": len(self.states),
            "success_count": len(completed_sids),
            "failure_count": len(all_failures),
            "failures": [failure.to_dict() for failure in all_failures],
            "segment_token_total": seg_token_total,
            "oov_token_total": oov_total,
            "low_confidence": low_conf,
            "reuse_counts": dict(self.reuse_counts),
            "snapshot_version": self.global_version,
            "recovery_version": self.recovery_version,
            "pending_sentences": list(self.pending),
            "stage_timings_ms": {k: round(v, 6) for k, v in self.stage_timings_ms.items()},
            "commands_processed": command_count,
            "sentences": per_sentence,
        }

    def run_command_stream(self, commands: list[dict], report_path: str | Path | None = None) -> list[dict]:
        responses: list[dict] = []
        budget_consumed = False
        for command in commands:
            affected = self.apply_command(command)
            self.global_version += 1
            response = self.run_due(affected)
            if response["code"] == E_INTERACT_BUDGET_PENDING:
                budget_consumed = True
            responses.append({"command_index": command.get("_index"), "response": response})
            self.commands_processed += 1
            if self.commands_processed % self.config.snapshot_commit_interval_commands == 0:
                self.commit_snapshot(budget_consumed)
        self.commit_snapshot(budget_consumed)
        report = self.build_report(len(commands))
        if report_path is not None:
            atomic_write_json(Path(report_path), report)
        return responses
