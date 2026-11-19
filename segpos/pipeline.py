"""增量分词与词性标注流水线引擎。

职责：文本加载、线程池并发处理、内容指纹缓存失效、版本化快照提交、
信号优雅退出、低置信度观测、异常隔离与 JSON 统计报告。
"""

from __future__ import annotations

import signal
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from .cache import DependencyCache
from .config import ComponentConfig
from .errors import ConfigError, StageError
from .fingerprints import stable_sort_key
from .models import Dictionary, DictEntry, ProbabilityTable, SentenceRecord
from .snapshot import SnapshotStore, atomic_write_json
from .tagger import segment_with_oov, tag_tokens

SENTENCE_INVALID_CODE = "E_SENTENCE_INVALID"
STAGE_FAIL_CODE = "E_STAGE_FAILED"
STAGES = ("segmentation", "oov", "tagging")


@dataclass
class RunOutcome:
    report: dict
    invalidated: List[str]
    reused: List[str]
    recomputed: List[str]
    stopped_early: bool


@dataclass
class _LoadedInputs:
    sentences: Dict[str, str]
    failures: List[dict]


def load_sentences(path: str | Path) -> _LoadedInputs:
    """逐行解析 '句子ID<TAB>文本'，校验 ID 唯一且文本非空。"""

    sentences: Dict[str, str] = {}
    failures: List[dict] = []
    seen: set[str] = set()
    raw_path = Path(path)
    with raw_path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.rstrip("\n").rstrip("\r")
            if not line:
                failures.append(
                    {
                        "sentence_id": f"__line_{line_no}",
                        "stage": "load",
                        "code": SENTENCE_INVALID_CODE,
                        "exception_type": "ValueError",
                        "message": "空行",
                        "line_no": line_no,
                    }
                )
                continue
            parts = line.split("\t")
            if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
                failures.append(
                    {
                        "sentence_id": parts[0].strip() if parts and parts[0].strip()
                        else f"__line_{line_no}",
                        "stage": "load",
                        "code": SENTENCE_INVALID_CODE,
                        "exception_type": "ValueError",
                        "message": "行格式应为 句子ID<TAB>文本 且字段非空",
                        "line_no": line_no,
                    }
                )
                continue
            sentence_id, text = parts[0].strip(), parts[1].strip()
            if sentence_id in seen:
                failures.append(
                    {
                        "sentence_id": sentence_id,
                        "stage": "load",
                        "code": SENTENCE_INVALID_CODE,
                        "exception_type": "ValueError",
                        "message": "句子 ID 重复",
                        "line_no": line_no,
                    }
                )
                continue
            seen.add(sentence_id)
            sentences[sentence_id] = text
    return _LoadedInputs(sentences=sentences, failures=failures)


class IncrementalPipeline:
    """可集成的增量流水线；编辑与运行在同一引擎实例上保持血缘一致。"""

    def __init__(
        self,
        config: ComponentConfig,
        dictionary: Dictionary,
        prob_table: ProbabilityTable,
        snapshot_dir: str | Path,
        sentences_path: str | Path,
    ):
        self.config = config
        self.dictionary = dictionary
        self.prob_table = prob_table
        self.snapshot = SnapshotStore(snapshot_dir)
        self.sentences_path = Path(sentences_path)
        self.cache = DependencyCache(prob_table.version)
        self.failures: List[dict] = []
        self.quarantined: set[str] = set()
        self._lineage_edits = 0
        self._last_invalidated: List[str] = []
        self._stop_event = threading.Event()
        self._previous_handlers: Dict[int, object] = {}
        # 测试钩子：{sentence_id: stage}，命中即在对应阶段抛出异常。
        self.fault_injection: Dict[str, str] = {}

    # ---------- 信号处理 ----------
    def _handle_signal(self, signum, frame):  # pragma: no cover - 平台分支
        self._stop_event.set()

    def install_signal_handlers(self) -> None:
        if not hasattr(signal, "SIGTERM"):  # pragma: no cover - Windows 分支
            return
        for sig in (signal.SIGTERM, signal.SIGINT):
            self._previous_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, self._handle_signal)

    def restore_signal_handlers(self) -> None:  # pragma: no cover - 平台分支
        if not hasattr(signal, "SIGTERM"):
            return
        for sig, handler in self._previous_handlers.items():
            signal.signal(sig, handler)
        self._previous_handlers.clear()

    def request_stop(self) -> None:
        self._stop_event.set()

    # ---------- 续跑 ----------
    def _restore_from_snapshot(self, inputs: _LoadedInputs) -> None:
        manifest = self.snapshot.verify_resume(
            self.dictionary.hash,
            self.prob_table.hash,
            self.prob_table.version,
        )
        records = self.snapshot.load_results(manifest)
        quarantine_ids = sorted(
            {item["sentence_id"] for item in manifest.failures}
        )
        self.cache.restore(records, inputs.sentences, quarantine_ids)
        self.failures = list(manifest.failures)
        self.quarantined = set(quarantine_ids)

    # ---------- 单句处理 ----------
    def _process_sentence(self, sentence_id: str, text: str) -> SentenceRecord:
        injected_stage = self.fault_injection.get(sentence_id)

        def maybe_fault(stage: str) -> None:
            if injected_stage == stage:
                raise StageError(
                    f"故障注入: {sentence_id} 在 {stage} 阶段失败",
                    stage=stage,
                    code=STAGE_FAIL_CODE,
                    exception_type="InjectedStageError",
                )

        maybe_fault("segmentation")
        tokens, hit_words, oov_smoothing = segment_with_oov(
            text, self.dictionary, self.prob_table, self.config.smoothing
        )
        maybe_fault("oov")
        tokens, sentence_logprob, tag_smoothing = tag_tokens(
            tokens, self.prob_table, self.config.smoothing
        )
        maybe_fault("tagging")

        key = self.cache.key_for(sentence_id, text, hit_words)
        record = SentenceRecord(
            sentence_id=sentence_id,
            text=text,
            tokens=tokens,
            cache_key=key,
            reused=False,
            sentence_logprob=sentence_logprob,
            smoothing_hits=oov_smoothing + tag_smoothing,
        )
        record.low_confidence_words = self._collect_low_confidence(record)
        return record, hit_words

    def _collect_low_confidence(self, record: SentenceRecord) -> List[dict]:
        items: List[dict] = []
        for token in record.tokens:
            if token.boundary_confidence < self.config.low_confidence_threshold:
                items.append(
                    {
                        "sentence_id": record.sentence_id,
                        "start": token.start,
                        "end": token.end,
                        "text": token.text,
                        "boundary_confidence": round(
                            float(token.boundary_confidence), 12
                        ),
                        "pos": token.pos,
                    }
                )
        return items

    # ---------- 主运行 ----------
    def run(self, report_path: str | Path | None = None) -> RunOutcome:
        inputs = load_sentences(self.sentences_path)
        load_failures = list(inputs.failures)
        load_quarantine = {item["sentence_id"] for item in load_failures}

        resumed = False
        if self.snapshot.exists() and self._lineage_edits == 0:
            self._restore_from_snapshot(inputs)
            resumed = True
            existing = {item["sentence_id"]: item for item in self.failures}
            for item in load_failures:
                existing.setdefault(item["sentence_id"], item)
            self.failures = list(existing.values())
            self.quarantined = set(self.cache.quarantine_ids()) | load_quarantine
        else:
            self.failures = load_failures
            self.quarantined = set(load_quarantine)

        ordered_ids = sorted(inputs.sentences)
        invalidated: List[str] = []
        reused: List[str] = []
        recomputed: List[str] = []
        pending: List[str] = []
        for sentence_id in ordered_ids:
            if self.cache.is_quarantined(sentence_id):
                continue
            text = inputs.sentences[sentence_id]
            if self.cache.has(sentence_id):
                record = self.cache.get_record(sentence_id)
                record.reused = True
                reused.append(sentence_id)
            else:
                pending.append(sentence_id)

        invalidated = list(self._last_invalidated)
        self._last_invalidated = []

        processed: List[SentenceRecord] = []
        stopped_early = False
        commit_interval = self.config.snapshot_commit_interval

        def task_key(sentence_id: str) -> str:
            return stable_sort_key(sentence_id, "analyze")

        ordered_pending = sorted(pending, key=task_key)
        executor = ThreadPoolExecutor(max_workers=self.config.workers)
        try:
            futures = {
                executor.submit(self._process_sentence, sid, inputs.sentences[sid]): sid
                for sid in ordered_pending
            }
            future_by_id = {sid: future for future, sid in futures.items()}
            batch: List[SentenceRecord] = []
            for order_index, sid in enumerate(ordered_pending):
                if self._stop_event.is_set():
                    for pending_id in ordered_pending[order_index:]:
                        future_by_id[pending_id].cancel()
                    if batch:
                        self._commit(inputs)
                        batch.clear()
                    stopped_early = True
                    break
                future = future_by_id[sid]
                try:
                    record, hit_words = future.result()
                except StageError as exc:
                    self.failures.append(
                        {
                            "sentence_id": sid,
                            "stage": exc.stage,
                            "code": exc.code,
                            "exception_type": exc.exception_type,
                            "message": exc.message,
                        }
                    )
                    self.quarantined.add(sid)
                    self.cache.quarantine(sid)
                    continue
                except Exception as exc:  # 异常隔离：其余句子继续
                    self.failures.append(
                        {
                            "sentence_id": sid,
                            "stage": "unknown",
                            "code": STAGE_FAIL_CODE,
                            "exception_type": type(exc).__name__,
                            "message": str(exc),
                        }
                    )
                    self.quarantined.add(sid)
                    self.cache.quarantine(sid)
                    continue
                record.reused = False
                self.cache.put(record, hit_words, inputs.sentences[sid])
                batch.append(record)
                processed.append(record)
                recomputed.append(sid)
                if len(batch) >= commit_interval:
                    self._commit(inputs)
                    batch.clear()
            if batch:
                self._commit(inputs)
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

        all_records = self.cache.all_records()
        low_confidence_count = self._count_low_confidence(all_records)

        self.failures.sort(
            key=lambda item: stable_sort_key(item["sentence_id"], item["stage"])
        )
        report = self._build_report(
            total=len(ordered_ids),
            all_records=all_records,
            invalidated=invalidated,
            reused=reused,
            recomputed=recomputed,
            low_confidence_count=low_confidence_count,
            stopped_early=stopped_early,
        )
        if report_path is not None and not stopped_early:
            atomic_write_json(Path(report_path), report)
        return RunOutcome(
            report=report,
            invalidated=invalidated,
            reused=reused,
            recomputed=recomputed,
            stopped_early=stopped_early,
        )

    def _commit(self, inputs: _LoadedInputs):
        records = self.cache.all_records()
        self.snapshot.commit(
            records=records,
            failures=self.failures,
            dict_hash=self.dictionary.hash,
            prob_table_hash=self.prob_table.hash,
            prob_table_version=self.prob_table.version,
        )

    def _count_low_confidence(self, records: List[SentenceRecord]) -> int:
        return sum(
            1
            for record in records
            for token in record.tokens
            if token.boundary_confidence < self.config.low_confidence_threshold
        )

    def _build_report(
        self,
        total: int,
        all_records: List[SentenceRecord],
        invalidated: List[str],
        reused: List[str],
        recomputed: List[str],
        low_confidence_count: int,
        stopped_early: bool,
    ) -> dict:
        failure_ids = {item["sentence_id"] for item in self.failures}
        return {
            "total_sentences": total,
            "success_sentences": len(all_records),
            "failed_sentences": len(failure_ids),
            "low_confidence_word_count": low_confidence_count,
            "dict_hash": self.dictionary.hash,
            "prob_table_hash": self.prob_table.hash,
            "prob_table_version": self.prob_table.version,
            "snapshot_version": self.snapshot.global_version,
            "invalidated_sentence_count": len(invalidated),
            "invalidated_sentences": invalidated,
            "reused_sentence_count": len(reused),
            "reused_sentences": reused,
            "recomputed_sentence_count": len(recomputed),
            "recomputed_sentences": recomputed,
            "stopped_early": stopped_early,
            "failures": self.failures,
        }

    # ---------- 增量编辑 ----------
    def _rebuild_dictionary(self, entries: List[DictEntry]) -> Dictionary:
        allowed = set(self.prob_table.tags)
        return Dictionary(entries, allowed_pos=allowed)

    def add_word(self, word: str, freq: int, pos: str) -> List[str]:
        if pos not in self.prob_table.tag_index:
            raise ConfigError(f"词性 {pos} 不在概率表词性集合内")
        entries = self.dictionary.to_entries()
        if self.dictionary.contains(word):
            raise ConfigError(f"词条已存在，请使用 update_freq: {word}")
        entries.append(DictEntry(word=word, freq=freq, pos=pos))
        self.dictionary = self._rebuild_dictionary(entries)
        invalidated = self.cache.invalidate_word_occurrences(word)
        self._lineage_edits += 1
        self._last_invalidated = invalidated
        return invalidated

    def remove_word(self, word: str) -> List[str]:
        if not self.dictionary.contains(word):
            raise ConfigError(f"待删除词条不存在: {word}")
        entries = [
            entry for entry in self.dictionary.to_entries() if entry.word != word
        ]
        self.dictionary = self._rebuild_dictionary(entries)
        invalidated = self.cache.invalidate_words([word])
        self._lineage_edits += 1
        self._last_invalidated = invalidated
        return invalidated

    def update_freq(self, word: str, freq: int) -> List[str]:
        current = self.dictionary.get(word)
        if current is None:
            raise ConfigError(f"待调整词频的词条不存在: {word}")
        if not isinstance(freq, int) or isinstance(freq, bool) or freq <= 0:
            raise ConfigError("词频必须为正整数")
        entries = [
            DictEntry(word=current.word, freq=freq, pos=current.pos)
            if entry.word == word
            else entry
            for entry in self.dictionary.to_entries()
        ]
        self.dictionary = self._rebuild_dictionary(entries)
        invalidated = self.cache.invalidate_words([word])
        self._lineage_edits += 1
        self._last_invalidated = invalidated
        return invalidated

    def batch_edit(self, edits: List[dict]) -> List[str]:
        """批量编辑：add/remove/update_freq，合并受影响词条后一次失效。"""

        entries = self.dictionary.to_entries()
        entry_map = {entry.word: entry for entry in entries}
        affected: List[str] = []
        for edit in edits:
            action = edit.get("action")
            word = edit.get("word")
            affected.append(word)
            if action == "add":
                if word in entry_map:
                    raise ConfigError(f"批量编辑中词条已存在: {word}")
                pos = edit["pos"]
                if pos not in self.prob_table.tag_index:
                    raise ConfigError(f"词性 {pos} 不在概率表词性集合内")
                entry_map[word] = DictEntry(word, int(edit["freq"]), pos)
            elif action == "remove":
                entry_map.pop(word, None)
            elif action == "update_freq":
                if word not in entry_map:
                    raise ConfigError(f"批量编辑中词条不存在: {word}")
                old = entry_map[word]
                entry_map[word] = DictEntry(word, int(edit["freq"]), old.pos)
            else:
                raise ConfigError(f"未知词典编辑动作: {action}")
        self.dictionary = self._rebuild_dictionary(list(entry_map.values()))
        invalidated = self.cache.invalidate_words(sorted(set(affected)))
        self._lineage_edits += 1
        self._last_invalidated = invalidated
        return invalidated

    def replace_prob_table(self, table: ProbabilityTable) -> List[str]:
        self.prob_table = table
        invalidated = self.cache.replace_prob_table(table.version)
        self._lineage_edits += 1
        self._last_invalidated = invalidated
        return invalidated

    def replace_sentence_text(self, sentence_id: str, new_text: str) -> bool:
        if not new_text or not new_text.strip():
            raise ConfigError("替换文本不能为空")
        removed = self.cache.invalidate_sentence(sentence_id)
        self.cache.release_quarantine(sentence_id)
        self.quarantined.discard(sentence_id)
        self.failures = [
            item for item in self.failures if item["sentence_id"] != sentence_id
        ]
        self._lineage_edits += 1
        self._last_invalidated = [sentence_id] if removed else []
        return removed

    def results_by_id(self) -> Dict[str, dict]:
        return {
            record.sentence_id: record.to_dict()
            for record in self.cache.all_records()
        }
