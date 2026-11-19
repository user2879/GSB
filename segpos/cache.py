"""基于内容指纹的依赖失效缓存与句子级倒排索引。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set

from .fingerprints import cache_key
from .models import SentenceRecord


@dataclass
class InvalidationReport:
    invalidated: List[str] = field(default_factory=list)
    reused: List[str] = field(default_factory=list)
    recomputed: List[str] = field(default_factory=list)


class DependencyCache:
    """缓存句子结果，并维护 词条 -> 句子集合 倒排索引。"""

    def __init__(self, prob_table_version: int):
        self._prob_table_version = prob_table_version
        self._records: Dict[str, SentenceRecord] = {}
        self._word_index: Dict[str, Set[str]] = {}
        self._texts: Dict[str, str] = {}
        self._hit_words: Dict[str, List[str]] = {}
        self._quarantine: Set[str] = set()

    @property
    def prob_table_version(self) -> int:
        return self._prob_table_version


    def quarantine(self, sentence_id: str) -> None:
        self._evict(sentence_id)
        self._quarantine.add(sentence_id)

    def is_quarantined(self, sentence_id: str) -> bool:
        return sentence_id in self._quarantine

    def release_quarantine(self, sentence_id: str) -> None:
        self._quarantine.discard(sentence_id)

    def quarantine_ids(self) -> List[str]:
        return sorted(self._quarantine)

    def replace_prob_table(self, version: int) -> List[str]:
        """概率表替换：全部句子失效，返回失效句子 ID（有序）。"""

        invalidated = sorted(self._records)
        self._prob_table_version = version
        self._records.clear()
        return invalidated

    def invalidate_words(self, words: List[str]) -> List[str]:
        """词典增删/词频调整：仅命中受影响词条的句子失效。"""

        affected: Set[str] = set()
        for word in words:
            affected.update(self._word_index.get(word, set()))
        for sentence_id in affected:
            self._evict(sentence_id)
        return sorted(affected)


    def invalidate_word_occurrences(self, word: str) -> List[str]:
        """新增词条场景：旧索引无该词，按当前缓存文本子串扫描受影响句子。"""

        affected = {
            sentence_id
            for sentence_id, text in self._texts.items()
            if word in text
        }
        for sentence_id in sorted(affected):
            self._evict(sentence_id)
        return sorted(affected)

    def invalidate_sentence(self, sentence_id: str) -> bool:
        """单句文本替换：仅该句失效。"""

        if sentence_id in self._records:
            self._evict(sentence_id)
            return True
        return False

    def _evict(self, sentence_id: str) -> None:
        self._records.pop(sentence_id, None)
        for word in self._hit_words.pop(sentence_id, []):
            bucket = self._word_index.get(word)
            if bucket is not None:
                bucket.discard(sentence_id)
                if not bucket:
                    self._word_index.pop(word, None)
        self._texts.pop(sentence_id, None)

    def key_for(self, sentence_id: str, text: str, hit_words: List[str]) -> str:
        return cache_key(text, hit_words, self._prob_table_version)

    def get(
        self, sentence_id: str, text: str, hit_words: List[str]
    ) -> SentenceRecord | None:
        record = self._records.get(sentence_id)
        if record is None:
            return None
        expected = self.key_for(sentence_id, text, hit_words)
        if record.cache_key != expected:
            return None
        return record

    def put(self, record: SentenceRecord, hit_words: List[str], text: str) -> None:
        if record.sentence_id in self._records:
            self._evict(record.sentence_id)
        self._records[record.sentence_id] = record
        self._hit_words[record.sentence_id] = sorted(set(hit_words))
        self._texts[record.sentence_id] = text
        for word in hit_words:
            self._word_index.setdefault(word, set()).add(record.sentence_id)

    def has(self, sentence_id: str) -> bool:
        return sentence_id in self._records

    def get_record(self, sentence_id: str) -> SentenceRecord | None:
        return self._records.get(sentence_id)

    def all_records(self) -> List[SentenceRecord]:
        return [self._records[key] for key in sorted(self._records)]

    def cached_ids(self) -> List[str]:
        return sorted(self._records)

    def restore(
        self,
        records: List[SentenceRecord],
        text_lookup: Dict[str, str],
        quarantine_ids: List[str] | None = None,
    ) -> None:
        """从快照恢复缓存、倒排索引与隔离集合。"""

        for record in records:
            hit_words = [
                token.text for token in record.tokens if token.hit
            ]
            self.put(record, sorted(set(hit_words)), text_lookup[record.sentence_id])
        if quarantine_ids:
            self._quarantine.update(quarantine_ids)
