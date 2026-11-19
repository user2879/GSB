"""内容指纹缓存与精确失效。

缓存键由词序列指纹、模型参数指纹、映射表指纹与版本号派生；
每个依存弧记录引用词集合，样本替换 / 映射变更 / 模型变更时仅失效受影响弧。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .fingerprints import (
    arc_fingerprint,
    artifact_fingerprint,
    sentence_fingerprint,
    word_sequence_fingerprint,
)
from .models import Corpus, Sentence


@dataclass
class ArcCacheEntry:
    arc_key: str
    referenced_words: list[int]
    relations: list[str]


@dataclass
class SentenceCacheEntry:
    sent_id: str
    sentence_key: str
    word_fp: str
    arcs: dict[int, ArcCacheEntry] = field(default_factory=dict)
    artifact_keys: dict[str, str] = field(default_factory=dict)  # version -> artifact fp
    valid: bool = True
    invalid_reasons: list[str] = field(default_factory=list)


class IncrementalCache:
    def __init__(self, model_fp: str, mapping_fp: str, version: str):
        self.model_fp = model_fp
        self.mapping_fp = mapping_fp
        self.version = version
        self.entries: dict[str, SentenceCacheEntry] = {}

    def rebuild_sentence(self, sent: Sentence) -> SentenceCacheEntry:
        word_fp = word_sequence_fingerprint(sent)
        entry = SentenceCacheEntry(
            sent_id=sent.sent_id,
            sentence_key=sentence_fingerprint(sent, word_fp, self.model_fp, self.mapping_fp, self.version),
            word_fp=word_fp,
        )
        for arc in sent.arcs:
            arc.arc_cache_key = arc_fingerprint(arc, self.model_fp, self.mapping_fp, self.version)
            entry.arcs[arc.word_idx] = ArcCacheEntry(
                arc_key=arc.arc_cache_key,
                referenced_words=sorted(set(arc.referenced_words)),
                relations=[arc.relation],
            )
        self.entries[sent.sent_id] = entry
        return entry

    def record_artifact(self, sent_id: str, version: str, serialized: bytes, upstream: list[str]) -> str:
        fp = artifact_fingerprint(serialized, upstream)
        self.entries[sent_id].artifact_keys[version] = fp
        return fp

    def invalidate_sample_replaced(self, corpus: Corpus, changed_sent_ids: set[str]) -> list[str]:
        """句子样本被替换：该句全部弧及其序列化产物失效。"""
        report: list[str] = []
        for sid in sorted(changed_sent_ids):
            entry = self.entries.get(sid)
            if entry:
                entry.valid = False
                entry.invalid_reasons.append("sample_replaced")
                report.append(sid)
        return report

    def invalidate_mapping_change(
        self, corpus: Corpus, changed_relations: set[str]
    ) -> list[dict[str, Any]]:
        """映射表增删关系名：仅引用受影响关系的弧失效。"""
        report: list[dict[str, Any]] = []
        for sid in sorted(self.entries):
            entry = self.entries[sid]
            sent = corpus.sentences[sid]
            hit_arcs: list[int] = []
            for arc in sent.arcs:
                if arc.relation in changed_relations:
                    hit_arcs.append(arc.word_idx)
            if hit_arcs:
                entry.valid = False
                entry.invalid_reasons.append("mapping_changed")
                report.append({"sent_id": sid, "invalid_arcs": sorted(hit_arcs)})
        return report

    def invalidate_model_change(self, corpus: Corpus, new_model_fp: str) -> list[str]:
        """模型参数变更：所有弧派生键变化，全部句子失效（指纹不匹配驱动）。"""
        self.model_fp = new_model_fp
        report: list[str] = []
        for sid in sorted(self.entries):
            entry = self.entries[sid]
            entry.valid = False
            entry.invalid_reasons.append("model_changed")
            report.append(sid)
        return report

    def invalidate_word_change(
        self, corpus: Corpus, affected_words: dict[str, set[int]]
    ) -> list[dict[str, Any]]:
        """精确到词：仅引用受影响词集合的弧失效。"""
        report: list[dict[str, Any]] = []
        for sid, words in sorted(affected_words.items()):
            entry = self.entries.get(sid)
            if not entry:
                continue
            hit = [
                idx
                for idx, arc_entry in entry.arcs.items()
                if set(arc_entry.referenced_words) & words
            ]
            if hit:
                entry.valid = False
                entry.invalid_reasons.append("word_changed")
                report.append({"sent_id": sid, "invalid_arcs": sorted(hit)})
        return report

    def reusable_sentences(self) -> list[str]:
        """未失效句子：复用已持久化产物，禁止重新解析。"""
        return sorted(sid for sid, entry in self.entries.items() if entry.valid)

    def invalidation_listing(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for sid in sorted(self.entries):
            entry = self.entries[sid]
            out.append(
                {
                    "sent_id": sid,
                    "valid": entry.valid,
                    "reasons": list(entry.invalid_reasons),
                    "reuse_artifact": entry.valid,
                }
            )
        return out
