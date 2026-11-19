"""统一中间表示（句子 / 词 / 依存弧三级结构）。

编号约定：统一中间表示一律采用 1 起始词序号，头词序号为 0 表示根（无头部）。
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

MISSING_HEAD = 0
HEAD_FILLER = "*NA*"


@dataclass
class Word:
    idx: int  # 1 起始词序号
    form: str
    pos: str

    def fingerprint_parts(self) -> list[str]:
        return [str(self.idx), self.form, self.pos]


@dataclass
class Arc:
    """单个词对应的依存弧。

    head 为统一 1 起始头词序号；0 表示根（缺失头部）。
    relation 为统一命名关系（v2 风格名称）。
    score 为 0~1 浮点得分。
    """

    word_idx: int
    head: int | None  # None / 0 均表示根缺失，规范化时统一为 0
    relation: str
    score: float
    precision: str = "exact"
    source_version: str = "v2"
    raw: dict[str, Any] = field(default_factory=dict)
    referenced_words: list[int] = field(default_factory=list)
    arc_cache_key: str = ""

    def normalized_head(self) -> int:
        return 0 if self.head is None else int(self.head)


@dataclass
class Sentence:
    sent_id: str
    words: list[Word]
    arcs: list[Arc]
    source_version: str = "v2"
    needs_reparse: bool = False
    raw_meta: dict[str, Any] = field(default_factory=dict)
    sentence_cache_key: str = ""

    def word_by_idx(self, idx: int) -> Word:
        return self.words[idx - 1]

    def arc_by_word(self, idx: int) -> Arc:
        return self.arcs[idx - 1]

    def word_count(self) -> int:
        return len(self.words)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Corpus:
    sentences: dict[str, Sentence] = field(default_factory=dict)

    def sorted_ids(self) -> list[str]:
        return sorted(self.sentences.keys())

    def get(self, sent_id: str) -> Sentence:
        return self.sentences[sent_id]
