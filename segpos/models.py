"""词典、概率表与句子记录等不可变数据模型。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np

from .errors import ConfigError
from .fingerprints import canonical_hash, sha256_hex


@dataclass(frozen=True)
class DictEntry:
    word: str
    freq: int
    pos: str


@dataclass
class Token:
    """单个词的分词与标注结果（标注阶段回填词性与路径得分）。"""

    start: int
    end: int
    text: str
    match_basis: str
    hit: bool
    boundary_confidence: float
    pos: str | None = None
    local_path_score: float | None = None

    def to_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "match_basis": self.match_basis,
            "hit": self.hit,
            "boundary_confidence": round(float(self.boundary_confidence), 12),
            "pos": self.pos,
            "local_path_score": (
                None
                if self.local_path_score is None
                else round(float(self.local_path_score), 12)
            ),
        }


@dataclass
class SentenceRecord:
    sentence_id: str
    text: str
    tokens: List[Token] = field(default_factory=list)
    cache_key: str = ""
    reused: bool = False
    sentence_logprob: float = 0.0
    smoothing_hits: int = 0
    low_confidence_words: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "sentence_id": self.sentence_id,
            "text": self.text,
            "cache_key": self.cache_key,
            "reused": self.reused,
            "sentence_logprob": round(float(self.sentence_logprob), 12),
            "smoothing_hits": self.smoothing_hits,
            "tokens": [token.to_dict() for token in self.tokens],
            "pos_sequence": [token.pos for token in self.tokens],
            "low_confidence_words": self.low_confidence_words,
        }


class Dictionary:
    """不可变词典快照，提供前缀检索与最长词长度。"""

    def __init__(self, entries: List[DictEntry], allowed_pos: set[str] | None = None):
        self._entries: Dict[str, DictEntry] = {}
        max_len = 0
        for entry in entries:
            if not entry.word:
                raise ConfigError("词典存在空词形")
            if not isinstance(entry.freq, int) or entry.freq <= 0:
                raise ConfigError(f"词条 {entry.word} 词频必须为正整数")
            if allowed_pos is not None and entry.pos not in allowed_pos:
                raise ConfigError(
                    f"词条 {entry.word} 的词性 {entry.pos} 不在预定义词性集合内"
                )
            if entry.word in self._entries:
                raise ConfigError(f"词典存在重复词条: {entry.word}")
            self._entries[entry.word] = entry
            max_len = max(max_len, len(entry.word))
        self._max_len = max_len
        self._hash = canonical_hash(
            [
                {"word": e.word, "freq": e.freq, "pos": e.pos}
                for e in sorted(self._entries.values(), key=lambda item: item.word)
            ]
        )

    @classmethod
    def load(cls, path: str | Path, allowed_pos: set[str] | None = None) -> "Dictionary":
        entries: List[DictEntry] = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_no, raw in enumerate(handle, start=1):
                line = raw.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 3:
                    raise ConfigError(
                        f"词典第 {line_no} 行格式应为 '词 词频 词性': {raw.rstrip()}"
                    )
                word, freq_text, pos = parts
                try:
                    freq = int(freq_text)
                except ValueError as exc:
                    raise ConfigError(
                        f"词典第 {line_no} 行词频不是整数: {freq_text}"
                    ) from exc
                entries.append(DictEntry(word=word, freq=freq, pos=pos))
        return cls(entries, allowed_pos=allowed_pos)

    @property
    def hash(self) -> str:
        return self._hash

    @property
    def max_word_len(self) -> int:
        return self._max_len

    @property
    def words(self) -> List[str]:
        return sorted(self._entries)

    def get(self, word: str) -> DictEntry | None:
        return self._entries.get(word)

    def contains(self, word: str) -> bool:
        return word in self._entries

    def prefix_entries(self, text: str, start: int) -> List[DictEntry]:
        """返回 text[start:] 上所有词典前缀命中，按长度降序排列。"""

        upper = min(len(text), start + self._max_len)
        hits: List[DictEntry] = []
        for end in range(start + 1, upper + 1):
            candidate = text[start:end]
            entry = self._entries.get(candidate)
            if entry is not None:
                hits.append(entry)
        hits.sort(key=lambda entry: (-len(entry.word), -entry.freq, entry.word))
        return hits

    def to_entries(self) -> List[DictEntry]:
        return [self._entries[word] for word in sorted(self._entries)]


class ProbabilityTable:
    """词性转移/发射概率表（自然对数域）。

    start_logprob: 句首词性先验 log P(pos|<s>)。
    trans_logprob: log P(next_pos|pos)。
    emit_logprob: log P(词形|pos)。
    char_emit_logprob: log P(单字|pos)，用于未登录词字符级 HMM。
    """

    def __init__(self, data: dict, version: int = 1):
        required = ("tags", "start_logprob", "trans_logprob", "emit_logprob")
        missing = [key for key in required if key not in data]
        if missing:
            raise ConfigError(f"概率表缺少字段: {', '.join(missing)}")
        tags = data["tags"]
        if not isinstance(tags, list) or not tags or len(set(tags)) != len(tags):
            raise ConfigError("概率表 tags 必须为非空且无重复的列表")
        self.tags: List[str] = list(tags)
        self.tag_index = {tag: idx for idx, tag in enumerate(self.tags)}
        self.start_logprob = self._validate_mapping(
            data["start_logprob"], self.tags, "start_logprob"
        )
        trans = data["trans_logprob"]
        if not isinstance(trans, dict):
            raise ConfigError("trans_logprob 必须为对象")
        self.trans_logprob: Dict[str, Dict[str, float]] = {
            head: self._validate_mapping(tail, self.tags, f"trans_logprob.{head}")
            for head, tail in trans.items()
        }
        self.emit_logprob = self._validate_emission(data["emit_logprob"], "emit_logprob")
        self.char_emit_logprob = self._validate_emission(
            data.get("char_emit_logprob", {}), "char_emit_logprob", allow_empty=True
        )
        self._content_hash = sha256_hex(
            json.dumps(
                {
                    "tags": self.tags,
                    "start_logprob": self.start_logprob,
                    "trans_logprob": self.trans_logprob,
                    "emit_logprob": self.emit_logprob,
                    "char_emit_logprob": self.char_emit_logprob,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        self.version = int(version)
        self.start_vec = np.array(
            [self.start_logprob[tag] for tag in self.tags], dtype=np.float64
        )

    @staticmethod
    def _validate_mapping(mapping, tags, label) -> Dict[str, float]:
        if not isinstance(mapping, dict):
            raise ConfigError(f"{label} 必须为对象")
        result: Dict[str, float] = {}
        for tag in tags:
            if tag not in mapping:
                raise ConfigError(f"{label} 缺少词性 {tag}")
            value = mapping[tag]
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ConfigError(f"{label}.{tag} 必须是数值（自然对数）")
            result[tag] = float(value)
        return result

    @staticmethod
    def _validate_emission(mapping, label, allow_empty=False) -> Dict[str, Dict[str, float]]:
        if not isinstance(mapping, dict):
            raise ConfigError(f"{label} 必须为对象")
        result: Dict[str, Dict[str, float]] = {}
        for word, pos_map in mapping.items():
            if not isinstance(pos_map, dict):
                raise ConfigError(f"{label}.{word} 必须为对象")
            result[word] = {
                pos: float(value)
                for pos, value in pos_map.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
        if not allow_empty and not result:
            raise ConfigError(f"{label} 不能为空")
        return result

    @classmethod
    def load(cls, path: str | Path, version: int = 1) -> "ProbabilityTable":
        raw_path = Path(path)
        try:
            data = json.loads(raw_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ConfigError(f"概率表不存在: {raw_path}") from exc
        except json.JSONDecodeError as exc:
            raise ConfigError(f"概率表不是合法 JSON: {raw_path}") from exc
        return cls(data, version=version)

    @property
    def hash(self) -> str:
        return self._content_hash

    def emit(self, word: str, pos: str, smoothing: float) -> tuple[float, bool]:
        pos_map = self.emit_logprob.get(word)
        if pos_map is not None and pos in pos_map:
            return pos_map[pos], False
        return smoothing, True

    def char_emit(self, char: str, pos: str, smoothing: float) -> tuple[float, bool]:
        pos_map = self.char_emit_logprob.get(char)
        if pos_map is not None and pos in pos_map:
            return pos_map[pos], False
        return smoothing, True

    def trans(self, head_pos: str, tail_pos: str, smoothing: float) -> tuple[float, bool]:
        tails = self.trans_logprob.get(head_pos)
        if tails is not None and tail_pos in tails:
            return tails[tail_pos], False
        return smoothing, True

    def start(self, pos: str) -> float:
        return self.start_logprob[pos]
