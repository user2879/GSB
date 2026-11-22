"""核心数据模型与确定性序列化。"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

from .fingerprint import fingerprint

ALLOWED_FIXED_BITS = (8, 16, 32)
ALLOWED_QUANT_LEVELS = (256, 1024, 65536)
MAX_CONCURRENCY = 16


@dataclass(frozen=True)
class Config:
    concurrency: int
    first_response_budget_ms: float
    snapshot_commit_interval_commands: int
    low_confidence_threshold: float
    fixed_point_bits: int
    quantization_levels: int
    smoothing: float
    oov_rule: str
    oov_max_length: int
    simulated_task_delay_ms: float = 0.0

    def param_fingerprint(self) -> str:
        return fingerprint(
            {
                "low_confidence_threshold": self.low_confidence_threshold,
                "fixed_point_bits": self.fixed_point_bits,
                "quantization_levels": self.quantization_levels,
                "smoothing": self.smoothing,
            },
            prefix="config",
        )

    def oov_param_fingerprint(self) -> str:
        return fingerprint(
            {"oov_rule": self.oov_rule, "oov_max_length": self.oov_max_length},
            prefix="oov-config",
        )


@dataclass
class ProbTable:
    tags: list[str]
    start_logp: dict[str, float]
    transition_logp: dict[str, dict[str, float]]
    emission_logp: dict[str, dict[str, float]]
    smoothing: float

    def table_fingerprint(self) -> str:
        return fingerprint(
            {
                "tags": self.tags,
                "start": self.start_logp,
                "transition": self.transition_logp,
                "emission": self.emission_logp,
                "smoothing": self.smoothing,
            },
            prefix="prob-table",
        )

    @property
    def log_smoothing(self) -> float:
        return math.log(self.smoothing)

    def emission(self, tag: str, word: str) -> tuple[float, bool]:
        """返回 (对数发射概率, 是否为平滑补齐)。"""
        table = self.emission_logp.get(word)
        if table is not None and tag in table:
            return table[tag], False
        return self.log_smoothing, True


@dataclass
class LexiconEntry:
    word: str
    freq: float
    tag: str
    sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"word": self.word, "freq": self.freq, "tag": self.tag, "sources": list(self.sources)}


@dataclass
class Lexicon:
    entries: dict[str, LexiconEntry]

    def snapshot_fingerprint(self) -> str:
        payload = [
            {"word": word, "freq": self.entries[word].freq, "tag": self.entries[word].tag}
            for word in sorted(self.entries)
        ]
        return fingerprint(payload, prefix="lexicon")


@dataclass
class Sentence:
    sid: str
    text: str
    source_line: int


@dataclass
class Token:
    text: str
    start: int
    end: int
    source: str  # lexicon | single | oov
    tag: str | None = None
    freq: float | None = None
    match_len: int | None = None
    confidence: float | None = None
    confidence_fixed: int | None = None
    path_score: float | None = None
    smoothed: bool = False
    segment_basis: str | None = None
    fingerprint: str | None = None

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "source": self.source,
            "tag": self.tag,
            "freq": self.freq,
            "match_len": self.match_len,
            "confidence": self.confidence,
            "confidence_fixed": self.confidence_fixed,
            "path_score": self.path_score,
            "smoothed": self.smoothed,
            "segment_basis": self.segment_basis,
            "fingerprint": self.fingerprint,
        }
