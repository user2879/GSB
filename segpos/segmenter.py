"""基于词典的正向最大匹配分词。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

from .models import Dictionary, Token


@dataclass
class Segment:
    start: int
    end: int
    text: str
    hit: bool
    match_basis: str
    boundary_confidence: float


def _margin_confidence(entries) -> float:
    """同一起点所有词典前缀的词频 softmax 中，被选最长词的占比。"""

    total = sum(entry.freq for entry in entries)
    chosen = entries[0]
    return float(chosen.freq) / float(total) if total else 0.0


def maximum_match(text: str, dictionary: Dictionary) -> List[Segment]:
    """正向最大匹配；同长按词频降序、词形字典序升序决断。"""

    segments: List[Segment] = []
    position = 0
    text_len = len(text)
    while position < text_len:
        candidates = dictionary.prefix_entries(text, position)
        if candidates:
            chosen = candidates[0]
            end = position + len(chosen.word)
            confidence = _margin_confidence(candidates)
            basis = (
                f"dict_max_match:len={len(chosen.word)};freq={chosen.freq};"
                f"prefix_candidates={len(candidates)}"
            )
            segments.append(
                Segment(
                    start=position,
                    end=end,
                    text=chosen.word,
                    hit=True,
                    match_basis=basis,
                    boundary_confidence=confidence,
                )
            )
            position = end
        else:
            segments.append(
                Segment(
                    start=position,
                    end=position + 1,
                    text=text[position],
                    hit=False,
                    match_basis="oov_single_char",
                    boundary_confidence=math.nan,
                )
            )
            position += 1
    return segments


def group_oov_runs(segments: List[Segment]) -> List[tuple[int, int]]:
    """返回连续未登录单字片段的 (起索引, 止索引) 区间。"""

    runs: List[tuple[int, int]] = []
    index = 0
    while index < len(segments):
        if not segments[index].hit:
            start = index
            while index < len(segments) and not segments[index].hit:
                index += 1
            runs.append((start, index))
        else:
            index += 1
    return runs


def segments_to_tokens(segments: List[Segment]) -> List[Token]:
    return [
        Token(
            start=segment.start,
            end=segment.end,
            text=segment.text,
            match_basis=segment.match_basis,
            hit=segment.hit,
            boundary_confidence=(
                0.0 if math.isnan(segment.boundary_confidence)
                else segment.boundary_confidence
            ),
        )
        for segment in segments
    ]
