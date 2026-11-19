"""未登录词 HMM 识别与词性标注维特比解码的组合阶段。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .hmm import viterbi_decode
from .models import Dictionary, ProbabilityTable, Token
from .segmenter import Segment, group_oov_runs, maximum_match


@dataclass
class AnalysisResult:
    tokens: List[Token]
    hit_words: List[str]
    sentence_logprob: float
    smoothing_hits: int


def _char_viterbi(chars: List[str], table: ProbabilityTable, smoothing: float):
    start_scores = table.start_vec.copy()

    def emission(observation, tag_idx):
        return table.char_emit(observation, table.tags[tag_idx], smoothing)

    def transition(prev_idx, tag_idx):
        return table.trans(table.tags[prev_idx], table.tags[tag_idx], smoothing)

    return viterbi_decode(chars, table.tags, start_scores, transition, emission)


def _word_viterbi(tokens: List[Token], table: ProbabilityTable, smoothing: float):
    observations = [token.text for token in tokens]
    start_scores = table.start_vec.copy()

    def emission(observation, tag_idx):
        return table.emit(observation, table.tags[tag_idx], smoothing)

    def transition(prev_idx, tag_idx):
        return table.trans(table.tags[prev_idx], table.tags[tag_idx], smoothing)

    return viterbi_decode(observations, table.tags, start_scores, transition, emission)


def _build_oov_tokens(
    run_chars: List[Segment], table: ProbabilityTable, smoothing: float
) -> List[Token]:
    chars = [segment.text for segment in run_chars]
    result = _char_viterbi(chars, table, smoothing)
    tokens: List[Token] = []
    index = 0
    while index < len(chars):
        tag = result.tags[index]
        end = index + 1
        while end < len(chars) and result.tags[end] == tag:
            end += 1
        start_pos = run_chars[index].start
        end_pos = run_chars[end - 1].end
        confidence = float(np.mean(result.margin_confidence[index:end]))
        tokens.append(
            Token(
                start=start_pos,
                end=end_pos,
                text="".join(chars[index:end]),
                match_basis=f"hmm_oov:pos={tag};chars={end - index}",
                hit=False,
                boundary_confidence=confidence,
            )
        )
        index = end
    return tokens, result.smoothing_hits


def analyze(
    text: str, dictionary: Dictionary, table: ProbabilityTable, smoothing: float
) -> AnalysisResult:
    """分词 -> 未登录词识别 -> 词性标注的单句完整流水线。"""

    segments = maximum_match(text, dictionary)
    oov_runs = group_oov_runs(segments)
    run_lookup = {start: end for start, end in oov_runs}

    tokens: List[Token] = []
    smoothing_hits = 0
    index = 0
    while index < len(segments):
        if index in run_lookup:
            run_end = run_lookup[index]
            oov_tokens, oov_smoothing = _build_oov_tokens(
                segments[index:run_end], table, smoothing
            )
            tokens.extend(oov_tokens)
            smoothing_hits += oov_smoothing
            index = run_end
        else:
            segment = segments[index]
            tokens.append(
                Token(
                    start=segment.start,
                    end=segment.end,
                    text=segment.text,
                    match_basis=segment.match_basis,
                    hit=True,
                    boundary_confidence=segment.boundary_confidence,
                )
            )
            index += 1

    tagging = _word_viterbi(tokens, table, smoothing)
    smoothing_hits += tagging.smoothing_hits
    for token, tag, score in zip(tokens, tagging.tags, tagging.deltas):
        token.pos = tag
        token.local_path_score = score

    hit_words = sorted({segment.text for segment in segments if segment.hit})
    return AnalysisResult(
        tokens=tokens,
        hit_words=hit_words,
        sentence_logprob=tagging.normalized_logprob,
        smoothing_hits=smoothing_hits,
    )


def segment_with_oov(
    text: str, dictionary: Dictionary, table: ProbabilityTable, smoothing: float
) -> tuple[List[Token], List[str], int]:
    """阶段一：最大匹配分词 + 连续单字 HMM 未登录词识别。"""

    segments = maximum_match(text, dictionary)
    oov_runs = group_oov_runs(segments)
    run_lookup = {start: end for start, end in oov_runs}

    tokens: List[Token] = []
    smoothing_hits = 0
    index = 0
    while index < len(segments):
        if index in run_lookup:
            run_end = run_lookup[index]
            oov_tokens, oov_smoothing = _build_oov_tokens(
                segments[index:run_end], table, smoothing
            )
            tokens.extend(oov_tokens)
            smoothing_hits += oov_smoothing
            index = run_end
        else:
            segment = segments[index]
            tokens.append(
                Token(
                    start=segment.start,
                    end=segment.end,
                    text=segment.text,
                    match_basis=segment.match_basis,
                    hit=True,
                    boundary_confidence=segment.boundary_confidence,
                )
            )
            index += 1
    hit_words = sorted({segment.text for segment in segments if segment.hit})
    return tokens, hit_words, smoothing_hits


def tag_tokens(
    tokens: List[Token], table: ProbabilityTable, smoothing: float
) -> tuple[List[Token], float, int]:
    """阶段二：词级词性维特比标注，回填词性与局部路径得分。"""

    tagging = _word_viterbi(tokens, table, smoothing)
    for token, tag, score in zip(tokens, tagging.tags, tagging.deltas):
        token.pos = tag
        token.local_path_score = score
    return tokens, tagging.normalized_logprob, tagging.smoothing_hits
