"""跨模态对齐打分、三通道召回与确定性融合。

全部对外得分均为定点整数：
- 视觉/文本特征量化后做固定顺序定点内积；
- 稳定归一化先减去向量最大值再 softmax；
- 融合温度以定点整数近似（round(T*scale)），保证档位切换真实改变整数与排序。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import Config
from .features import TextFeatures, VisualFeatures, stable_normalize
from .fingerprint import digest_many, digest_obj
from .fixedpoint import QuantSpec, fixed_weight

LEXICAL = "lexical"
VISUAL = "visual"
HYBRID = "hybrid"
CHANNEL_ORDER = (LEXICAL, VISUAL, HYBRID)


@dataclass
class AlignmentResult:
    query: str
    candidate: str
    similarity_f: float
    alignment_q: int
    norm_q: int
    lexical_q: int
    used_tiles: list[tuple[int, int]]
    tile_similarities: list[dict[str, Any]]
    fingerprint: str


def select_tile_keys(grid: tuple[int, int], tile_range: str) -> list[tuple[int, int]]:
    rows, cols = grid
    keys = [(r, c) for r in range(rows) for c in range(cols)]
    if tile_range == "full":
        return keys
    # center_half：取行列索引居中的一半（偶数尺寸时半开区间，确定性）
    r_half, c_half = rows // 2, cols // 2
    r0, r1 = r_half // 2, rows - r_half // 2
    c0, c1 = c_half // 2, cols - c_half // 2
    return [(r, c) for (r, c) in keys if r0 <= r < r1 and c0 <= c < c1]


def _weighted_tile_similarity(visual_q: np.ndarray, text_q: np.ndarray,
                              raw_stats: np.ndarray, weights_q: dict[str, int],
                              spec: QuantSpec, include_edge: bool) -> tuple[int, float]:
    """逐块对齐：维度升序内积，颜色/梯度/(边缘) 按定点权重加权。"""

    dot_q = int((visual_q * text_q).sum() // spec.scale)
    stat = raw_stats
    if include_edge:
        color_f = float(stat[0:3].mean())
        gradient_f = float(stat[3])
        edge_f = float(stat[4])
    else:
        color_f = float(stat[0:3].mean())
        gradient_f = float(stat[3])
        edge_f = 0.0
    weight_sum = weights_q["color"] + weights_q["gradient"] \
        + (weights_q["edge"] if include_edge else 0)
    factors_f = (
        color_f * weights_q["color"]
        + gradient_f * weights_q["gradient"]
        + edge_f * (weights_q["edge"] if include_edge else 0)
    )
    factor_q = int(round((factors_f / max(1, weight_sum)) * spec.scale))
    score_q = (dot_q * factor_q) // spec.scale
    score_f = (dot_q / spec.scale) * (factors_f / max(1, weight_sum))
    return score_q, score_f


def align_pair(query_text: TextFeatures, candidate_visual: VisualFeatures,
               config: Config, spec: QuantSpec, tile_range: str) -> AlignmentResult:
    include_edge = candidate_visual.precision == "full"
    weights_q = {key: fixed_weight(value, spec) for key, value in config.weights.items()}
    keys = select_tile_keys(candidate_visual.grid, tile_range)
    index_of = {key: index for index, key in enumerate(candidate_visual.tile_keys)}

    visual_q = candidate_visual.q()
    text_q = query_text.q()
    # 文本侧：对多个断言向量做稳定归一化后按维度升序合成一个查询向量（定点）
    text_norm_f = stable_normalize(query_text.projected.mean(axis=0))
    text_query_q = np.rint(text_norm_f * spec.scale).astype(np.int64)

    tile_records: list[dict[str, Any]] = []
    total_q = 0
    total_f = 0.0
    for key in keys:  # 行列号升序（tile_keys 本身升序，过滤保持顺序）
        tile_index = index_of[key]
        score_q, score_f = _weighted_tile_similarity(
            visual_q[tile_index], text_query_q,
            candidate_visual.raw_stats[tile_index], weights_q, spec, include_edge
        )
        total_q += score_q
        total_f += score_f
        tile_records.append({"tile": list(key), "score_q": int(score_q),
                             "score_f": round(score_f, 9)})

    norm_q = int(round(stable_normalize(np.array([total_f], dtype=np.float64))[0] * spec.scale))
    fingerprint = digest_many(
        "alignment", query_text.feature_fingerprint, candidate_visual.feature_fingerprint,
        digest_obj(config.weights), tile_range, spec.key(), total_q
    )
    return AlignmentResult(
        query=query_text.sample,
        candidate=candidate_visual.sample,
        similarity_f=round(total_f, 9),
        alignment_q=int(total_q),
        norm_q=norm_q,
        lexical_q=0,
        used_tiles=[list(key) for key in keys],
        tile_similarities=tile_records,
        fingerprint=fingerprint,
    )


def lexical_score(query_text: TextFeatures, candidate, spec: QuantSpec) -> int:
    """词面通道：实体重合 + 谓词/对象词表命中，转定点整数。"""

    score_f = 0.0
    query_objects = [ref["object"] for ref in query_text.assertion_refs if not ref["padded"]]
    query_predicates = [ref["predicate"] for ref in query_text.assertion_refs if not ref["padded"]]
    if candidate.entity in query_objects:
        score_f += 1.0
    candidate_predicates = [item["predicate"] for item in candidate.assertions]
    candidate_objects = [item["object"] for item in candidate.assertions]
    score_f += 0.25 * len(set(query_predicates) & set(candidate_predicates))
    score_f += 0.25 * len(set(query_objects) & set(candidate_objects))
    score_f += 0.5 if query_text.sample == candidate.name else 0.0
    return int(round(score_f * spec.scale))


@dataclass
class CandidateRecord:
    sample: str
    fused_q: int
    alignment_q: int
    norm_q: int
    channel_contrib: dict[str, int]
    lexical_q: int
    visual_q: int
    hybrid_q: int

    def tie_break_key(self) -> tuple:
        contrib_sum = sum(self.channel_contrib.values())
        return (-self.fused_q, -self.alignment_q, -contrib_sum, self.sample)


def fuse_candidate(alignment: AlignmentResult, lexical_q: int,
                   config: Config, spec: QuantSpec,
                   active_channels: list[str]) -> CandidateRecord:
    visual_q = alignment.norm_q
    hybrid_q = int(round(((lexical_q + visual_q) / 2.0)))
    raw = {LEXICAL: lexical_q, VISUAL: visual_q, HYBRID: hybrid_q}
    temp_q = max(1, int(round(config.fusion_temperature * spec.scale)))
    channel_contrib: dict[str, int] = {}
    fused_q = 0
    for channel in CHANNEL_ORDER:
        if channel not in active_channels:
            channel_contrib[channel] = 0
            continue
        weight_q = fixed_weight(config.channel_weights[channel], spec)
        # 温度作用：整数除以定点温度，再乘通道权重
        contribution = (raw[channel] * spec.scale // temp_q) * weight_q // spec.scale
        channel_contrib[channel] = int(contribution)
        fused_q += contribution
    return CandidateRecord(
        sample=alignment.candidate,
        fused_q=int(fused_q),
        alignment_q=alignment.alignment_q,
        norm_q=alignment.norm_q,
        channel_contrib=channel_contrib,
        lexical_q=lexical_q,
        visual_q=visual_q,
        hybrid_q=hybrid_q,
    )


@dataclass
class QueryResult:
    query: str
    candidates: list[CandidateRecord]
    fingerprint: str
    channels: list[str]

    def to_dict(self) -> dict[str, Any]:
        ordered = sorted(self.candidates, key=lambda item: item.tie_break_key())
        return {
            "query": self.query,
            "channels": list(self.channels),
            "fingerprint": self.fingerprint,
            "candidate_order": [item.sample for item in ordered],
            "candidates": [
                {
                    "sample": item.sample,
                    "fused_q": item.fused_q,
                    "alignment_q": item.alignment_q,
                    "norm_q": item.norm_q,
                    "lexical_q": item.lexical_q,
                    "visual_q": item.visual_q,
                    "hybrid_q": item.hybrid_q,
                    "channel_contrib": item.channel_contrib,
                }
                for item in ordered
            ],
        }


def run_query(query_name: str, visual_store: dict[str, VisualFeatures],
              text_store: dict[str, TextFeatures], sample_store: dict,
              config: Config, spec: QuantSpec, decision,
              candidate_names: list[str]) -> QueryResult:
    query_text = text_store[query_name]
    records: list[CandidateRecord] = []
    alignments: list[AlignmentResult] = []
    for candidate_name in sorted(candidate_names):
        candidate_visual = visual_store[candidate_name]
        alignment = align_pair(query_text, candidate_visual, config, spec,
                               decision.tile_range)
        lexical_q = lexical_score(query_text, sample_store[candidate_name], spec)
        alignment.lexical_q = lexical_q
        alignments.append(alignment)
        records.append(fuse_candidate(alignment, lexical_q, config, spec,
                                      decision.channels))
    ordered = sorted(records, key=lambda item: item.tie_break_key())
    capped = ordered[: decision.candidate_cap]
    fingerprint = digest_many(
        "recall", query_name, digest_obj([item.fingerprint for item in alignments]),
        digest_obj(config.channel_weights), config.fusion_temperature,
        digest_obj(decision.channels), decision.candidate_cap, spec.key()
    )
    alignment_details = {
        item.candidate: {
            "alignment_q": item.alignment_q,
            "norm_q": item.norm_q,
            "lexical_q": item.lexical_q,
            "similarity_f": item.similarity_f,
            "used_tiles": item.used_tiles,
            "tile_similarities": item.tile_similarities,
            "fingerprint": item.fingerprint,
        }
        for item in alignments
    }
    result = QueryResult(query=query_name, candidates=capped,
                         fingerprint=fingerprint, channels=list(decision.channels))
    result.tile_trace = {
        item.candidate: item.used_tiles for item in alignments
    }
    result.alignment_trace = {
        item.candidate: item.tile_similarities for item in alignments
    }
    result.alignment_details = alignment_details
    return result
