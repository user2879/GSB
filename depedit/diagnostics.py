"""跨版本一致性校验：v1/v2 序列化-反序列化往返对比。"""
from __future__ import annotations

from collections import Counter
from typing import Any

from .codecs import (
    v1_parse_document,
    v1_serialize_document,
    v2_parse_document,
    v2_serialize_document,
)
from .config import AppConfig
from .models import Arc, Sentence


def roundtrip_compare(
    sentences: list[Sentence],
    cfg: AppConfig,
    mapping: dict[str, Any],
) -> dict[str, Any]:
    head_hist: Counter[int] = Counter()
    max_score_diff = 0.0
    max_score_sent: str | None = None
    relation_failures: list[dict[str, str]] = []
    per_sentence: list[dict[str, Any]] = []

    for sent in sorted(sentences, key=lambda s: s.sent_id):
        # v1 往返
        v1_text, v1_losses = v1_serialize_document([sent], cfg, mapping["v2_to_v1"])
        v1_back = v1_parse_document(v1_text, cfg, mapping["v1_to_v2"], strict_relation=False)[0]
        # v2 往返
        v2_doc, _ = v2_serialize_document([sent], cfg)
        v2_back = v2_parse_document(v2_doc, cfg, mapping["named_relations"], strict_relation=False)[0]

        arc_count_match = len(sent.arcs) == len(v1_back.arcs) == len(v2_back.arcs)
        for orig, a1, a2 in zip(
            sorted(sent.arcs, key=lambda a: a.word_idx),
            sorted(v1_back.arcs, key=lambda a: a.word_idx),
            sorted(v2_back.arcs, key=lambda a: a.word_idx),
        ):
            head_diff = abs(orig.normalized_head() - a1.normalized_head())
            head_hist[head_diff] += 1
            head_diff2 = abs(orig.normalized_head() - a2.normalized_head())
            head_hist[head_diff2] += 1

            diff = abs(orig.score - a1.score)
            if diff > max_score_diff:
                max_score_diff = diff
                max_score_sent = sent.sent_id
            diff2 = abs(orig.score - a2.score)
            if diff2 > max_score_diff:
                max_score_diff = diff2
                max_score_sent = sent.sent_id

            if a1.relation != orig.relation:
                relation_failures.append(
                    {
                        "sent_id": sent.sent_id,
                        "word_idx": orig.word_idx,
                        "original": orig.relation,
                        "restored": a1.relation,
                        "channel": "v1",
                    }
                )
            if a2.relation != orig.relation:
                relation_failures.append(
                    {
                        "sent_id": sent.sent_id,
                        "word_idx": orig.word_idx,
                        "original": orig.relation,
                        "restored": a2.relation,
                        "channel": "v2",
                    }
                )
        per_sentence.append(
            {
                "sent_id": sent.sent_id,
                "arc_count": len(sent.arcs),
                "arc_count_match": arc_count_match,
                "v1_loss_count": len(v1_losses),
            }
        )

    return {
        "head_deviation_histogram": {str(k): head_hist[k] for k in sorted(head_hist)},
        "max_score_deviation": max_score_diff,
        "max_score_sentence": max_score_sent,
        "relation_restore_failures": relation_failures,
        "per_sentence": per_sentence,
    }
