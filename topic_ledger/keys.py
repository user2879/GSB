from __future__ import annotations

from typing import Any

from .canonical import fingerprint, sha256_text


def document_fingerprint(document: dict[str, Any]) -> str:
    return fingerprint(
        {
            "doc_id": document["doc_id"],
            "revision": document["revision"],
            "term_ids": list(document["term_ids"]),
        }
    )


def topic_parameter_fingerprint(config: dict[str, Any], vocab_size: int) -> str:
    return fingerprint(
        {
            "topic_count": config["topic_count"],
            "vocab_size": vocab_size,
            "version": config["topic_parameter_version"],
            "seed": config["topic_parameter_seed"],
            "fixed_point": config["fixed_point"],
        }
    )


def slice_cache_key(*, shard_id: str, document_refs: list[dict[str, Any]], term_counts: list[list[int]]) -> str:
    return "slice_" + sha256_text(
        fingerprint(
            {
                "stage": "term_counts",
                "shard_id": shard_id,
                "documents": sorted(document_refs, key=lambda item: item["doc_id"]),
                "term_counts": term_counts,
            }
        )
    )[:24]


def round_cache_key(
    *,
    slice_key: str,
    topic_fp: str,
    bit_width_tier: str,
    rounds: int,
    round_index: int,
    round_fingerprint: str,
) -> str:
    return "round_" + sha256_text(
        "|".join(
            [
                "sampling_round",
                slice_key,
                topic_fp,
                bit_width_tier,
                str(rounds),
                str(round_index),
                round_fingerprint,
            ]
        )
    )[:24]


def shard_cache_key(
    *,
    slice_key: str,
    topic_fp: str,
    bit_width_tier: str,
    rounds: int,
    final_fingerprint: str,
) -> str:
    return "shard_" + sha256_text(
        "|".join(
            [
                "shard_summary",
                slice_key,
                topic_fp,
                bit_width_tier,
                str(rounds),
                final_fingerprint,
            ]
        )
    )[:24]


def shard_product_key(
    *,
    slice_key: str,
    topic_fp: str,
    bit_width_tier: str,
    rounds: int,
) -> str:
    return "shardproduct_" + sha256_text(
        "|".join(["shard_product", slice_key, topic_fp, bit_width_tier, str(rounds)])
    )[:24]


def global_cache_key(*, shard_keys: list[str], topic_fp: str, bit_width_tier: str) -> str:
    return "global_" + sha256_text(
        "|".join(["global_distribution", topic_fp, bit_width_tier, *sorted(shard_keys)])
    )[:24]
