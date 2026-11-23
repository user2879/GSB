"""Deterministic character feature hashing."""

from __future__ import annotations

from typing import Any

from .constants import FEATURE_ALGORITHM
from .models import CorpusSample, VocabEntry
from .util import digest, sha256_int


def _covering_entries(
    sample: CorpusSample,
    vocabulary: dict[str, VocabEntry],
) -> tuple[dict[int, list[VocabEntry]], list[VocabEntry]]:
    text = sample.text
    covering: dict[int, list[VocabEntry]] = {index: [] for index in range(len(text))}
    used: dict[str, VocabEntry] = {}
    for entry in sorted(vocabulary.values(), key=lambda item: (item.text, item.entry_id)):
        if not entry.text:
            continue
        start = text.find(entry.text)
        while start >= 0:
            for index in range(start, start + len(entry.text)):
                covering[index].append(entry)
            used[entry.entry_id] = entry
            start = text.find(entry.text, start + 1)
    return covering, [used[key] for key in sorted(used)]


def _relative_bucket(position: int, length: int) -> str:
    if length <= 1:
        return "single"
    if position == 0:
        return "start"
    if position == length - 1:
        return "end"
    return "middle"


def matching_vocabulary_texts(
    sample: CorpusSample,
    vocabulary: dict[str, VocabEntry],
) -> list[str]:
    return sorted({entry.text for entry in _covering_entries(sample, vocabulary)[1]})


def feature_key(sample: CorpusSample, vocabulary: dict[str, VocabEntry], feature_dim: int, hash_buckets: int) -> str:
    return digest(
        {
            "stage": "feature",
            "algorithm": FEATURE_ALGORITHM,
            "sample_id": sample.sample_id,
            "chars": list(sample.chars),
            "vocab_texts": matching_vocabulary_texts(sample, vocabulary),
            "feature_dim": feature_dim,
            "hash_buckets": hash_buckets,
        }
    )


def build_features(
    sample: CorpusSample,
    vocabulary: dict[str, VocabEntry],
    feature_dim: int,
    hash_buckets: int,
) -> dict[str, Any]:
    covering, used_entries = _covering_entries(sample, vocabulary)
    positions: list[dict[str, Any]] = []
    for position, char in enumerate(sample.chars):
        entries = covering.get(position, [])
        hit_length = max((len(entry.text) for entry in entries), default=0)
        previous_char = sample.chars[position - 1] if position > 0 else "@BOS"
        next_char = sample.chars[position + 1] if position + 1 < len(sample.chars) else "@EOS"
        relative = _relative_bucket(position, len(sample.chars))
        channel_tokens = [
            ("cur", f"cur:{char}", char),
            ("prev", f"prev:{previous_char}", previous_char),
            ("next", f"next:{next_char}", next_char),
            ("rel", f"rel:{relative}", f"@rel:{relative}"),
            ("hit", f"hit:{hit_length}", f"@hit:{hit_length}"),
        ]
        features: list[dict[str, Any]] = []
        for channel, token, lookup_key in channel_tokens:
            features.append(
                {
                    "channel": channel,
                    "token": token,
                    "lookup_key": lookup_key,
                    "feature_index": sha256_int("fi|" + token, offset=0, size=8)
                    % feature_dim,
                    "bucket": sha256_int("fb|" + token, offset=0, size=8)
                    % hash_buckets,
                }
            )
        positions.append(
            {
                "position": position,
                "char": char,
                "features": features,
                "vocab_ids": [entry.entry_id for entry in entries],
                "vocab_texts": sorted({entry.text for entry in entries}),
                "hit_length": hit_length,
            }
        )
    payload = {
        "sample_id": sample.sample_id,
        "chars": list(sample.chars),
        "positions": positions,
        "vocab_texts": sorted({entry.text for entry in used_entries}),
        "vocab_entry_ids": [entry.entry_id for entry in used_entries],
        "feature_dim": feature_dim,
        "hash_buckets": hash_buckets,
        "algorithm": FEATURE_ALGORITHM,
    }
    return payload


def feature_output_fingerprint(payload: dict[str, Any]) -> str:
    # References are diagnostic metadata. Output is the exact computation.
    return digest(
        {
            "sample_id": payload["sample_id"],
            "chars": payload["chars"],
            "positions": payload["positions"],
            "feature_dim": payload["feature_dim"],
            "hash_buckets": payload["hash_buckets"],
            "algorithm": payload["algorithm"],
        }
    )


def refresh_feature_refs(
    payload: dict[str, Any],
    vocabulary: dict[str, VocabEntry],
) -> dict[str, Any]:
    sample_text = "".join(payload.get("chars", []))
    refs = []
    for entry in sorted(vocabulary.values(), key=lambda item: (item.text, item.entry_id)):
        if entry.text and entry.text in sample_text:
            refs.append(entry.entry_id)
    return {
        "sample_id": payload.get("sample_id"),
        "vocab_entry_ids": refs,
        "vocab_texts": payload.get("vocab_texts", []),
    }

