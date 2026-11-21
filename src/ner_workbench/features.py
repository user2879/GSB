from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from .errors import E_NONFINITE_SCORE, E_SAMPLE_QUOTA_EXCEEDED, WorkbenchError
from .fixedpoint import fixed_mul_div
from .params import ParameterPackage
from .types import CorpusSample


@dataclass(frozen=True)
class FeatureHit:
    key: str
    bucket: int
    vector: tuple[int, ...]


def stable_feature_bucket(key: str, salt: str, hash_buckets: int) -> int:
    payload = f"{salt}\x1f{key}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=16).digest()
    return int.from_bytes(digest, "big") % int(hash_buckets)


def relative_position_bucket(index: int, length: int) -> int:
    if length <= 1:
        return 0
    return min(3, (int(index) * 4) // int(length))


def vocab_hit_length(chars: Sequence[str], index: int, vocab: frozenset[str], max_length: int) -> int:
    limit = min(int(max_length), len(chars) - int(index))
    for length in range(limit, 0, -1):
        candidate = "".join(chars[index : index + length])
        if candidate in vocab:
            return length
    return 0


def _char_vector(params: ParameterPackage, char: str) -> tuple[int, ...]:
    vector = params.char_embeddings.get(char)
    if vector is not None:
        return vector
    unknown = params.char_embeddings.get("<UNK>")
    if unknown is not None:
        return unknown
    return tuple(0 for _ in range(params.feature_dim))


def feature_hits_for_position(
    sample: CorpusSample,
    index: int,
    params: ParameterPackage,
    max_vocab_length: int,
) -> tuple[FeatureHit, ...]:
    chars = sample.chars
    current = chars[index]
    previous = chars[index - 1] if index > 0 else "<BOS>"
    following = chars[index + 1] if index + 1 < len(chars) else "<EOS>"
    position_bucket = relative_position_bucket(index, len(chars))
    hit_length = vocab_hit_length(chars, index, params.vocab, max_vocab_length)

    specs: tuple[tuple[str, tuple[int, ...]], ...] = (
        (f"U:{current}", _char_vector(params, current)),
        (f"L:{previous}", _char_vector(params, previous)),
        (f"R:{following}", _char_vector(params, following)),
        (f"P:{position_bucket}", params.position_embeddings[position_bucket]),
        (f"V:{hit_length}", params.vocab_embeddings[hit_length]),
    )
    return tuple(
        FeatureHit(
            key=key,
            bucket=stable_feature_bucket(key, params.hash_salt, params.hash_buckets),
            vector=vector,
        )
        for key, vector in specs
    )


def max_vocab_length(params: ParameterPackage) -> int:
    return max(params.vocab_embeddings)


def feature_hits_for_sample(sample: CorpusSample, params: ParameterPackage) -> tuple[tuple[FeatureHit, ...], ...]:
    longest = max_vocab_length(params)
    return tuple(
        feature_hits_for_position(sample, index, params, longest)
        for index in range(len(sample.chars))
    )


def emission_scores_for_positions(
    sample: CorpusSample,
    params: ParameterPackage,
    positions: Iterable[int],
    *,
    max_scored_chars: int | None = None,
    nonfinite_at: int | None = None,
) -> tuple[list[list[int]], int]:
    """Return a full-length emission matrix and the number of scored positions.

    Positions absent from ``positions`` remain zero and do not count toward the
    budget.  This supports sliding-window execution units while preserving one
    score vector per character position.
    """

    length = len(sample.chars)
    matrix: list[list[int]] = [[0 for _ in params.labels] for _ in range(length)]
    scored = 0
    longest_vocab = max_vocab_length(params)
    scale = params.scale
    for index in positions:
        if max_scored_chars is not None and scored >= max_scored_chars:
            raise WorkbenchError(
                E_SAMPLE_QUOTA_EXCEEDED,
                "emission",
                f"sample {sample.sample_id} exhausted quota after {scored} scored chars",
            )
        hits = feature_hits_for_position(sample, int(index), params, longest_vocab)
        row: list[int] = []
        for label_index in range(len(params.labels)):
            label_weights = params.emission_weights[label_index]
            total = 0
            for hit in hits:
                weights = label_weights[hit.bucket]
                for dimension, value in enumerate(hit.vector):
                    total += fixed_mul_div(value, weights[dimension], scale)
            row.append(total)
        if nonfinite_at is not None and int(index) == int(nonfinite_at):
            row[-1] = math.nan
        if not all(isinstance(score, int) and math.isfinite(score) for score in row):
            raise WorkbenchError(
                E_NONFINITE_SCORE,
                "emission",
                f"non-finite emission score at sample {sample.sample_id}, position {index}",
            )
        matrix[int(index)] = row
        scored += 1
    return matrix, scored


def score_sample_once(
    sample: CorpusSample,
    params: ParameterPackage,
    *,
    nonfinite_at: int | None = None,
) -> list[list[int]]:
    matrix, _scored = emission_scores_for_positions(
        sample,
        params,
        range(len(sample.chars)),
        nonfinite_at=nonfinite_at,
    )
    return matrix


def block_signature(
    sample: CorpusSample,
    params: ParameterPackage,
    start: int,
    end: int,
) -> str:
    longest_vocab = max_vocab_length(params)
    digest = hashlib.sha256()
    for index in range(max(0, start), min(len(sample.chars), end)):
        for hit in feature_hits_for_position(sample, index, params, longest_vocab):
            digest.update(hit.key.encode("utf-8"))
            digest.update(b"|")
            digest.update(str(hit.bucket).encode("ascii"))
            digest.update(b"|")
            digest.update(",".join(str(value) for value in hit.vector).encode("ascii"))
            digest.update(b"\x1e")
    return digest.hexdigest()


def hash_collision_report(samples: Sequence[CorpusSample], params: ParameterPackage) -> dict[str, object]:
    buckets: dict[int, set[str]] = {}
    total_features = 0
    for sample in samples:
        for hits in feature_hits_for_sample(sample, params):
            for hit in hits:
                total_features += 1
                buckets.setdefault(hit.bucket, set()).add(hit.key)
    collisions = []
    collision_events = 0
    for bucket in sorted(buckets):
        keys = sorted(buckets[bucket])
        if len(keys) > 1:
            collision_events += len(keys) - 1
            collisions.append({"bucket": bucket, "key_count": len(keys), "keys": keys})
    return {
        "hash_buckets": params.hash_buckets,
        "distinct_feature_keys": len({key for keys in buckets.values() for key in keys}),
        "total_feature_hits": total_features,
        "occupied_buckets": len(buckets),
        "collision_events": collision_events,
        "colliding_bucket_count": len(collisions),
        "colliding_buckets": collisions[:16],
    }
