from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from dataclasses import asdict, is_dataclass
from fractions import Fraction
from typing import Any, Iterable


MASK64 = (1 << 64) - 1
FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211


def canonical_json(value: Any) -> str:
    """Serialize a value in the one canonical order used by fingerprints."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
        allow_nan=False,
    )


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, Fraction):
        return {"num": value.numerator, "den": value.denominator}
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def short_fingerprint(value: Any, length: int = 16) -> str:
    return fingerprint(value)[:length]


def fnv1a_64(text: str) -> int:
    h = FNV_OFFSET
    for byte in text.encode("utf-8"):
        h ^= byte
        h = (h * FNV_PRIME) & MASK64
    return h


def is_strict_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def round_half_up(num: int, den: int) -> int:
    """Round num/den to the nearest integer, ties away from zero."""
    if den <= 0:
        raise ValueError("den must be positive")
    if num >= 0:
        return (2 * num + den) // (2 * den)
    return -((-2 * num + den) // (2 * den))


def fraction_decimal_text(value: Fraction, places: int = 6) -> str:
    sign = "-" if value < 0 else ""
    value = abs(value)
    rounded_scaled = round_half_up(value.numerator * (10**places), value.denominator)
    integer_part, fractional_part = divmod(abs(rounded_scaled), 10**places)
    if places == 0:
        return f"{sign}{integer_part}"
    return f"{sign}{integer_part}.{fractional_part:0{places}d}"


_WORD_RE = re.compile(r"[0-9a-z]+")


def normalize_text(text: str, normalize: bool) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    if normalize:
        text = text.lower()
    return text


def whitespace_terms(text: str, normalize: bool) -> list[str]:
    text = normalize_text(text, normalize)
    return [part for part in re.split(r"\s+", text.strip()) if part]


def subword_tokens(text: str, rule_version: str, normalize: bool) -> list[str]:
    """Deterministic word and character-subword expansion.

    sw-v1 emits words and trigrams; sw-v2 emits boundary-marked bigrams and
    trigrams.  The rule version is part of every feature cache key.
    """
    text = normalize_text(text, normalize)
    words = _WORD_RE.findall(text)
    tokens: list[str] = []
    for word in words:
        tokens.append(f"w:{word}")
        if rule_version == "sw-v1":
            if len(word) <= 3:
                tokens.append(f"sw:{word}")
            else:
                for i in range(len(word) - 2):
                    tokens.append(f"tri:{word[i:i + 3]}")
        elif rule_version == "sw-v2":
            marked = f"^{word}$"
            for n in (2, 3):
                if len(marked) < n:
                    continue
                for i in range(len(marked) - n + 1):
                    tokens.append(f"g{n}:{marked[i:i + n]}")
        else:
            raise ValueError(f"unknown subword rule: {rule_version}")
    return tokens


def bucket_vector(text: str, bucket_count: int, rule_version: str, normalize: bool) -> list[int]:
    counts = [0] * int(bucket_count)
    for token in subword_tokens(text, rule_version, normalize):
        bucket = fnv1a_64(token) % bucket_count
        counts[bucket] += 1
    return counts


def project_feature_vector(buckets: list[int], feature_dim: int) -> list[int]:
    features = [0] * feature_dim
    bucket_count = len(buckets)
    for bucket_index in range(bucket_count):
        value = buckets[bucket_index]
        if value:
            features[bucket_index % feature_dim] += value
    return features


def ordered_dot(left: Iterable[int], right: Iterable[int]) -> int:
    total = 0
    for a, b in zip(left, right):
        total += int(a) * int(b)
    return total


def ordered_sum(values: Iterable[int]) -> int:
    total = 0
    for value in values:
        total += int(value)
    return total


def session_order_key(session_id: str, turn: int, window_length: int | None = None) -> tuple[str, int, int]:
    return (str(session_id), int(turn), 10**9 if window_length is None else int(window_length))


def stable_task_hash(session_id: str, turn: int, window_length: int | None = None) -> str:
    return fingerprint({"sid": session_id, "turn": turn, "L": window_length})
