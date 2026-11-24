from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from .canonical import cache_key, fingerprint
from .config import ConfigBundle


def fnv1a_64(text: str) -> int:
    value = 0xCBF29CE484222325
    prime = 0x100000001B3
    mask = 0xFFFFFFFFFFFFFFFF
    for byte in text.encode("utf-8"):
        value ^= byte
        value = (value * prime) & mask
    return value


def subword_units(term: str, min_n: int = 3, max_n: int = 4) -> List[str]:
    """Boundary-marked character n-grams plus the whole term."""

    marked = f"^{term.lower()}$"
    units = [f"word:{term.lower()}"]
    for n in range(min_n, max_n + 1):
        if n == 1:
            units.extend(f"char:{char}" for char in term.lower())
            continue
        for index in range(0, max(0, len(marked) - n + 1)):
            units.append(f"ng{n}:{marked[index:index + n]}")
    return units


def feature_bucket(feature: str, hash_buckets: int) -> int:
    return fnv1a_64(feature) % int(hash_buckets)


def build_count_vector(terms: List[str], config: ConfigBundle) -> Tuple[List[int], List[Dict[str, Any]]]:
    buckets = int(config.data["hash_buckets"])
    counts = np.zeros(buckets, dtype=np.int64)
    examples: List[Dict[str, Any]] = []
    min_n = int(config.data["subword_min_n"])
    max_n = int(config.data["subword_max_n"])
    for term_index, term in enumerate(terms):
        for feature in subword_units(term, min_n=min_n, max_n=max_n):
            bucket = feature_bucket(feature, buckets)
            counts[bucket] += 1
            if len(examples) < 24:
                examples.append({"term_index": term_index, "feature": feature, "bucket": bucket})
    return counts.tolist(), examples


def feature_cache_key(content_fingerprint: str, config: ConfigBundle) -> str:
    return cache_key("feature-v1", content_fingerprint, config.feature_fingerprint)


def projection_cache_key(feature_key: str, config: ConfigBundle) -> str:
    return cache_key("projection-v1", feature_key, config.projection_fingerprint)


def make_projection_matrix(config: ConfigBundle, columns: int | None = None) -> np.ndarray:
    rows = int(config.data["hash_buckets"])
    cols = int(columns or config.data["projection_pool_dim"])
    seed_material = f"dense-index-projection|{config.projection_fingerprint}|{rows}|{cols}"
    seed = fnv1a_64(seed_material)
    generator = np.random.Generator(np.random.PCG64(seed))
    return generator.standard_normal((rows, cols), dtype=np.float64)


def project_counts(counts: List[int] | np.ndarray, config: ConfigBundle) -> List[float]:
    pool_dim = int(config.data["projection_pool_dim"])
    matrix = make_projection_matrix(config, pool_dim)
    vector = np.asarray(counts, dtype=np.float64) @ matrix
    return vector[: int(config.data["projection_dim"])].tolist()


def project_pool(counts: List[int] | np.ndarray, config: ConfigBundle) -> List[float]:
    matrix = make_projection_matrix(config)
    vector = np.asarray(counts, dtype=np.float64) @ matrix
    return vector.tolist()


def content_fingerprint_for_block(block: Dict[str, Any], config: ConfigBundle) -> str:
    return fingerprint(
        {
            "block_id": block["block_id"],
            "document_id": block["document_id"],
            "block_seq": block["block_seq"],
            "paragraph_id": block["paragraph_id"],
            "chunk_ord": block["chunk_ord"],
            "start": block["start"],
            "terms": block["terms"],
            "padded": block["padded"],
            "padding_count": block["padding_count"],
            "chunk_fingerprint": config.chunk_fingerprint,
        },
        24,
    )
