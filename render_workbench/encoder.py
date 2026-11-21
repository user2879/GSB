"""Deterministic restricted-vocab hash bag-of-words prompt encoder."""

import numpy as np

from .hashing import array_fingerprint, fingerprint, random_state


def prompt_fingerprint(prompt):
    """Fingerprint over the prompt's semantic inputs (id + token sequence)."""
    return fingerprint({"prompt_id": prompt["id"], "tokens": prompt["tokens"]})


def encode_prompt(prompt, vocab, embedding_dim):
    """Build the normalized bag-of-words embedding.

    Tokens are counted in restricted-vocab order, each token's hash-derived
    word vector comes from an isolated RNG, and the summed vector is
    L2-normalized. Pure function: identical inputs always yield identical
    arrays, independently of any runtime global random state.
    """
    vocab_index = {token: i for i, token in enumerate(vocab)}
    counts = np.zeros(len(vocab), dtype=np.float64)
    for token in prompt["tokens"]:
        counts[vocab_index[token]] += 1.0
    counts = counts / counts.sum()

    embedding = np.zeros(embedding_dim, dtype=np.float64)
    for token, weight in zip(vocab, counts):
        if weight == 0.0:
            continue
        rng = random_state("word-vector", token, embedding_dim)
        embedding += weight * rng.standard_normal(embedding_dim)

    norm = float(np.linalg.norm(embedding))
    if norm > 0.0:
        embedding = embedding / norm
    return {
        "embedding": embedding.astype(np.float64, copy=False),
        "token_counts": counts.astype(np.float64, copy=False),
        "prompt_fp": prompt_fingerprint(prompt),
        "embedding_fp": array_fingerprint(embedding),
    }
