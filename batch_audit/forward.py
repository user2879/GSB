"""Deterministic small MLP forward inference and output digest binding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .canonical import (
    digest_batch_output,
    digest_sample_output,
    rounded_output_vector,
)


@dataclass(frozen=True)
class MemberOutput:
    sample_id: str
    values: np.ndarray
    sample_output_digest: str


@dataclass(frozen=True)
class BatchOutput:
    values_by_sample: Mapping[str, np.ndarray]
    sample_digests: Mapping[str, str]
    batch_output_digest: str


def forward_one(inputs: np.ndarray, model: Mapping[str, np.ndarray]) -> np.ndarray:
    x = np.asarray(inputs, dtype="<f8")
    h1 = np.tanh(_linear_fixed_order(x, model["W1"], model["b1"]))
    h2 = np.tanh(_linear_fixed_order(h1, model["W2"], model["b2"]))
    out = np.tanh(_linear_fixed_order(h2, model["W3"], model["b3"]))
    return rounded_output_vector(out)


def _linear_fixed_order(
    vector: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    """Compute ``bias + sum_i vector[i] * weights[i]`` in fixed scalar order.

    Deliberately avoiding BLAS keeps the arithmetic sequence stable across
    matrix shapes and BLAS implementations.  The tensors are tiny, so this is a
    worthwhile trade for auditability.
    """

    result = np.asarray(bias, dtype="<f8").copy()
    for index in range(vector.shape[0]):
        result = result + vector[index] * weights[index]
    return result


def forward_batch(
    sample_ids: Sequence[str],
    inputs_by_id: Mapping[str, np.ndarray],
    model: Mapping[str, np.ndarray],
) -> BatchOutput:
    """Run one batch transaction while keeping each row independent.

    The batch digest is built from sample-id-sorted rows, so member position in
    the caller's list cannot affect either inference or the digest.
    """

    sample_digests: dict[str, str] = {}
    values: dict[str, np.ndarray] = {}
    for sample_id in sample_ids:
        output = forward_one(inputs_by_id[sample_id], model)
        values[sample_id] = output
        sample_digests[sample_id] = digest_sample_output(sample_id, output)
    batch_digest = digest_batch_output(
        [(sample_id, sample_digests[sample_id]) for sample_id in sample_ids]
    )
    return BatchOutput(
        values_by_sample=values,
        sample_digests=sample_digests,
        batch_output_digest=batch_digest,
    )
