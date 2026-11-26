"""Deterministic two-stage sampling and NumPy fixed-point block verification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .canonical import stable_u64
from .spec import FragmentEvent, NodeSpec, RATE_SCALE, RevisionSpec


BLOCK_WEIGHTS = np.array([1, 3, 7, 13, 17, 19, 23, 29], dtype=np.int64)


@dataclass(frozen=True)
class SamplingOutcome:
    payload: dict[str, Any]

    @property
    def sampled_blocks(self) -> list[int]:
        return list(self.payload["sampled_blocks"])

    @property
    def status(self) -> str:
        return str(self.payload["status"])


def _rate_hit(node_seed: int, fragment_id: str, stage: str, index: int, rate_ppm: int) -> bool:
    if rate_ppm <= 0:
        return False
    if rate_ppm >= RATE_SCALE:
        return True
    bucket = stable_u64("edgecache", stage, node_seed, fragment_id, index) % RATE_SCALE
    return bucket < rate_ppm


def select_blocks(node: NodeSpec, fragment: FragmentEvent, revision: RevisionSpec) -> tuple[bool, list[int]]:
    fragment_selected = _rate_hit(
        node.seed,
        fragment.fragment_id,
        "fragment",
        fragment.sequence,
        revision.fragment_rate_ppm,
    )
    if not fragment_selected:
        return False, []
    selected = [
        block_index
        for block_index in range(fragment.block_count)
        if _rate_hit(
            node.seed,
            fragment.fragment_id,
            "block",
            block_index,
            revision.block_rate_ppm,
        )
    ]
    return True, selected


def _rng(node: NodeSpec, fragment: FragmentEvent) -> np.random.Generator:
    fragment_entropy = stable_u64("fragment-content", fragment.fragment_id)
    return np.random.Generator(
        np.random.PCG64(
            np.random.SeedSequence(
                [node.seed, fragment_entropy, fragment.block_count, node.fixed_scale]
            )
        )
    )


def generate_fixed_content(node: NodeSpec, fragment: FragmentEvent, corrupt: bool) -> np.ndarray:
    rng = _rng(node, fragment)
    low = -4 * node.fixed_scale
    high = 4 * node.fixed_scale + 1
    values = rng.integers(low, high, size=(fragment.block_count, 8), dtype=np.int64)
    if corrupt:
        for block_index, delta in sorted(fragment.corrupt_blocks.items()):
            values[block_index, 0] = np.int64(values[block_index, 0] + delta)
    return values


def block_digests(node: NodeSpec, fragment: FragmentEvent, corrupt: bool = False) -> np.ndarray:
    values = generate_fixed_content(node, fragment, corrupt=corrupt)
    return np.asarray(values @ BLOCK_WEIGHTS, dtype=np.int64)


def compute_sampling(
    node: NodeSpec,
    fragment: FragmentEvent,
    revision: RevisionSpec,
    *,
    duration_monotonic_ms: int | None = None,
) -> SamplingOutcome:
    fragment_selected, sampled = select_blocks(node, fragment, revision)
    clean = block_digests(node, fragment, corrupt=False)
    actual = block_digests(node, fragment, corrupt=True)
    if sampled:
        selected_indices = np.asarray(sampled, dtype=np.int64)
        expected_values = clean[selected_indices]
        actual_values = actual[selected_indices]
        differences = actual_values - expected_values
        error_count = int(np.count_nonzero(differences))
        error_signed = int(np.sum(differences))
        error_abs = int(np.sum(np.abs(differences)))
        digest_mismatches = sorted(
            sampled[int(position)] for position in np.flatnonzero(differences)
        )
    else:
        error_count = 0
        error_signed = 0
        error_abs = 0
        digest_mismatches = []
    covered = len(sampled)
    uncovered = fragment.block_count - covered
    uncovered_ppm = (uncovered * RATE_SCALE) // fragment.block_count
    if error_count:
        status = "fail"
    elif covered == fragment.block_count and fragment_selected:
        status = "pass"
    elif covered == 0:
        status = "unsampled"
    else:
        status = "partial"
    block_candidates = fragment.block_count if fragment_selected else 0
    counts = {
        "fragment_candidates": 1,
        "fragment_selected": int(fragment_selected),
        "block_candidates": block_candidates,
        "block_selected": covered,
        "block_uncovered": uncovered,
    }
    payload: dict[str, Any] = {
        "fragment_id": fragment.fragment_id,
        "node_id": node.node_id,
        "batch_id": fragment.batch_id,
        "quota_version": revision.revision_id,
        "quota": revision.quota,
        "fragment_rate_ppm": revision.fragment_rate_ppm,
        "block_rate_ppm": revision.block_rate_ppm,
        "fixed_scale": node.fixed_scale,
        "fragment_sampled": fragment_selected,
        "sampled_blocks": sampled,
        "sampling_counts": counts,
        "covered_blocks": covered,
        "uncovered_blocks": uncovered,
        "uncovered_ratio_ppm": uncovered_ppm,
        "quantization_error_bound": {
            "numerator": uncovered,
            "denominator": 2 * node.fixed_scale,
            "name": "uncovered_blocks_half_fixed_unit",
        },
        "error_count": error_count,
        "error_signed_fixed": error_signed,
        "error_abs_fixed": error_abs,
        "digest_mismatches": digest_mismatches,
        "status": status,
        "passed_blocks_are_only_checked": True,
    }
    if duration_monotonic_ms is not None:
        payload["duration_monotonic_ms"] = int(duration_monotonic_ms)
    return SamplingOutcome(payload=payload)


def deterministic_sampling_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "duration_monotonic_ms"}

