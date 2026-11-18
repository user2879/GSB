"""Frozen base forward representation stage with content-addressed caching."""

from __future__ import annotations

import os

import numpy as np

from .loader import BasePack
from .utils import hash_json, read_json, write_json_atomic


def sample_content_fingerprint(sample: dict) -> str:
    """Fingerprint of every training-relevant field of a sample."""
    return hash_json(
        {
            "sample_id": sample["sample_id"],
            "task": sample["task"],
            "input_tokens": list(sample["input_tokens"]),
            "target_tokens": list(sample["target_tokens"]),
            "label": sample["label"],
        }
    )


def forward_cache_key(base_fingerprint: str, sample_fingerprint: str) -> str:
    """Per-sample representation cache key (base-level dependency)."""
    return hash_json(
        {"base": base_fingerprint, "sample": sample_fingerprint}
    )[:16]


def compute_forward_cache(sample: dict, base: BasePack) -> dict[str, np.ndarray]:
    """Run one deterministic forward pass; layer order is ascending.

    The base weights are never modified and no gradient state is kept: this is
    a strictly read-only stage.
    """
    return base.forward(sample["input_tokens"])


def build_forward_stage(
    samples: list[dict],
    base: BasePack,
    stage_dir: str,
    reuse_dir: str | None = None,
) -> dict:
    """Materialise per-sample representations, reusing unchanged sample caches.

    Iteration order is fixed: sample id ascending. Existing per-sample cache
    files from a previous base stage are reused only when the cache key matches,
    so a sample edit only recomputes that sample and a base replacement
    recomputes every representation.
    """
    os.makedirs(stage_dir, exist_ok=True)
    previous_index: dict[str, str] = {}
    if reuse_dir:
        index_path = os.path.join(reuse_dir, "index.json")
        if os.path.exists(index_path):
            previous_index = {
                entry["cache_key"]: entry["file"]
                for entry in read_json(index_path)["entries"]
            }
    entries = []
    # Explicit sorted iteration satisfies the fixed accumulation order.
    for sample in sorted(samples, key=lambda item: item["sample_id"]):
        sample_fp = sample_content_fingerprint(sample)
        cache_key = forward_cache_key(base.base_fingerprint, sample_fp)
        file_name = f"{cache_key}.npz"
        target_path = os.path.join(stage_dir, file_name)
        if cache_key in previous_index and not os.path.exists(target_path):
            previous_file = os.path.join(reuse_dir, previous_index[cache_key])
            if os.path.exists(previous_file):
                with open(previous_file, "rb") as src, open(
                    target_path, "wb"
                ) as dst:
                    dst.write(src.read())
        if not os.path.exists(target_path):
            cache = compute_forward_cache(sample, base)
            np.savez(target_path, **cache)
        entries.append(
            {
                "sample_id": sample["sample_id"],
                "sample_fingerprint": sample_fp,
                "cache_key": cache_key,
                "file": file_name,
            }
        )
    entries.sort(key=lambda entry: entry["sample_id"])
    manifest = {
        "stage": "forward",
        "base_version": base.base_version,
        "base_fingerprint": base.base_fingerprint,
        "entry_count": len(entries),
        "entries": entries,
    }
    write_json_atomic(os.path.join(stage_dir, "index.json"), manifest)
    return manifest


def load_forward_entry(stage_dir: str, entry: dict) -> dict[str, np.ndarray]:
    with np.load(os.path.join(stage_dir, entry["file"])) as data:
        return {key: data[key] for key in data.files}


def forward_index_by_sample(stage_manifest: dict) -> dict[str, dict]:
    return {entry["sample_id"]: entry for entry in stage_manifest["entries"]}
