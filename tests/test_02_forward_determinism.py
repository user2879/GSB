import numpy as np

from lora_bench.forward import (
    build_forward_stage,
    compute_forward_cache,
    forward_cache_key,
    sample_content_fingerprint,
)
from lora_bench.loader import load_base_pack, load_task_manifest, load_samples
import os


def _load(input_dir):
    base = load_base_pack(os.path.join(input_dir, "base_v1"))
    manifest = load_task_manifest(os.path.join(input_dir, "tasks.json"))
    samples, failures = load_samples(
        os.path.join(input_dir, "samples.jsonl"),
        {entry["task"] for entry in manifest["tasks"]},
        vocab_size=32,
    )
    assert failures == []
    return base, samples


def test_forward_bitwise_repeatable(input_dir):
    base, samples = _load(input_dir)
    first = compute_forward_cache(samples[0], base)
    second = compute_forward_cache(samples[0], base)
    assert first.keys() == second.keys()
    for key in first:
        assert np.array_equal(first[key], second[key])


def test_forward_order_and_manifest(input_dir, tmp_path):
    base, samples = _load(input_dir)
    stage_a = str(tmp_path / "a")
    stage_b = str(tmp_path / "b")
    manifest_a = build_forward_stage(samples, base, stage_a)
    manifest_b = build_forward_stage(samples, base, stage_b)
    assert manifest_a == manifest_b
    ids = [entry["sample_id"] for entry in manifest_a["entries"]]
    assert ids == sorted(ids)
    assert manifest_a["base_fingerprint"] == base.base_fingerprint


def test_cache_key_changes_with_sample_and_base(input_dir):
    base, samples = _load(input_dir)
    sample = dict(samples[0])
    key_a = forward_cache_key(
        base.base_fingerprint, sample_content_fingerprint(sample)
    )
    edited = dict(sample, input_tokens=list(sample["input_tokens"]) + [1])
    key_b = forward_cache_key(
        base.base_fingerprint, sample_content_fingerprint(edited)
    )
    base_v2 = load_base_pack(os.path.join(input_dir, "base_v2"))
    key_c = forward_cache_key(
        base_v2.base_fingerprint, sample_content_fingerprint(sample)
    )
    assert key_a != key_b
    assert key_a != key_c
