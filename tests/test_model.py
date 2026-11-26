from __future__ import annotations

from ckptsim.model import (
    build_batch_samples,
    compute_batch,
    compute_loss,
    generate_parameter_set,
)
from ckptsim.spec import load_bundle


def _bundle(files):
    return load_bundle(files.shards, files.layers, files.checkpoints, files.events)


def test_parameter_and_loss_digests_are_order_independent(spec_files) -> None:
    bundle = _bundle(spec_files)
    shard = bundle.shards.by_id()["shard-a"]
    version = bundle.layers.by_version()[1]
    first = generate_parameter_set(shard, version)
    second = generate_parameter_set(shard, version)
    assert first.digest == second.digest
    batch_a = compute_batch(shard, version, "b-a1", [3, 1, 2])
    samples = build_batch_samples(shard, first, "b-a1", [1, 2, 3])
    reversed_samples = {key: samples[key] for key in reversed(list(samples))}
    batch_b = compute_batch(
        shard, version, "b-a1", [2, 3, 1], samples=reversed_samples
    )
    assert batch_a.parameter_digest == batch_b.parameter_digest
    assert batch_a.loss.digest == batch_b.loss.digest
    assert batch_a.loss.sample_ids == (1, 2, 3)
    assert batch_a.loss.payload["sample_ids"] == [1, 2, 3]


def test_loss_summary_binds_version_and_sample_set(spec_files) -> None:
    bundle = _bundle(spec_files)
    shard = bundle.shards.by_id()["shard-a"]
    v1 = bundle.layers.by_version()[1]
    v2 = bundle.layers.by_version()[2]
    first = compute_batch(shard, v1, "batch", [1, 2])
    same = compute_batch(shard, v1, "batch", [2, 1])
    other_version = compute_batch(shard, v2, "batch", [1, 2])
    other_samples = compute_batch(shard, v1, "batch", [1, 3])
    assert first.loss.digest == same.loss.digest
    assert first.loss.digest != other_version.loss.digest
    assert first.loss.digest != other_samples.loss.digest
    assert first.loss.payload["parameter_digest"] == first.parameter_digest


def test_missing_optional_samples_are_regenerated_deterministically(spec_files) -> None:
    bundle = _bundle(spec_files)
    shard = bundle.shards.by_id()["shard-b"]
    version = bundle.layers.by_version()[1]
    parameters = generate_parameter_set(shard, version)
    generated = build_batch_samples(shard, parameters, "b-b1", [4, 5])
    with_samples = compute_loss(shard, parameters, "b-b1", [4, 5], samples=generated)
    without_samples = compute_loss(shard, parameters, "b-b1", [4, 5])
    assert with_samples.digest == without_samples.digest

