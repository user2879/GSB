import json
import os

import numpy as np
import pytest

from lora_bench.errors import (
    E_BASE_FINITE_MISMATCH,
    E_BASE_SHAPE_MISMATCH,
    E_SAMPLE_INVALID,
    WorkbenchError,
)
from lora_bench.loader import load_base_pack, load_samples, load_task_manifest


def test_valid_load(input_dir):
    manifest = load_task_manifest(os.path.join(input_dir, "tasks.json"))
    allowed = {entry["task"] for entry in manifest["tasks"]}
    samples, failures = load_samples(
        os.path.join(input_dir, "samples.jsonl"), allowed, vocab_size=32
    )
    assert failures == []
    ids = [s["sample_id"] for s in samples]
    assert len(ids) == len(set(ids))
    assert all(s["input_tokens"] for s in samples)


def test_sample_failures_are_classified(input_dir, tmp_path):
    manifest = load_task_manifest(os.path.join(input_dir, "tasks.json"))
    allowed = {entry["task"] for entry in manifest["tasks"]}
    path = os.path.join(input_dir, "samples.jsonl")
    with open(path, encoding="utf-8") as handle:
        lines = handle.readlines()
    good = json.loads(lines[1])
    other = json.loads(lines[2])
    missing_field = dict(good)
    del missing_field["input_tokens"]
    duplicate = dict(other, sample_id=other["sample_id"])
    bad_tokens = dict(good, sample_id="bad-tokens", input_tokens=[0, 999])
    bad_task = dict(good, sample_id="bad-task", task="not-a-task")
    broken = "{not json"
    override = tmp_path / "samples.jsonl"
    override.write_text(
        "\n".join(
            [
                json.dumps(other),
                json.dumps(missing_field),
                json.dumps(duplicate),
                json.dumps(bad_tokens),
                json.dumps(bad_task),
                broken,
            ]
        ),
        encoding="utf-8",
    )
    samples, failures = load_samples(str(override), allowed, vocab_size=32)
    codes = [failure["code"] for failure in failures]
    assert codes.count(E_SAMPLE_INVALID) == 5
    assert any("duplicate" in failure["message"] for failure in failures)


def test_base_shape_mismatch(input_dir):
    base_dir = os.path.join(input_dir, "base_v1")
    tensor = os.path.join(base_dir, "layer0.weight.bin")
    raw = np.fromfile(tensor, dtype=np.float32)
    raw[:-2].tofile(tensor)
    with pytest.raises(WorkbenchError) as info:
        load_base_pack(base_dir)
    assert info.value.code == E_BASE_SHAPE_MISMATCH


def test_base_non_finite(input_dir):
    base_dir = os.path.join(input_dir, "base_v1")
    tensor = os.path.join(base_dir, "layer0.bias.bin")
    raw = np.fromfile(tensor, dtype=np.float32).copy()
    raw[0] = np.nan
    raw.tofile(tensor)
    with pytest.raises(WorkbenchError) as info:
        load_base_pack(base_dir)
    assert info.value.code == E_BASE_FINITE_MISMATCH


def test_base_fingerprint_tampering_detected(input_dir):
    base_dir = os.path.join(input_dir, "base_v1")
    shapes_path = os.path.join(base_dir, "shapes.json")
    descriptor = json.loads(open(shapes_path, encoding="utf-8").read())
    descriptor["base_fingerprint"] = "deadbeef"
    with open(shapes_path, "w", encoding="utf-8") as handle:
        json.dump(descriptor, handle)
    with pytest.raises(WorkbenchError) as info:
        load_base_pack(base_dir)
    assert info.value.code == E_BASE_SHAPE_MISMATCH
