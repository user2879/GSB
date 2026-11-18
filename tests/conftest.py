from __future__ import annotations

import json
import os
import shutil

import pytest

from lora_bench.generator import build_default_dataset
from lora_bench.utils import read_json, write_json_atomic


@pytest.fixture()
def input_dir(tmp_path):
    target = str(tmp_path / "input")
    build_default_dataset(target)
    return target


@pytest.fixture()
def workspace(tmp_path):
    target = str(tmp_path / "workspace")
    os.makedirs(target, exist_ok=True)
    return target


@pytest.fixture()
def base_config():
    return {
        "rank": 2,
        "alpha": 4.0,
        "target_layers": [0, 2],
        "learning_rate": 0.02,
        "batch_size": 4,
        "epochs": 3,
        "metric": "accuracy",
        "metric_threshold": 0.5,
        "concurrency": 1,
        "commit_interval": 2,
    }


@pytest.fixture()
def edit_tools(input_dir):
    """Helpers that mutate sample/config inputs between incremental runs."""

    def read_samples():
        with open(os.path.join(input_dir, "samples.jsonl"), encoding="utf-8") as h:
            return [json.loads(line) for line in h if line.strip()]

    def write_samples(samples):
        path = os.path.join(input_dir, "samples.jsonl")
        with open(path, "w", encoding="utf-8", newline="\n") as h:
            for sample in sorted(samples, key=lambda item: item["sample_id"]):
                h.write(json.dumps(sample, sort_keys=True) + "\n")

    def update_sample(sample_id, mutate):
        samples = read_samples()
        for sample in samples:
            if sample["sample_id"] == sample_id:
                mutate(sample)
        write_samples(samples)

    def add_sample(record):
        samples = read_samples()
        assert not any(s["sample_id"] == record["sample_id"] for s in samples)
        samples.append(record)
        write_samples(samples)

    def delete_sample(sample_id):
        write_samples(
            [s for s in read_samples() if s["sample_id"] != sample_id]
        )
        # Keep the task manifest consistent by pruning the same id.
        tasks_path = os.path.join(input_dir, "tasks.json")
        manifest = read_json(tasks_path)
        for entry in manifest["tasks"]:
            entry["train_sample_ids"] = [
                sid
                for sid in entry["train_sample_ids"]
                if sid != sample_id
            ]
            entry["val_sample_ids"] = [
                sid for sid in entry["val_sample_ids"] if sid != sample_id
            ]
        write_json_atomic(tasks_path, manifest)

    def update_manifest(mutate):
        path = os.path.join(input_dir, "tasks.json")
        manifest = read_json(path)
        mutate(manifest)
        write_json_atomic(path, manifest)

    def replace_base(version):
        shutil.copytree(
            os.path.join(input_dir, f"base_{version}"),
            os.path.join(input_dir, "base_active"),
            dirs_exist_ok=True,
        )

    return {
        "read_samples": read_samples,
        "update_sample": update_sample,
        "add_sample": add_sample,
        "delete_sample": delete_sample,
        "update_manifest": update_manifest,
        "replace_base": replace_base,
    }
