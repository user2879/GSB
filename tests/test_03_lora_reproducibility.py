import os

import numpy as np

from lora_bench.config import parse_config
from lora_bench.forward import build_forward_stage
from lora_bench.loader import load_base_pack, load_task_manifest
from lora_bench.lora import train_task
from lora_bench.pipeline import run_pipeline
from lora_bench.utils import read_json


def _prepare(input_dir, tmp_path, config):
    base = load_base_pack(os.path.join(input_dir, "base_v1"))
    manifest = load_task_manifest(os.path.join(input_dir, "tasks.json"))
    from lora_bench.loader import load_samples

    samples, failures = load_samples(
        os.path.join(input_dir, "samples.jsonl"),
        {entry["task"] for entry in manifest["tasks"]},
        vocab_size=32,
    )
    assert failures == []
    forward_dir = str(tmp_path / "forward")
    forward_manifest = build_forward_stage(samples, base, forward_dir)
    parsed = parse_config(config, base.layer_count)
    return base, manifest, samples, forward_dir, forward_manifest, parsed


def test_training_bitwise_reproducible(input_dir, tmp_path, base_config):
    base, manifest, samples, fwd_dir, fwd_manifest, cfg = _prepare(
        input_dir, tmp_path, base_config
    )
    entry = manifest["tasks"][0]
    first = train_task(entry, samples, base, cfg, fwd_dir, fwd_manifest)
    second = train_task(entry, samples, base, cfg, fwd_dir, fwd_manifest)
    assert first["epoch_losses"] == second["epoch_losses"]
    assert first["epoch_metrics"] == second["epoch_metrics"]
    assert first["final_fingerprint"] == second["final_fingerprint"]
    for key in first["arrays"]:
        assert np.array_equal(first["arrays"][key], second["arrays"][key])
    assert len(first["epoch_losses"]) == cfg.epochs
    assert all(np.isfinite(loss) for loss in first["epoch_losses"])


def test_tie_break_fields_present(input_dir, tmp_path, base_config):
    base, manifest, samples, fwd_dir, fwd_manifest, cfg = _prepare(
        input_dir, tmp_path, base_config
    )
    result = train_task(
        manifest["tasks"][0], samples, base, cfg, fwd_dir, fwd_manifest
    )
    best = result["best"]
    assert 0 <= best["best_epoch"] < cfg.epochs
    assert "passed" in best and "score" in best
    assert len(result["epoch_fingerprints"]) == cfg.epochs


def test_pipeline_repeat_run_identical_artifacts(
    input_dir, workspace, base_config
):
    first = run_pipeline(input_dir, workspace, base_config)
    first_summary = {
        task: read_json(
            os.path.join(workspace, "current", "train", f"{task}.json")
        )
        for task in first["manifest"]["completed_tasks"]
    }
    second = run_pipeline(input_dir, workspace, base_config)
    second_summary = {
        task: read_json(
            os.path.join(workspace, "current", "train", f"{task}.json")
        )
        for task in second["manifest"]["completed_tasks"]
    }
    assert first_summary == second_summary
    assert second["reused_tasks"] == first["manifest"]["completed_tasks"]
