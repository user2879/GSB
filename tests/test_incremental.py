"""列级精确失效、增量与全量一致 (业务要求 6/7)."""

from __future__ import annotations

import json

import numpy as np

from cqdiag.cli import compare_results
from cqdiag.config import validate_config
from cqdiag.edits import apply_batch, config_to_raw
from cqdiag.errors import FailureList
from cqdiag.pipeline import Pipeline
from cqdiag.samples import SampleSet, SampleTensor, load_samples, samples_to_records
from cqdiag.store import SnapshotStore

from conftest import make_config, make_config_raw, make_samples


def _run(config, samples, workdir):
    store = SnapshotStore(workdir)
    return Pipeline(config, samples, store).run()


def test_replace_column_invalidates_only_that_column(tmp_path):
    config = make_config()
    samples = make_samples(config)
    _run(config, samples, tmp_path)  # 基线
    failures = FailureList()
    config, _ = apply_batch(config, samples,
                            {"ops": [{"op": "replace_column", "tensor": "t",
                                      "column": 2, "seed": 99}]}, failures)
    result = _run(config, samples, tmp_path)
    assert result.stats["calib"]["recomputed"] == ["t:2"]
    assert result.stats["scale"]["recomputed"] == ["t:2"]
    assert result.stats["quant"]["recomputed"] == ["t:2"]
    assert result.stats["dequant"]["recomputed"] == ["t:2"]
    assert sorted(result.stats["calib"]["reused"]) == ["t:0", "t:1", "t:3"]


def test_quantile_change_invalidates_calib_only(tmp_path):
    config = make_config()
    samples = make_samples(config)
    _run(config, samples, tmp_path)
    failures = FailureList()
    config, _ = apply_batch(config, samples,
                            {"ops": [{"op": "set_quantile", "lo_permille": 10,
                                      "hi_permille": 990}]}, failures)
    result = _run(config, samples, tmp_path)
    assert len(result.stats["calib"]["recomputed"]) == 4  # 全部列重算
    assert len(result.stats["scale"]["recomputed"]) == 4  # 下游级联
    assert result.stats["stratify"]["recomputed"] == ["t:groups"]


def test_bits_change_keeps_calib_reused(tmp_path):
    config = make_config()
    samples = make_samples(config)
    _run(config, samples, tmp_path)
    failures = FailureList()
    config, _ = apply_batch(config, samples,
                            {"ops": [{"op": "set_quant_bits", "bits": 4}]}, failures)
    result = _run(config, samples, tmp_path)
    assert result.stats["calib"]["recomputed"] == []
    assert len(result.stats["calib"]["reused"]) == 4
    assert len(result.stats["scale"]["recomputed"]) == 4
    assert len(result.stats["quant"]["recomputed"]) == 4


def test_group_repartition_invalidates_stratify_only(tmp_path):
    config = make_config()
    samples = make_samples(config)
    _run(config, samples, tmp_path)
    failures = FailureList()
    config, _ = apply_batch(config, samples,
                            {"ops": [{"op": "set_groups", "tensor": "t",
                                      "labels": [0, 1, 2, 3]}]}, failures)
    result = _run(config, samples, tmp_path)
    for stage in ("calib", "scale", "quant", "dequant"):
        assert result.stats[stage]["recomputed"] == []
        assert len(result.stats[stage]["reused"]) == 4
    assert result.stats["stratify"]["recomputed"] == ["t:groups"]


def test_incremental_equals_full_rebuild(tmp_path):
    config = make_config()
    samples = make_samples(config)
    failures = FailureList()
    result = _run(config, samples, tmp_path / "inc")
    batches = [
        {"ops": [{"op": "replace_column", "tensor": "t", "column": 1, "seed": 5},
                 {"op": "set_quantile", "lo_permille": 5, "hi_permille": 995}]},
        {"ops": [{"op": "set_quant_bits", "bits": 4},
                 {"op": "set_frac_bits", "bits": 5}]},
        {"ops": [{"op": "set_saturation", "strategy": "mask"},
                 {"op": "set_groups", "tensor": "t", "labels": [1, 1, 0, 0]}]},
        {"ops": [{"op": "set_quant_bits", "bits": 8},
                 {"op": "set_saturation", "strategy": "clamp"},
                 {"op": "set_tolerances", "abs_tol": 0.05, "rel_tol": 0.05}]},
    ]
    for batch in batches:
        config, _ = apply_batch(config, samples, batch, failures)
        result = _run(config, samples, tmp_path / "inc")
    # 从空白全量重建
    fresh_config = validate_config(config_to_raw(config))
    records = samples_to_records(samples, config)
    sample_file = tmp_path / "final.jsonl"
    with open(sample_file, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    fresh_samples = load_samples(str(sample_file), fresh_config)
    fresh_result = _run(fresh_config, fresh_samples, tmp_path / "full")
    assert compare_results(result, fresh_result) == []


def test_reused_artifacts_not_recomputed(tmp_path):
    config = make_config()
    samples = make_samples(config)
    _run(config, samples, tmp_path)
    result = _run(config, samples, tmp_path)  # 无编辑再跑
    for stage in ("calib", "scale", "quant", "dequant", "stratify"):
        assert result.stats[stage]["recomputed"] == []
