from __future__ import annotations

import json

import pytest

from ner_workbench.errors import E_SNAPSHOT_STALE, WorkbenchError
from ner_workbench.generator import generate_dataset
from ner_workbench.runner import Workbench, compare_reports


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("integration-dataset")
    samples = root / "samples"
    generate_dataset(samples, seed=20261121, sample_count=32)
    return samples


def test_parallel_determinism_injections_and_dashboard(dataset, tmp_path):
    single = Workbench(dataset, tmp_path / "workers1", workers=1).run()
    parallel = Workbench(dataset, tmp_path / "workers4", workers=4).run()
    assert single.exit_code == 0
    assert parallel.exit_code == 0

    comparison = compare_reports(single.report_path, parallel.report_path)
    assert comparison["equal"] is True
    assert comparison["left_digest"] == comparison["right_digest"]

    deterministic = single.report["deterministic"]
    assert deterministic["summary"]["consumed_char_budget"] == 489
    assert deterministic["summary"]["injection_hit_count"] == 4
    assert deterministic["summary"]["cache_eviction_count"] > 0
    assert deterministic["summary"]["cache_recomputation_count"] > 0
    assert deterministic["feature_hash_collisions"]["collision_events"] > 0
    assert {hit["code"] for hit in deterministic["injections"]["hits"]} == {
        "E_NONFINITE_SCORE",
        "E_LABEL_LENGTH_MISMATCH",
        "E_PARAM_SHAPE_MISMATCH",
        "E_SAMPLE_QUOTA_EXCEEDED",
    }
    assert any(item["code"] == "E_GLOBAL_BUDGET_EXHAUSTED" for item in deterministic["failures"])
    html = single.dashboard_path.read_text(encoding="utf-8")
    assert "NER 确定性调度与预算诊断工作台" in html
    assert "实体高亮与逐样本诊断" in html
    assert "SMP-0032" in html


def test_stop_then_resume_matches_uninterrupted(dataset, tmp_path):
    full = Workbench(dataset, tmp_path / "full", workers=1).run()
    drill_dir = tmp_path / "drill"
    interrupted = Workbench(dataset, drill_dir, workers=1, stop_after=12).run()
    assert interrupted.status == "interrupted"
    recovery_point = interrupted.report["deterministic"]["snapshot"]["recovery_point_sample_id"]
    assert recovery_point is not None

    resumed = Workbench(dataset, drill_dir, workers=1, resume=True).run()
    assert resumed.status == "complete"
    comparison = compare_reports(full.report_path, resumed.report_path)
    assert comparison["equal"] is True

    with pytest.raises(WorkbenchError) as exc_info:
        Workbench(dataset, drill_dir, workers=2, resume=True).run()
    assert exc_info.value.code == E_SNAPSHOT_STALE


def test_snapshot_current_is_atomic_symlink(dataset, tmp_path):
    outcome = Workbench(dataset, tmp_path / "atomic", workers=1, stop_after=4).run()
    current = tmp_path / "atomic" / "snapshots" / "CURRENT"
    assert current.is_symlink()
    manifest = json.loads((current / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["global_version"] >= 2
    assert manifest["completed_sample_ids"] == sorted(manifest["completed_sample_ids"])
    assert (current / "load.json").exists()
    assert (current / "emission.json").exists()
    assert (current / "decode.json").exists()
    assert (current / "merge.json").exists()
    assert outcome.report["deterministic"]["snapshot"]["recovery_point_sample_id"] == manifest["recovery_point_sample_id"]


def test_workbench_keeps_processing_after_corpus_failures(tmp_path):
    samples = tmp_path / "samples"
    generate_dataset(samples, seed=55, sample_count=6)
    with (samples / "corpus.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "sample_id": "BAD-EMPTY",
                    "chars": "",
                    "labels": "",
                    "entity_types": [],
                    "entities": [],
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        handle.write(
            json.dumps(
                {
                    "sample_id": "BAD-LABEL",
                    "chars": "甲 乙",
                    "labels": "O X-人物",
                    "entity_types": [],
                    "entities": [],
                },
                ensure_ascii=False,
            )
            + "\n"
        )
    outcome = Workbench(samples, tmp_path / "out", workers=2).run()
    assert outcome.exit_code == 0
    codes = {item["code"] for item in outcome.report["deterministic"]["failures"]}
    assert "E_CORPUS_INVALID" in codes
    assert "E_LABEL_UNKNOWN" in codes
    summary = outcome.report["deterministic"]["summary"]
    assert summary["valid_sample_count"] == 6
    assert summary["processed_sample_count"] == 6
    assert summary["successful_sample_count"] >= 1


def test_invalid_parameter_package_is_reported_without_crashing(tmp_path):
    samples = tmp_path / "samples"
    generate_dataset(samples, seed=56, sample_count=4)
    payload = json.loads((samples / "model_params.json").read_text(encoding="utf-8"))
    payload["emission_weights"] = payload["emission_weights"][:-1]
    (samples / "model_params.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    outcome = Workbench(samples, tmp_path / "out", workers=1).run()
    assert outcome.exit_code == 2
    assert outcome.status == "model_invalid"
    assert outcome.report["deterministic"]["failures"][0]["code"] == "E_PARAM_SHAPE_MISMATCH"
