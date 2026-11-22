from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from domain_pipeline.config import load_config_yaml, validate_config
from domain_pipeline.edits import EditBatch, EditOperation, EditPlan, apply_batch, load_edit_plan
from domain_pipeline.errors import SnapshotStaleError
from domain_pipeline.features import compute_features
from domain_pipeline.loader import SampleStore, parse_samples, state_failures
from domain_pipeline.pipeline import PipelineRunner
from domain_pipeline.snapshot import SnapshotStore
from domain_pipeline.synthetic import (
    DEMO_CONFIG,
    demo_edit_plan,
    generate_demo_workspace,
    make_sample,
)
from domain_pipeline.training import train_model
from domain_pipeline.yaml_compat import safe_load as compat_safe_load


def sample_lines(indices: list[int]) -> list[str]:
    return [
        json.dumps(make_sample(index), sort_keys=True, separators=(",", ":"))
        for index in indices
    ]


def new_runner(tmp_path: Path, config: dict | None = None, indices: list[int] | None = None):
    raw_config = copy.deepcopy(config or DEMO_CONFIG)
    store = SnapshotStore(tmp_path / "snapshots")
    runner = PipelineRunner(
        snapshot_store=store,
        sample_store=SampleStore.from_lines(sample_lines(indices or list(range(16)))),
        raw_config=raw_config,
        pipeline_complete=False,
    )
    return runner, store


def test_loader_and_config_recover_per_entry() -> None:
    raw_config = copy.deepcopy(DEMO_CONFIG)
    raw_config["features"]["bins"] = 0
    raw_config["concurrency"]["max_workers"] = 99
    raw_config["concurrency"]["max_workers_limit"] = 4
    raw_config["concurrency"]["batch_budget"] = 0
    config = validate_config(raw_config)
    codes = {code for failure in config.failures for code in failure["codes"]}
    assert {"E_CONFIG_BINS", "E_CONFIG_CONCURRENCY_LIMIT", "E_CONFIG_BUDGET"} <= codes
    assert config.concurrency["max_workers"] == 4

    valid = json.dumps(make_sample(0), sort_keys=True)
    duplicate = json.dumps(make_sample(1) | {"id": "S0001"}, sort_keys=True)
    nonfinite = json.dumps(make_sample(2) | {"age": "not-a-number"}, sort_keys=True)
    bad_category = json.dumps(make_sample(3) | {"region": "unknown"}, sort_keys=True)
    bad_label = json.dumps(make_sample(4) | {"label": 2}, sort_keys=True)
    malformed = "{not-json"
    parsed = parse_samples(
        [valid, duplicate, nonfinite, bad_category, bad_label, malformed],
        validate_config(copy.deepcopy(DEMO_CONFIG)),
    )
    assert [sample.sample_id for sample in parsed.samples] == ["S0001"]
    failures = state_failures([], parsed.failures)
    assert len(failures) == 5
    assert [failure["line"] for failure in failures] == [2, 3, 4, 5, 6]
    all_codes = {code for failure in failures for code in failure["codes"]}
    assert {
        "E_SAMPLE_ID_DUPLICATE",
        "E_NUM_NONFINITE",
        "E_CATEGORY_DOMAIN",
        "E_LABEL_BINARY",
        "E_JSON_PARSE",
    } <= all_codes


def test_feature_artifacts_are_deterministic_and_missing_is_a_bin() -> None:
    config = validate_config(copy.deepcopy(DEMO_CONFIG))
    parsed = parse_samples(sample_lines(list(range(24))), config)
    first = compute_features(config, parsed.samples, row_cache={})
    second = compute_features(config, parsed.samples, row_cache={})
    assert first.complete and second.complete
    assert first.artifact == second.artifact
    assert len(first.artifact["content_fingerprint"]) == 64
    for column in first.artifact["numeric_columns"]:
        assert len(column["content_fingerprint"]) == 64
    for column in first.artifact["categorical_columns"]:
        assert len(column["content_fingerprint"]) == 64
    age = next(
        column
        for column in first.artifact["numeric_columns"]
        if column["name"] == "age"
    )
    assert age["boundaries"] == sorted(age["boundaries"])
    missing_row = next(
        row for row in first.artifact["rows"] if row["sample_id"] == "S0001"
    )
    assert missing_row["numeric"]["age"] == {"bin": age["missing_bin"], "missing": True}


def test_training_is_identical_for_serial_and_concurrent_execution() -> None:
    config_serial_raw = copy.deepcopy(DEMO_CONFIG)
    config_serial_raw["concurrency"]["max_workers"] = 1
    config_parallel_raw = copy.deepcopy(DEMO_CONFIG)
    config_parallel_raw["concurrency"]["max_workers"] = 4
    config_serial = validate_config(config_serial_raw)
    config_parallel = validate_config(config_parallel_raw)
    parsed = parse_samples(sample_lines(list(range(24))), config_serial)
    features = compute_features(config_serial, parsed.samples, row_cache={}).artifact
    serial = train_model(config_serial, parsed.train_samples, features).artifact
    parallel = train_model(config_parallel, parsed.train_samples, features).artifact
    assert serial["losses"] == parallel["losses"]
    assert serial["final_weights"] == parallel["final_weights"]
    assert serial["final_bias"] == parallel["final_bias"]
    assert serial["batch_order"] == parallel["batch_order"]


def test_quantization_changes_internal_integers_and_metrics(tmp_path: Path) -> None:
    runner, _ = new_runner(tmp_path, indices=list(range(24)))
    first = runner.process_current(batch_id="base")
    assert first.complete
    first_inference = first.stages["inference"]
    first_summary = first.stages["summary"]

    runner.raw_config["quantization"] = {
        "fractional_bits": 10,
        "scale_level": 8,
    }
    second = runner.process_current(batch_id="quantized")
    assert second.complete
    assert "features" in second.report["reuses"]
    assert "training" in second.report["reuses"]
    assert "inference" in second.report["invalidations"]
    assert "summary" in second.report["invalidations"]
    first_predictions = first_inference["predictions"]
    second_predictions = second.stages["inference"]["predictions"]
    assert [row["quantized_integer"] for row in first_predictions] != [
        row["quantized_integer"] for row in second_predictions
    ]
    assert first_summary["quantized_integer_sum"] != second.stages["summary"][
        "quantized_integer_sum"
    ]
    assert first_summary["log_loss"] != second.stages["summary"]["log_loss"]


def test_demo_edit_sequence_matches_full_rebuild_after_every_batch(tmp_path: Path) -> None:
    paths = generate_demo_workspace(tmp_path / "workspace", reset=True)
    store = SnapshotStore(paths["workspace"] / "snapshots")
    runner = PipelineRunner(
        snapshot_store=store,
        sample_store=SampleStore.from_path(paths["samples"]),
        raw_config=load_config_yaml(str(paths["config"])),
        pipeline_complete=False,
    )
    plan = load_edit_plan(paths["plan"])
    result = runner.run_plan(plan, verify_full=True)
    assert len(result.reports) == 4
    assert all(
        report["full_rebuild"]["equal"] is True for report in result.reports
    )
    assert all(report["pipeline_complete"] for report in result.reports)
    assert result.reports[-1]["budget"]["pending"] == []


def test_invalidation_edges_and_unrelated_config_reuse(tmp_path: Path) -> None:
    runner, _ = new_runner(tmp_path, indices=list(range(20)))
    base = runner.process_current(batch_id="base")
    assert base.complete

    label_plan = EditPlan(
        batches=[
            EditBatch(
                batch_id="label-only",
                operations=[
                    EditOperation(
                        index=0,
                        operation_type="correct_sample",
                        payload={
                            "type": "correct_sample",
                            "sample_id": "S0002",
                            "patch": {"label": 1},
                        },
                    )
                ],
            )
        ],
        failures=[],
        fingerprint="label-only",
    )
    label_result = runner.run_plan(label_plan)
    report = label_result.reports[0]
    assert "features" in report["reuses"]
    assert {"training", "inference", "summary"} <= set(report["invalidations"])

    config_plan = EditPlan(
        batches=[
            EditBatch(
                batch_id="concurrency-only",
                operations=[
                    EditOperation(
                        index=0,
                        operation_type="update_config",
                        payload={
                            "type": "update_config",
                            "patch": {"concurrency.max_workers": 2},
                        },
                    )
                ],
            )
        ],
        failures=[],
        fingerprint="concurrency-only",
    )
    config_result = runner.run_plan(config_plan)
    config_report = config_result.reports[0]
    assert config_report["invalidations"] == []
    assert config_report["reuses"] == ["features", "training", "inference", "summary"]
    assert config_report["budget"]["consumed"] == 0

    model_plan = EditPlan(
        batches=[
            EditBatch(
                batch_id="learning-rate-only",
                operations=[
                    EditOperation(
                        index=0,
                        operation_type="update_model_hyperparams",
                        payload={
                            "type": "update_model_hyperparams",
                            "patch": {"learning_rate": 0.07},
                        },
                    )
                ],
            )
        ],
        failures=[],
        fingerprint="learning-rate-only",
    )
    model_report = runner.run_plan(model_plan).reports[0]
    assert "features" in model_report["reuses"]
    assert {"training", "inference", "summary"} <= set(model_report["invalidations"])

    bins_plan = EditPlan(
        batches=[
            EditBatch(
                batch_id="bins-only",
                operations=[
                    EditOperation(
                        index=0,
                        operation_type="update_feature_params",
                        payload={"type": "update_feature_params", "patch": {"bins": 5}},
                    )
                ],
            )
        ],
        failures=[],
        fingerprint="bins-only",
    )
    bins_report = runner.run_plan(bins_plan).reports[0]
    assert bins_report["invalidations"] == [
        "features",
        "training",
        "inference",
        "summary",
    ]


def test_budget_pending_is_serial_concurrent_equivalent(tmp_path: Path) -> None:
    artifacts = {}
    budget_sequences = {}
    pending_sequences = {}
    for workers in (1, 4):
        config = copy.deepcopy(DEMO_CONFIG)
        config["concurrency"]["max_workers"] = workers
        config["concurrency"]["batch_budget"] = 3
        runner, store = new_runner(
            tmp_path / f"workers-{workers}", config=config, indices=list(range(8))
        )
        sequences = []
        pendings = []
        for _ in range(3):
            outcome = runner.process_current(batch_id=f"step-{len(sequences)}")
            sequences.append(outcome.report["budget"]["sequence"])
            pendings.append(outcome.report["pending"])
            runner._commit()
            if not outcome.complete:
                reloaded = PipelineRunner.from_snapshot(store)
                assert reloaded is not None
                runner = reloaded
        assert runner.pipeline_complete
        assert runner.stages
        artifacts[workers] = runner.stages
        budget_sequences[workers] = sequences
        pending_sequences[workers] = pendings
    assert budget_sequences[1] == budget_sequences[4]
    assert pending_sequences[1] == pending_sequences[4]
    assert pending_sequences[1][0] == [
        "feature:S0004",
        "feature:S0005",
        "feature:S0006",
        "feature:S0007",
        "feature:S0008",
    ]
    assert artifacts[1] == artifacts[4]


def test_snapshot_stale_detection_restore_and_commit_interval(tmp_path: Path) -> None:
    paths = generate_demo_workspace(tmp_path / "workspace", reset=True)
    config = load_config_yaml(str(paths["config"]))
    config["snapshot"]["commit_interval_batches"] = 2
    store = SnapshotStore(paths["workspace"] / "snapshots")
    runner = PipelineRunner(
        snapshot_store=store,
        sample_store=SampleStore.from_path(paths["samples"]),
        raw_config=config,
        pipeline_complete=False,
    )
    result = runner.run_plan(load_edit_plan(paths["plan"]))
    assert result.manifest is not None
    versions = store.list_versions()
    assert len(versions) == 2
    assert versions[-1]["global_version"] == 2
    with pytest.raises(SnapshotStaleError) as stale_version:
        store.validate_resume(expected_version=99)
    assert stale_version.value.code == "E_SNAPSHOT_STALE"
    with pytest.raises(SnapshotStaleError) as stale_config:
        store.validate_resume(expected_config_hash="not-the-hash")
    assert stale_config.value.code == "E_SNAPSHOT_STALE"
    restored = store.restore(1)
    assert restored["global_version"] == 1
    current = store.load_current()
    assert current is not None
    assert current.manifest["global_version"] == 1


def test_unchanged_rerun_does_not_create_a_new_snapshot_version(tmp_path: Path) -> None:
    runner, store = new_runner(tmp_path, indices=list(range(8)))
    first = runner.process_current(batch_id="base")
    assert first.complete
    first_manifest = runner._commit()
    second = runner.process_current(batch_id="rerun")
    assert all(stage in second.report["reuses"] for stage in ("features", "training", "inference", "summary"))
    second_manifest = runner._commit()
    assert second_manifest["global_version"] == first_manifest["global_version"]
    assert len(store.list_versions()) == 1


def test_invalid_sample_failures_are_identical_to_full_rebuild(tmp_path: Path) -> None:
    config = copy.deepcopy(DEMO_CONFIG)
    valid_lines = sample_lines(list(range(12)))
    invalid_lines = [
        json.dumps(make_sample(20) | {"region": "outside"}, sort_keys=True),
        json.dumps(make_sample(21) | {"age": "not-finite"}, sort_keys=True),
    ]
    store = SnapshotStore(tmp_path / "snapshots")
    runner = PipelineRunner(
        snapshot_store=store,
        sample_store=SampleStore.from_lines(valid_lines + invalid_lines),
        raw_config=config,
        pipeline_complete=False,
    )
    plan = EditPlan(
        batches=[
            EditBatch(
                batch_id="bins",
                operations=[
                    EditOperation(
                        index=0,
                        operation_type="update_feature_params",
                        payload={"type": "update_feature_params", "patch": {"bins": 6}},
                    )
                ],
            )
        ],
        failures=[],
        fingerprint="invalid-parity",
    )
    result = runner.run_plan(plan, verify_full=True)
    report = result.reports[0]
    assert report["full_rebuild"]["equal"] is True
    codes = {
        code for failure in report["data_failures"] for code in failure["codes"]
    }
    assert codes == {"E_CATEGORY_DOMAIN", "E_NUM_NONFINITE"}


def test_yaml_subset_fallback_and_nonfinite_entries_do_not_abort() -> None:
    from domain_pipeline.config import dump_config_yaml

    assert compat_safe_load(dump_config_yaml(DEMO_CONFIG)) == DEMO_CONFIG
    config = validate_config(copy.deepcopy(DEMO_CONFIG))
    lines = sample_lines([0, 1])
    lines.append(
        json.dumps(make_sample(5) | {"income": float("nan")}, allow_nan=True)
    )
    store = SampleStore.from_lines(lines)
    store.append_records([make_sample(6) | {"age": float("inf")}])
    parsed = parse_samples(store.lines, config)
    failures = state_failures(config.failures, parsed.failures)
    assert [sample.sample_id for sample in parsed.samples] == ["S0001", "S0002"]
    assert {code for failure in failures for code in failure["codes"]} == {
        "E_NUM_NONFINITE"
    }
