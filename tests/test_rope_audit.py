from __future__ import annotations

import io
import json
from contextlib import redirect_stderr
from pathlib import Path

import pytest

from rope_audit.cli import build_fixture_commands, build_fixture_config, run_until_drained
from rope_audit.commands import apply_command
from rope_audit.config import parse_config
from rope_audit.engine import AuditEngine
from rope_audit.injections import InjectionHub
from rope_audit.pipeline import error_summary, fixed_position_encoding
from rope_audit.samples import generate_samples, validate_sample
from rope_audit.snapshot import SnapshotStore


def make_config(tmp_path: Path, **budget) -> dict:
    raw = build_fixture_config()
    raw["snapshot_dir"] = str(tmp_path / "snapshots")
    raw["budget"].update(budget)
    return parse_config(raw)


def make_world(tmp_path: Path, count: int = 6):
    config = make_config(tmp_path, batch_point_budget=10_000, sample_quota_points=10_000)
    samples = generate_samples(config, count=count)
    return config, samples


def payload(engine: AuditEngine) -> str:
    return json.dumps(engine.comparison_payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def test_config_validation_reports_fields():
    base = build_fixture_config()
    bad_tolerance = json.loads(json.dumps(base))
    bad_tolerance["tolerances"]["dot"] = 0
    with pytest.raises(Exception) as exc:
        parse_config(bad_tolerance, json.dumps(bad_tolerance, indent=2))
    assert exc.value.code == "E_CONFIG_INVALID"
    assert exc.value.field == "tolerances.dot"
    assert exc.value.line is not None

    bad_quant = json.loads(json.dumps(base))
    bad_quant["fixed_point"]["quant_level"] = 9
    with pytest.raises(Exception) as exc:
        parse_config(bad_quant, json.dumps(bad_quant, indent=2))
    assert "quant_level" in exc.value.field
    assert exc.value.line is not None

    bad_scheme = json.loads(json.dumps(base))
    bad_scheme["schemes"][0]["block_size"] = 3
    with pytest.raises(Exception) as exc:
        parse_config(bad_scheme)
    assert exc.value.field == "schemes[0].block_size"


def test_sample_validation_line_and_rules():
    dimension = 8
    sample = {
        "id": "s1",
        "query_position": 4,
        "train_length": 12,
        "key_positions": [1, 3, 3],
        "query_vector": [0.0] * dimension,
        "key_vectors": [[0.0] * dimension for _ in range(3)],
    }
    with pytest.raises(Exception) as exc:
        validate_sample(sample, dimension, line=7)
    assert exc.value.line == 7
    assert exc.value.field == "key_positions"

    sample["key_positions"] = [1, 3, 5]
    sample["query_vector"] = [float("nan")] * dimension
    with pytest.raises(Exception) as exc:
        validate_sample(sample, dimension, line=8)
    assert "finite" in str(exc.value)


def test_quant_bit_width_and_level_change_integers(tmp_path):
    config, samples = make_world(tmp_path, count=1)
    sample = samples[0]
    scheme = config["schemes"][0]
    fine = fixed_position_encoding(sample, scheme, config["fixed_point"], config["dimension"])

    coarse_config = json.loads(json.dumps(config))
    coarse_config["fixed_point"]["quant_level"] = 3
    coarse_config["fixed_point"]["effective_frac_bits"] = 11
    coarse = fixed_position_encoding(sample, scheme, coarse_config["fixed_point"], config["dimension"])

    fine_int = fine["query_row"][0]["angle_int"]
    coarse_int = coarse["query_row"][0]["angle_int"]
    assert fine_int != coarse_int
    fine_residuals = [item["quant_trace"]["residual"] for item in fine["query_row"]]
    coarse_residuals = [item["quant_trace"]["residual"] for item in coarse["query_row"]]
    assert any(a != b for a, b in zip(fine_residuals, coarse_residuals))

    fewer_bits = json.loads(json.dumps(config))
    fewer_bits["fixed_point"]["frac_bits"] = 16
    fewer_bits["fixed_point"]["effective_frac_bits"] = 13
    bits_result = fixed_position_encoding(sample, scheme, fewer_bits["fixed_point"], config["dimension"])
    assert bits_result["query_row"][1]["angle_int"] != fine["query_row"][1]["angle_int"]


def test_zero_truth_relative_error_uses_absolute_branch():
    metric = error_summary([0.25, 0.0], [0.25, 0.0])
    assert metric["zero_truth_branch"] is True
    assert metric["relative_error_branch"] == "truth_zero_uses_absolute"
    assert metric["rel_error"] == 0.0

    metric = error_summary([0.5], [0.0])
    assert metric["rel_error"] == 0.5


def test_serial_threaded_and_incremental_rebuild_match(tmp_path):
    config, samples = make_world(tmp_path, count=6)
    commands = build_fixture_commands(samples)
    # The fixture command deletes s006 and exercises every required edit class.
    incremental = AuditEngine(json.loads(json.dumps(config)), json.loads(json.dumps(samples)))
    current_config = config
    current_samples = samples
    for raw_command in commands:
        current_config, current_samples, _ = apply_command(current_config, current_samples, dict(raw_command))
        incremental.replace_state(current_config, current_samples)
        run_until_drained(incremental, True, None, 2)

    serial = AuditEngine(json.loads(json.dumps(current_config)), json.loads(json.dumps(current_samples)))
    run_until_drained(serial, True, None, 1)
    threaded = AuditEngine(json.loads(json.dumps(current_config)), json.loads(json.dumps(current_samples)))
    run_until_drained(threaded, True, None, 4)

    assert payload(incremental) == payload(serial)
    assert payload(threaded) == payload(serial)
    assert threaded.ledger == serial.ledger


def test_precise_invalidation_scheme_vector_and_length(tmp_path):
    config, samples = make_world(tmp_path, count=3)
    engine = AuditEngine(config, samples)
    engine.run_batch(workers=1)

    changed_config = json.loads(json.dumps(config))
    changed_config["schemes"][0]["base_freq"] = 14444.0
    changed_engine_config = parse_config(changed_config)
    engine.replace_state(changed_engine_config, samples)
    report = engine.run_batch(workers=1)
    entries = report["run_ledger"]
    affected_position = [
        e for e in entries if e["scheme_id"] == "rope_base" and e["stage"] == "position" and e["sample_id"]
    ]
    unaffected_position = [
        e for e in entries if e["scheme_id"] == "rope_flash" and e["stage"] == "position" and e["sample_id"]
    ]
    assert all(e["status"] == "miss" for e in affected_position)
    assert any("scheme_parameters_changed" in e["basis"] for e in affected_position)
    assert all(e["status"] == "hit" for e in unaffected_position)

    replacement = json.loads(json.dumps(samples[1]))
    replacement["query_vector"][0] += 0.2
    changed_samples = [replacement if s["id"] == replacement["id"] else s for s in samples]
    engine.replace_state(engine.config, changed_samples)
    report = engine.run_batch(workers=1)
    changed_entries = [e for e in report["run_ledger"] if e["sample_id"] == replacement["id"] and e["scheme_id"] == "rope_base"]
    by_stage = {e["stage"]: e["status"] for e in changed_entries}
    assert by_stage["position"] == "hit"
    assert by_stage["rotation"] == "miss"
    assert by_stage["dot"] == "miss"
    assert by_stage["normalization"] == "miss"

    length_config = json.loads(json.dumps(engine.config))
    length_config["train_length"] = 18
    length_config["target_length"] = 64
    length_config = parse_config({**build_fixture_config(), **length_config}) if False else engine.config
    length_config["train_length"] = 18
    length_config["target_length"] = 64
    for sample in engine.samples:
        sample["train_length"] = 18
    engine.replace_state(length_config, engine.samples)
    report = engine.run_batch(workers=1)
    low_level = [e for e in report["run_ledger"] if e["stage"] in {"position", "rotation", "dot", "normalization"}]
    summaries = [e for e in report["run_ledger"] if e["stage"] == "summary"]
    assert low_level and all(e["status"] == "hit" for e in low_level)
    assert summaries and all(e["status"] == "miss" for e in summaries)


def test_budget_pending_priority_and_drain_equivalence(tmp_path):
    config, samples = make_world(tmp_path, count=6)
    full = AuditEngine(json.loads(json.dumps(config)), json.loads(json.dumps(samples)))
    full.config["budget"]["batch_point_budget"] = 10_000
    full.run_batch(workers=1)
    expected = payload(full)

    for budget in (48, 95, 96, 143, 144, 240):
        engine = AuditEngine(json.loads(json.dumps(config)), json.loads(json.dumps(samples)))
        engine.config["budget"]["batch_point_budget"] = budget
        first = engine.run_batch(workers=1)
        if budget < 288:
            assert first["budget"]["pending_sample_ids"]
            assert first["budget"]["plan"]["pending"][0]["code"] == "E_POINT_BUDGET_PENDING"
        for _ in range(20):
            if not engine.pending:
                break
            engine.run_batch(workers=1)
        assert engine.pending == []
        assert payload(engine) == expected


def test_quota_exceeded_does_not_block_other_budget_work(tmp_path):
    config, samples = make_world(tmp_path, count=6)
    engine = AuditEngine(config, samples)
    report = engine.run_batch(batch_budget=10_000, sample_quota_points=40, workers=1)
    assert len(report["failures"]) == len(samples)
    assert all(f["code"] == "E_POINT_BUDGET_EXCEEDED" for f in report["failures"])


def test_injection_hit_stderr_and_isolated_failure(tmp_path):
    config, samples = make_world(tmp_path, count=3)
    injection = {
        "id": "INJ-TEST-1",
        "sample_id": samples[1]["id"],
        "scheme_id": "rope_base",
        "stage": "rotation",
        "fault_type": "nonfinite_feature",
        "code": "E_TEST_INJECTED",
    }
    hub = InjectionHub([injection])
    engine = AuditEngine(config, samples, hub)
    stderr = io.StringIO()
    with redirect_stderr(stderr):
        report = engine.run_batch(workers=1)
    assert "INJ-TEST-1" in stderr.getvalue()
    failures = [f for f in report["failures"] if f["code"] == "E_TEST_INJECTED"]
    assert len(failures) == 1
    assert failures[0]["stage"] == "rotation"
    ok_cells = [
        item
        for sample in report["samples"]
        for item in sample["schemes"]
        if item["status"] == "ok"
    ]
    assert len(ok_cells) == len(samples) * len(config["schemes"]) - 1


def test_snapshot_atomic_commit_recovery_and_config_hash(tmp_path):
    config, samples = make_world(tmp_path, count=2)
    engine = AuditEngine(config, samples)
    engine.run_batch(workers=1)
    store = SnapshotStore(tmp_path / "snapshot")
    committed = store.commit(engine, engine.last_report)
    assert Path(committed["target"], "COMMITTED").exists()

    orphan = store.root / "tmp-stage-crashed"
    orphan.mkdir()
    (orphan / "state.json").write_text("{}", encoding="utf-8")
    recovery = store.recover(engine.config_hash)
    assert recovery["status"] == "current_snapshot_ready"
    assert recovery["orphaned_temp_dirs_removed"] == [str(orphan)]
    assert not orphan.exists()

    store.current_pointer.unlink()
    recovery = store.recover(engine.config_hash)
    assert recovery["status"] == "current_snapshot_ready"
    assert recovery["global_version"] == 1

    mismatch = store.recover("different-hash")
    assert mismatch["status"] == "config_hash_mismatch_cold_rebuild"
    assert store.load_state("different-hash") is None
    state = store.load_state(engine.config_hash)
    assert state is not None
    assert len(state["samples"]) == 2
