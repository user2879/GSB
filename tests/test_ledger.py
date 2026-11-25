"""Requirement 8: transactional batches, snapshot wholesale replace,
forced-kill recovery, first-mismatch localization."""

import os
import subprocess
import sys

import pytest

from obs_pipeline import gen
from obs_pipeline.ledger import FORMAT_VERSION, Ledger
from obs_pipeline.pipeline import commit_report, content_digest, run_pipeline
from obs_pipeline.spec import Window
from obs_pipeline.stages import run_stage


@pytest.fixture()
def ledger(tmp_path):
    ledger = Ledger(str(tmp_path / "ledger.sqlite"))
    yield ledger
    ledger.close()


def _small_report(spec, ledger, records):
    return run_pipeline(spec, ledger, records, global_budget=10 ** 6)


def test_batch_commit_atomic_and_versioned(ledger, sample_spec, tmp_path):
    records = gen.generate_source_records  # noqa: F821 - clarity
    records_by_source = {s.id: gen.generate_source_records(s) for s in sample_spec.sources}
    digest = content_digest(sample_spec, records_by_source)
    assert ledger.open_snapshot(sample_spec.spec_version, digest) is False
    assert ledger.open_snapshot(sample_spec.spec_version, digest) is True
    report = _small_report(sample_spec, ledger, records_by_source)
    batch_id = commit_report(ledger, report)
    rows = ledger.conn.execute(
        "SELECT spec_version, format_version, content_digest, committed FROM batches"
        " WHERE batch_id = ?", (batch_id,)).fetchone()
    assert rows[0] == sample_spec.spec_version
    assert rows[1] == FORMAT_VERSION
    assert rows[2] == digest
    assert rows[3] == 1
    assert ledger.load_cursor() is not None
    assert ledger.load_error_attributions(spec_version=1)


def test_snapshot_mismatch_replaces_wholesale(ledger, sample_spec):
    records_by_source = {s.id: gen.generate_source_records(s) for s in sample_spec.sources}
    digest = content_digest(sample_spec, records_by_source)
    ledger.open_snapshot(sample_spec.spec_version, digest)
    report = _small_report(sample_spec, ledger, records_by_source)
    commit_report(ledger, report)
    assert ledger.load_stage_summaries()
    # Reopen with a different identity: state must be wiped, not merged.
    assert ledger.open_snapshot(sample_spec.spec_version + 1, "0" * 64) is False
    assert ledger.load_stage_summaries() == {}
    assert ledger.load_metric_stats() == {}
    assert ledger.load_cursor() is None
    # Error attribution history survives for version comparison.
    assert ledger.load_error_attributions(spec_version=1)


def test_forced_kill_leaves_unknown_batch_and_replay_matches(ledger, sample_spec, tmp_path):
    records_by_source = {s.id: gen.generate_source_records(s) for s in sample_spec.sources}
    digest = content_digest(sample_spec, records_by_source)
    ledger.open_snapshot(sample_spec.spec_version, digest)
    report = _small_report(sample_spec, ledger, records_by_source)
    commit_report(ledger, report)
    path = ledger.path
    ledger.close()

    proc = subprocess.run(
        [sys.executable, "tools/crash_writer.py", path, "1", digest],
        capture_output=True,
    )
    assert proc.returncode == 137  # killed, no exit report produced

    ledger = Ledger(path)
    status = ledger.recovery_status()
    assert status["committed_batches"] == [report.batch_id]
    assert status["unknown_batches"], "uncommitted batch must be marked unknown"
    assert status["requires_rerun"]
    # Partial uncommitted rows were rolled back.
    assert all(k[0] != "partial" for k in ledger.load_stage_summaries())

    replay = _small_report(sample_spec, ledger, records_by_source)
    assert ledger.verify_stage_summaries(replay.stage_results) is None
    ledger.close()


def test_verify_locates_first_mismatch(ledger, sample_spec):
    records_by_source = {s.id: gen.generate_source_records(s) for s in sample_spec.sources}
    ledger.open_snapshot(sample_spec.spec_version, content_digest(sample_spec, records_by_source))
    report = _small_report(sample_spec, ledger, records_by_source)
    commit_report(ledger, report)
    # Corrupt one stored summary, then locate it.
    ledger.conn.execute(
        "UPDATE stage_summaries SET summary = 'tampered' WHERE stage_id = 's_avg' AND window_id = 'w1'"
    )
    ledger.conn.commit()
    replay = _small_report(sample_spec, ledger, records_by_source)
    mismatch = ledger.verify_stage_summaries(replay.stage_results)
    assert mismatch is not None
    stage_id, window_id, field, old, new = mismatch
    assert (stage_id, window_id, field) == ("s_avg", "w1", "summary")
    assert old == "tampered"
