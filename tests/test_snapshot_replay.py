from __future__ import annotations

import json
import os
import stat

import pytest

from slowprof import (
    E_REPLAY,
    E_VERSION,
    SlowProfError,
    load_snapshot,
    profile_file,
    replay_snapshot,
    restore_snapshot,
    save_snapshot,
)


def test_snapshot_roundtrip_contains_required_sections(tmp_path, samples_dir, policy, small_interval):
    session = profile_file(samples_dir / "small.ndjson", policy, small_interval)
    path = tmp_path / "session.json"
    save_snapshot(session, path)
    document = load_snapshot(path)
    assert document["metric_version"] == session.policy.metric_version
    assert document["interval"]["interval_version"] == "obs-v1"
    assert document["folded_result"]["frames"] == [
        item.to_dict() for item in session.aggregation.frame_stats
    ]
    assert document["hotspot_ranking"] == session.ranking_report["ranking"]
    assert document["error_annotation"] == session.ranking_report["error_annotation"]
    assert document["sample"]["file_sha256"] == session.metadata.file_sha256
    assert document["aggregation"]["path"] == "full"


def test_missing_metric_version_is_e_version(samples_dir):
    with pytest.raises(SlowProfError) as raised:
        load_snapshot(samples_dir / "anomalies" / "snapshot_missing_version.json")
    assert raised.value.code == E_VERSION
    assert raised.value.field == "metric_version"


def test_consistent_restore_uses_snapshot_state(tmp_path, samples_dir, policy, small_interval):
    session = profile_file(samples_dir / "small.ndjson", policy, small_interval)
    path = tmp_path / "session.json"
    save_snapshot(session, path)
    outcome = restore_snapshot(path, session)
    assert outcome.strategy == "restore"
    assert outcome.mismatches == ()
    assert outcome.archived_snapshot_path is None
    assert outcome.active_session.conclusion_sha256 == session.conclusion_sha256


def test_mismatched_restore_replaces_whole_state_and_archives_old_snapshot(
    tmp_path, samples_dir, policy, small_interval, changed_interval
):
    old_session = profile_file(samples_dir / "small.ndjson", policy, small_interval)
    current_session = profile_file(samples_dir / "small.ndjson", policy, changed_interval)
    path = tmp_path / "old-session.json"
    save_snapshot(old_session, path)
    old_bytes = path.read_bytes()

    outcome = restore_snapshot(path, current_session)
    assert outcome.strategy == "replace"
    assert outcome.active_session is current_session
    assert outcome.active_session.conclusion_sha256 == current_session.conclusion_sha256
    fields = {item["field"] for item in outcome.mismatches}
    assert "interval.start_ms" in fields
    assert "interval.interval_version" in fields
    assert "conclusion_sha256" in fields

    changed_names = [item["frame"] for item in outcome.changed_frames]
    assert changed_names == sorted(changed_names)
    assert "sql.execute" in changed_names

    moves = {item["frame"]: item for item in outcome.rank_moves}
    assert moves["cache.get"]["snapshot_rank"] == 4
    assert moves["cache.get"]["current_rank"] is None
    assert moves["redis.get"]["snapshot_rank"] is None
    assert moves["redis.get"]["current_rank"] == 5

    assert outcome.archived_snapshot_path is not None
    archived = outcome.archived_snapshot_path
    assert open(archived, "rb").read() == old_bytes
    write_bits = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
    assert (os.stat(archived).st_mode & write_bits) == 0
    assert outcome.archived_snapshot_read_only is True

    replacement = tmp_path / "replacement.json"
    save_snapshot(outcome.active_session, replacement)
    assert load_snapshot(replacement)["interval"]["interval_version"] == "obs-narrow-v2"


def test_replay_success(tmp_path, samples_dir, policy, large_interval):
    session = profile_file(samples_dir / "large.ndjson", policy, large_interval)
    path = tmp_path / "large.json"
    save_snapshot(session, path)
    outcome = replay_snapshot(path, samples_dir / "large.ndjson")
    assert outcome.report["status"] == "reproducible"
    assert outcome.report["replay_key"]["aggregation_path"] == "streaming"
    assert outcome.session.conclusion_sha256 == session.conclusion_sha256


def test_replay_reports_first_frame_field_and_expected_value(
    tmp_path, samples_dir, policy, small_interval
):
    session = profile_file(samples_dir / "small.ndjson", policy, small_interval)
    path = tmp_path / "good.json"
    save_snapshot(session, path)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["folded_result"]["frames"][0]["self_time_ms"] = 1
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")

    with pytest.raises(SlowProfError) as raised:
        replay_snapshot(tampered, samples_dir / "small.ndjson")
    error = raised.value
    assert error.code == E_REPLAY
    first = error.details["first_mismatch"]
    assert first["frame"] == "api.handle"
    assert first["field"] == "self_time_ms"
    assert first["expected"] == 1
    assert first["actual"] == 0


def test_replay_reports_sample_digest_mismatch(tmp_path, samples_dir, policy, small_interval):
    session = profile_file(samples_dir / "small.ndjson", policy, small_interval)
    path = tmp_path / "small.json"
    save_snapshot(session, path)
    with pytest.raises(SlowProfError) as raised:
        replay_snapshot(path, samples_dir / "large.ndjson")
    assert raised.value.code == E_REPLAY
    assert raised.value.details["mismatch"] == "sample_sha256"
    assert raised.value.details["expected"] == session.metadata.file_sha256


def test_replay_reports_metric_version_mismatch(tmp_path, samples_dir, policy, small_interval):
    session = profile_file(samples_dir / "small.ndjson", policy, small_interval)
    path = tmp_path / "small.json"
    save_snapshot(session, path)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["metric_version"] = "older-metric-v0"
    path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
    with pytest.raises(SlowProfError) as raised:
        replay_snapshot(path, samples_dir / "small.ndjson")
    assert raised.value.code == E_REPLAY
    assert raised.value.details["mismatch"] == "metric_version"


def test_replay_reports_interval_version_mismatch(tmp_path, samples_dir, policy, small_interval):
    session = profile_file(samples_dir / "small.ndjson", policy, small_interval)
    path = tmp_path / "small.json"
    save_snapshot(session, path)
    with pytest.raises(SlowProfError) as raised:
        replay_snapshot(
            path,
            samples_dir / "small.ndjson",
            current_interval_version="obs-v2",
        )
    assert raised.value.code == E_REPLAY
    assert raised.value.details["mismatch"] == "interval_version"

