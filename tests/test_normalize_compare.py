from __future__ import annotations

import copy
import json

import pytest

from ckptmeta.errors import InteropError
from ckptmeta.formats import decode_manifest
from ckptmeta.ledger import write_direct_manifest
from ckptmeta.normalize import (
    compare_manifests,
    compare_upgrade_paths,
    compare_views,
    normalize_manifest,
    upgrade_manifest_to_current,
)
from ckptmeta.view import view_bytes, view_digest


def test_v1_v2_v3_normalize_to_identical_bytes(fixtures, loaded):
    config, ledger = loaded
    v1 = normalize_manifest(decode_manifest(fixtures.v1_target), ledger)
    v2 = normalize_manifest(decode_manifest(fixtures.v2_target), ledger)
    v3 = normalize_manifest(decode_manifest(fixtures.direct_target), ledger)
    truth = ledger.entry(fixtures.target_step)
    assert v1.view_bytes == v2.view_bytes == v3.view_bytes == view_bytes(truth.view)
    assert v1.view_digest == v2.view_digest == v3.view_digest == truth.view_digest
    assert v1.report["source_format_version"] == 1
    assert v2.report["source_format_version"] == 2
    assert v3.report["source_format_version"] == 3
    assert "shards[*].momentum_digest" in v1.report["filled_fields"]
    assert "rng_state" in v1.report["filled_fields"]
    assert "shards[*].data_cursor" in v2.report["filled_fields"]
    assert v3.report["filled_fields"] == []
    assert (v1.report["verified_digest_count"], v2.report["verified_digest_count"], v3.report["verified_digest_count"]) == (4, 8, 8)
    assert all(report.report["whole_view_replaced"] for report in (v1, v2, v3))


def test_upgrade_paths_compare_equal(fixtures, loaded):
    _, ledger = loaded
    report = compare_upgrade_paths(
        decode_manifest(fixtures.v1_target),
        decode_manifest(fixtures.v2_target),
        decode_manifest(fixtures.direct_target),
        ledger,
    )
    assert report["byte_identical"] is True
    assert report["filled_fields"]["v1"]
    assert report["filled_fields"]["v2"]


def test_direct_write_api_matches_normalization(tmp_path, fixtures, loaded):
    _, ledger = loaded
    path = tmp_path / "direct.json"
    manifest = write_direct_manifest(path, ledger, fixtures.target_step)
    normalized = normalize_manifest(manifest, ledger, manifest_path=path)
    assert normalized.view_bytes == view_bytes(ledger.entry(fixtures.target_step).view)


def test_upgrade_manifest_api_reports_filled_fields(fixtures, loaded):
    _, ledger = loaded
    upgraded, filled, verified = upgrade_manifest_to_current(decode_manifest(fixtures.v1_target), ledger)
    assert upgraded.version == 3
    assert "rng_state" in filled
    assert verified == 4
    assert upgraded.to_view() == ledger.entry(fixtures.target_step).view


def test_restore_mismatch_reports_first_shard_and_field(fixtures, loaded):
    _, ledger = loaded
    expected = copy.deepcopy(ledger.entry(fixtures.target_step).view)
    expected["shards"][2]["parameter_digest"] = "f" * 64
    with pytest.raises(InteropError) as raised:
        normalize_manifest(decode_manifest(fixtures.v1_target), ledger, expected_view=expected)
    assert raised.value.code == "E_RESTORE"
    assert raised.value.shard_index == 2
    assert raised.value.details["first_mismatch"]["field"] == "parameter_digest"
    assert raised.value.details["first_mismatch"]["global_step"] == fixtures.target_step


def test_diverge_reports_first_shard_field_and_step(loaded):
    _, ledger = loaded
    left = copy.deepcopy(ledger.entry(20).view)
    right = copy.deepcopy(left)
    right["shards"][1]["data_cursor"]["offset"] += 1
    with pytest.raises(InteropError) as raised:
        compare_views(left, right)
    assert raised.value.code == "E_DIVERGE"
    assert raised.value.shard_index == 1
    assert raised.value.details["first_mismatch"]["field"] == "data_cursor.offset"
    assert raised.value.details["first_mismatch"]["global_step"] == 20


def test_integrity_failure_before_normalization(fixtures, loaded):
    _, ledger = loaded
    raw = json.loads(fixtures.v1_target.read_text(encoding="utf-8"))
    raw["shards"][3]["parameter_digest"] = "0" * 64
    from ckptmeta.formats import decode_manifest_data

    with pytest.raises(InteropError) as raised:
        normalize_manifest(decode_manifest_data(raw), ledger, manifest_path=fixtures.v1_target)
    assert raised.value.code == "E_INTEGRITY"
    assert raised.value.shard_index == 3
