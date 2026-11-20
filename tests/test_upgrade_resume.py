from __future__ import annotations

import json
import os
import shutil
import stat

import pytest

from ckptmeta.canonical import pretty_json_bytes, sha256_hex
from ckptmeta.catalog import Catalog, ForcedTermination, scan_catalog
from ckptmeta.errors import InteropError
from ckptmeta.formats import decode_manifest


class StopAfterCheckpoint(BaseException):
    pass


def _hook(phase: str, item: dict):
    if phase == "after_checkpoint":
        raise StopAfterCheckpoint(phase)


def test_upgrade_rewrites_atomically_and_keeps_readonly_backup(fixtures, loaded):
    config, ledger = loaded
    catalog = Catalog(fixtures.historical_v1, config, ledger)
    report = catalog.open()
    assert report.upgrade["status"] == "complete"
    assert report.upgrade["completed"] == 1
    assert report.format_versions == (3,)
    backup = fixtures.historical_v1 / ".ckptmeta" / "backups" / "step-00000020.v1.json.v1.bak"
    assert backup.exists()
    assert stat.S_IMODE(backup.stat().st_mode) & stat.S_IWRITE == 0
    assert decode_manifest(backup).version == 1
    assert decode_manifest(fixtures.v1_target).version == 3
    assert catalog.checkpoint_path.exists()

    second = catalog.open()
    assert second.upgrade["processed"] == []
    assert second.upgrade["skipped_done"] == ["step-00000020.v1.json"]


def test_forced_stop_after_checkpoint_resumes_without_reprocessing(fixtures, loaded):
    config, ledger = loaded
    catalog = Catalog(fixtures.crash_catalog, config, ledger)
    with pytest.raises(StopAfterCheckpoint):
        catalog.open(fault_hook=_hook)
    first_path = fixtures.crash_catalog / "step-00000010.v1.json"
    first_digest = sha256_hex(first_path.read_bytes())
    versions = {manifest.source_path.name: manifest.version for manifest in scan_catalog(fixtures.crash_catalog)}
    assert versions["step-00000010.v1.json"] == 3
    assert versions["step-00000020.v2.json"] == 2

    resumed = catalog.open()
    assert sha256_hex(first_path.read_bytes()) == first_digest
    assert resumed.upgrade["status"] == "resumed_complete"
    assert resumed.upgrade["skipped_done"] == ["step-00000010.v1.json"]
    assert [item["manifest_path"] for item in resumed.upgrade["processed"]] == ["step-00000020.v2.json"]
    assert all(manifest.version == 3 for manifest in scan_catalog(fixtures.crash_catalog))


def test_rewrite_window_is_unknown_and_requires_explicit_retry(fixtures, loaded):
    config, ledger = loaded
    catalog = Catalog(fixtures.crash_catalog, config, ledger)

    def stop_after_rewrite(phase: str, item: dict):
        if phase == "after_rewrite":
            raise StopAfterCheckpoint(phase)

    with pytest.raises(StopAfterCheckpoint):
        catalog.open(fault_hook=stop_after_rewrite)

    with pytest.raises(InteropError) as raised:
        catalog.open()
    assert raised.value.code == "E_UPGRADE"
    assert raised.value.details["requires_retry"] is True
    checkpoint = json.loads(catalog.checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint["items"][0]["status"] == "unknown"
    backup = fixtures.crash_catalog / checkpoint["items"][0]["backup_path"]
    assert backup.exists()
    assert decode_manifest(backup).version == 1

    resumed = catalog.open(retry_unknown=True)
    assert resumed.upgrade["completed"] == 2
    assert all(manifest.version == 3 for manifest in scan_catalog(fixtures.crash_catalog))


def test_checkpoint_content_mismatch_is_upgrade_error_and_preserves_manifests(fixtures, loaded):
    config, ledger = loaded
    catalog = Catalog(fixtures.crash_catalog, config, ledger)
    catalog.open()
    before = {
        manifest.source_path.name: sha256_hex(manifest.source_path.read_bytes())
        for manifest in scan_catalog(fixtures.crash_catalog)
    }
    checkpoint = json.loads(catalog.checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["items"][0]["source_sha256"] = "0" * 64
    catalog.checkpoint_path.write_bytes(pretty_json_bytes(checkpoint))

    with pytest.raises(InteropError) as raised:
        catalog.open()
    assert raised.value.code == "E_UPGRADE"
    after = {
        manifest.source_path.name: sha256_hex(manifest.source_path.read_bytes())
        for manifest in scan_catalog(fixtures.crash_catalog)
    }
    assert after == before


def test_chain_missing_parent_is_chain_error(fixtures, loaded):
    config, ledger = loaded
    raw = json.loads(fixtures.direct_target.read_text(encoding="utf-8"))
    raw["parent_manifest"]["manifest_id"] = "step-missing"
    fixtures.direct_target.write_bytes(pretty_json_bytes(raw))
    with pytest.raises(InteropError) as raised:
        Catalog(fixtures.direct, config, ledger).open()
    assert raised.value.code == "E_CHAIN"
    assert raised.value.path == "parent_manifest.manifest_id"


def test_self_parent_reference_is_chain_error(fixtures, loaded):
    config, ledger = loaded
    raw = json.loads(fixtures.direct_target.read_text(encoding="utf-8"))
    raw["parent_manifest"]["manifest_id"] = raw["manifest_id"]
    fixtures.direct_target.write_bytes(pretty_json_bytes(raw))
    with pytest.raises(InteropError) as raised:
        Catalog(fixtures.direct, config, ledger).open()
    assert raised.value.code == "E_CHAIN"
    assert raised.value.path == "parent_manifest.manifest_id"


def test_integrity_error_prevents_any_rewrite(fixtures, loaded):
    config, ledger = loaded
    before_other = (fixtures.historical_v1 / "step-00000010.v3.json").read_bytes()
    raw = json.loads(fixtures.v1_target.read_text(encoding="utf-8"))
    raw["shards"][2]["parameter_digest"] = "0" * 64
    fixtures.v1_target.write_bytes(pretty_json_bytes(raw))
    before_target = fixtures.v1_target.read_bytes()
    with pytest.raises(InteropError) as raised:
        Catalog(fixtures.historical_v1, config, ledger).open()
    assert raised.value.code == "E_INTEGRITY"
    assert fixtures.v1_target.read_bytes() == before_target
    assert (fixtures.historical_v1 / "step-00000010.v3.json").read_bytes() == before_other
    assert not (fixtures.historical_v1 / ".ckptmeta").exists()


def test_upgrade_identity_collision_is_rejected_before_rewrite(fixtures, loaded):
    config, ledger = loaded
    shutil.copy2(fixtures.direct_target, fixtures.historical_v1 / "step-00000020.v3.json")
    original_v1 = fixtures.v1_target.read_bytes()
    with pytest.raises(InteropError) as raised:
        Catalog(fixtures.historical_v1, config, ledger).open()
    assert raised.value.code == "E_UPGRADE"
    assert raised.value.details["field"] == "manifest_id"
    assert fixtures.v1_target.read_bytes() == original_v1
    assert not (fixtures.historical_v1 / ".ckptmeta").exists()
