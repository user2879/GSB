from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from walcrash.errors import BusinessError
from walcrash.snapshots import create_snapshot, restore_snapshot
from walcrash.verification import verify_replay

from .conftest import ALT_SAMPLE, make_replayed


def test_snapshot_self_digest_is_enforced_without_mutation(tmp_path: Path) -> None:
    storage = make_replayed(tmp_path, name="source")
    created = create_snapshot(storage, "snapshots/session.json")
    before = storage.ledger_rows()
    path = storage.workspace / created["path"]
    document = json.loads(path.read_text(encoding="utf-8"))
    document["content_digest"] = "0" * 64
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(BusinessError) as error:
        restore_snapshot(storage, "snapshots/session.json")
    assert error.value.code == "E_SNAPSHOT"
    assert error.value.context["field"] == "content_digest"
    assert storage.ledger_rows() == before


def test_snapshot_restore_is_whole_replacement_and_lists_diff(tmp_path: Path) -> None:
    source = make_replayed(tmp_path, name="source")
    created = create_snapshot(source, "snapshots/session.json")
    alt = make_replayed(tmp_path, source=ALT_SAMPLE, name="alt")
    alt.snapshots_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source.workspace / created["path"], alt.snapshots_dir / "session.json")
    result = restore_snapshot(alt, "snapshots/session.json")
    assert result["restore_mode"] == "whole_state_replacement"
    assert result["page_diff"]["added"] == ["p3"]
    assert result["page_diff"]["deleted"] == ["p4"]
    assert [item["page_id"] for item in result["page_diff"]["changed"]] == ["p1", "p2"]
    assert result["log_changed"] is True
    assert result["ledger"]["ok"]
    assert {page["page_id"] for page in alt.status()["pages"]} == {"p1", "p2", "p3"}
    assert alt.status()["cursor"] == 12
    assert verify_replay(alt)["ok"]
    assert alt.verify_ledger()["ok"]
    assert len(alt.checkpoint_rows()) == len(source.checkpoint_rows())


def test_snapshot_format_version_is_integer_and_checked(tmp_path: Path) -> None:
    storage = make_replayed(tmp_path, name="source")
    created = create_snapshot(storage, "snapshots/session.json")
    path = storage.workspace / created["path"]
    document = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(document["snapshot_format_version"], int)
    assert isinstance(document["budget_params_version"], int)
    assert isinstance(document["page_table_version"], int)
    document["snapshot_format_version"] = 99
    # Re-sign so the version check, not the self-digest check, is exercised.
    from walcrash.content import digest_json

    unsigned = dict(document)
    unsigned.pop("content_digest")
    document["content_digest"] = digest_json(unsigned)
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(BusinessError) as error:
        restore_snapshot(storage, "snapshots/session.json")
    assert error.value.code == "E_SNAPSHOT"
    assert error.value.context["field"] == "snapshot_format_version"


def test_disk_wal_mismatch_causes_whole_replacement(tmp_path: Path) -> None:
    storage = make_replayed(tmp_path, name="disk-log")
    create_snapshot(storage, "snapshots/session.json")
    wal_path = storage.inputs_dir / "wal.ndjson"
    original = wal_path.read_text(encoding="utf-8")
    wal_path.write_text(original + "\n", encoding="utf-8", newline="\n")
    result = restore_snapshot(storage, "snapshots/session.json")
    assert result["log_changed"] is True
    assert result["restore_mode"] == "whole_state_replacement"
    assert wal_path.read_text(encoding="utf-8") == original
    assert verify_replay(storage)["ok"]
