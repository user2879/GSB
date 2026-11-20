import json
import os
from pathlib import Path

from cfgdist.manifest import load_manifest
from cfgdist.upgrade import read_upgrade_trail, upgrade_manifest


def _v1(tmp_path: Path) -> Path:
    path = tmp_path / "manifest_v1.json"
    path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "entries": [
                    {"key": "k1", "value": 1},
                    {"key": "k2", "value": "x"},
                    {"key": "k3", "value": True},
                    {"key": "k4", "value": 1.5},
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_upgrade_resumes_from_checkpoint_and_processes_each_entry_once(tmp_path):
    manifest_path = _v1(tmp_path)
    runtime = tmp_path / "run"

    first = upgrade_manifest(manifest_path, ["node-a", "node-b"], runtime, limit=2)
    assert first["status"] == "interrupted"
    assert first["processed"] == 2

    checkpoint = json.loads((runtime / "checkpoints" / "manifest_v1.checkpoint.json").read_text())
    assert checkpoint["completed_steps"] == ["v1:k1", "v1:k2"]

    # Original manifest must stay untouched while the upgrade is incomplete.
    interrupted = load_manifest(manifest_path)
    assert interrupted["format_version"] == 1

    second = upgrade_manifest(manifest_path, ["node-a", "node-b"], runtime, limit=2)
    assert second["status"] == "interrupted"
    assert second["processed"] == 2
    still_v1 = load_manifest(manifest_path)
    assert still_v1["format_version"] == 1

    third = upgrade_manifest(manifest_path, ["node-a", "node-b"], runtime)
    assert third["status"] == "completed"
    assert third["processed"] == 4
    final = third
    # Reopening an already-current manifest must not reprocess anything.
    again = upgrade_manifest(manifest_path, ["node-a", "node-b"], runtime)
    assert again["status"] == "noop"

    upgraded = load_manifest(manifest_path)
    assert upgraded["format_version"] == 3
    assert upgraded["derivations"] == []
    for entry in upgraded["entries"]:
        assert entry["node"] == "node-a"
        assert entry["config_version"] == 3
        assert entry["strategy"]["mode"] == "targeted"

    trail = read_upgrade_trail(runtime / "checkpoints" / "manifest_v1.upgrade-trail.jsonl")
    keys = [record["key"] for record in trail]
    assert keys == ["k1", "k2", "k3", "k4", "k1", "k2", "k3", "k4"]
    assert [record["from_version"] for record in trail] == [1] * 4 + [2] * 4
    assert [record["to_version"] for record in trail] == [2] * 4 + [3] * 4
    # Every (step, entry) pair appears exactly once.
    markers = [f"v{record['from_version']}:{record['key']}" for record in trail]
    assert len(markers) == len(set(markers))

    backup = runtime / "backups" / "manifest_v1.v.backup.json"
    assert backup.exists()
    assert not os.access(backup, os.W_OK)
    backup_manifest = json.loads(backup.read_text())
    assert backup_manifest["format_version"] == 1


def test_upgrade_v2_to_v3_preserves_targets(tmp_path):
    path = tmp_path / "m2.json"
    path.write_text(
        json.dumps(
            {
                "format_version": 2,
                "entries": [
                    {"key": "k", "value": 1, "node": "node-b", "config_version": 2}
                ],
            }
        ),
        encoding="utf-8",
    )
    report = upgrade_manifest(path, ["node-a", "node-b"], tmp_path / "run")
    assert report["status"] == "completed"
    upgraded = load_manifest(path)
    assert upgraded["format_version"] == 3
    assert upgraded["entries"][0]["node"] == "node-b"
    assert upgraded["entries"][0]["config_version"] == 3


def test_upgrade_current_version_is_noop(tmp_path):
    path = tmp_path / "m3.json"
    path.write_text(
        json.dumps(
            {
                "format_version": 3,
                "entries": [
                    {
                        "key": "k",
                        "value": 1,
                        "node": "node-a",
                        "config_version": 3,
                        "strategy": {"mode": "targeted", "retries": 1, "timeout_ms": 500},
                    }
                ],
                "derivations": [],
            }
        ),
        encoding="utf-8",
    )
    report = upgrade_manifest(path, ["node-a"], tmp_path / "run")
    assert report["status"] == "noop"
