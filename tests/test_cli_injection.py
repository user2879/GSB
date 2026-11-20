import json
import subprocess
import sys
from pathlib import Path

from cfgdist.errors import EXIT_CODE_FOR_BIZ, ErrorCode
from cfgdist.logging_setup import read_log_records
from cfgdist.manifest import manifest_to_json


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "cfgdist.cli", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def _nodes_file(tmp_path: Path, node_specs) -> Path:
    path = tmp_path / "nodes.json"
    path.write_text(json.dumps({"nodes": node_specs}), encoding="utf-8")
    return path


def test_cli_corrupt_manifest_returns_e_format(tmp_path, v3_manifest, node_specs):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(manifest_to_json(v3_manifest), encoding="utf-8")
    workdir = tmp_path / "run"
    result = _run_cli(
        "inject-corrupt",
        str(manifest_path),
        "--kind",
        "unknown-version",
        "--workdir",
        str(workdir),
    )
    assert result.returncode == EXIT_CODE_FOR_BIZ[ErrorCode.E_FORMAT]
    payload = json.loads(result.stdout)
    assert payload["code"] == ErrorCode.E_FORMAT
    records = read_log_records(workdir / "logs" / "events.jsonl")
    assert any(item.get("error_code") == ErrorCode.E_FORMAT for item in records)


def test_cli_port_conflict_returns_e_port(tmp_path, node_specs):
    nodes_path = _nodes_file(tmp_path, node_specs)
    workdir = tmp_path / "run"
    result = _run_cli(
        "inject-port",
        "--nodes",
        str(nodes_path),
        "--workdir",
        str(workdir),
    )
    assert result.returncode == EXIT_CODE_FOR_BIZ[ErrorCode.E_PORT]
    payload = json.loads(result.stdout)
    assert payload["code"] == ErrorCode.E_PORT
    assert len(payload["details"]["busy"]) == len(node_specs)


def test_cli_snapshot_conflict_returns_e_version(
    tmp_path, v3_manifest, node_specs
):
    # Build a committed session + snapshot through the public CLI.
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(manifest_to_json(v3_manifest), encoding="utf-8")
    nodes_path = _nodes_file(tmp_path, node_specs)
    workdir = tmp_path / "run"

    distribute = _run_cli(
        "distribute",
        str(manifest_path),
        "--nodes",
        str(nodes_path),
        "--workdir",
        str(workdir),
    )
    assert distribute.returncode == 0

    save = _run_cli(
        "snapshot-save",
        str(manifest_path),
        "--nodes",
        str(nodes_path),
        "--workdir",
        str(workdir),
    )
    assert save.returncode == 0
    snapshot_path = json.loads(save.stdout)["snapshot"]

    conflict = _run_cli(
        "inject-snapshot",
        snapshot_path,
        "--manifest",
        str(manifest_path),
        "--workdir",
        str(workdir),
    )
    assert conflict.returncode == EXIT_CODE_FOR_BIZ[ErrorCode.E_VERSION]
    payload = json.loads(conflict.stdout)
    assert payload["code"] == ErrorCode.E_VERSION
    assert payload["details"]["replaced"] is True

    archive = Path(workdir) / "snapshots" / "archive"
    assert any(archive.glob("*.json"))


def test_cli_upgrade_limit_then_complete(tmp_path, node_specs):
    manifest_path = tmp_path / "v1.json"
    manifest_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "entries": [
                    {"key": "k1", "value": 1},
                    {"key": "k2", "value": 2},
                ],
            }
        ),
        encoding="utf-8",
    )
    nodes_path = _nodes_file(tmp_path, node_specs)
    workdir = tmp_path / "run"

    first = _run_cli(
        "upgrade",
        str(manifest_path),
        "--nodes",
        str(nodes_path),
        "--workdir",
        str(workdir),
        "--limit",
        "1",
    )
    assert first.returncode == 0
    assert json.loads(first.stdout)["status"] == "interrupted"

    second = _run_cli(
        "upgrade",
        str(manifest_path),
        "--nodes",
        str(nodes_path),
        "--workdir",
        str(workdir),
    )
    assert second.returncode == 0
    assert json.loads(second.stdout)["status"] == "completed"

    validate = _run_cli("validate", str(manifest_path))
    assert validate.returncode == 0
    assert json.loads(validate.stdout)["format_version"] == 3
