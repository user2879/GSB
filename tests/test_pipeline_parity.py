import json
from pathlib import Path

from conftest import ROOT
from qcalib.hashing import stable_json

from tests._helpers import build_inputs, comparable, execute


def _canonical(payload: dict) -> dict:
    payload = comparable(payload)
    payload.pop("reuse_ledger", None)
    return payload


def _run_sequence(tmp_path: Path, prefix: str) -> dict:
    state = tmp_path / f"{prefix}_state"
    output = tmp_path / f"{prefix}_ledger.json"
    previous = None
    commands = json.loads((ROOT / "configs" / "commands.json").read_text())["commands"]
    # Incremental path: apply each command on top of the previous cached run.
    for index, _ in enumerate(commands):
        stream = tmp_path / f"{prefix}_cmd_{index}.json"
        stream.write_text(json.dumps({"commands": commands[:index + 1]}),
                          encoding="utf-8")
        result = execute(state, stream, previous_cache=previous)
        previous = result["cache"]
    output.write_text(stable_json(comparable(result)), encoding="utf-8")
    return result


def test_incremental_sequence_equals_fresh_full_build(tmp_path):
    incremental = _run_sequence(tmp_path, "inc")
    full_stream = tmp_path / "full.json"
    full_stream.write_text(
        (ROOT / "configs" / "commands.json").read_text(), encoding="utf-8")
    full = execute(tmp_path / "fresh_state", full_stream)
    assert stable_json(_canonical(full)) == stable_json(_canonical(incremental))
    # The incremental run must actually reuse unchanged artifacts.
    assert any(item["hit"] for item in incremental["reuse_ledger"])
    assert not any(item["hit"] for item in full["reuse_ledger"])


def test_serial_and_parallel_runs_are_identical(tmp_path):
    stream = ROOT / "configs" / "commands.json"
    parallel = execute(tmp_path / "parallel", stream)
    serial_inputs, failures = build_inputs(stream)
    object.__setattr__(serial_inputs.config, "workers", 1)
    from qcalib.runner import run_pipeline
    from qcalib.snapshot import SnapshotStore
    serial = run_pipeline(
        serial_inputs, SnapshotStore(tmp_path / "serial"))
    assert stable_json(serial["layers"]) == stable_json(parallel["layers"])
    assert stable_json(serial["reuse_ledger"]) == stable_json(
        parallel["reuse_ledger"])


def test_failure_list_is_deterministically_ordered(tmp_path):
    stream = ROOT / "configs" / "commands.json"
    first = execute(tmp_path / "a", stream)
    second = execute(tmp_path / "b", stream)
    assert first["failures"] == second["failures"]
    codes = [item["code"] for item in first["failures"]]
    assert codes == sorted(codes)


def test_ledger_contains_all_required_artifacts(tmp_path):
    result = execute(tmp_path / "art", None)
    for name, artifact in result["layers"].items():
        assert "group_calibration" in artifact
        assert "group_quantization" in artifact
        assert "distillation" in artifact
        assert "stage_attribution" in artifact
        assert "cache_keys" in artifact
        for group in artifact["group_quantization"]:
            assert len(group["integers"]) == len(group["dequant"])
            assert len(group["error_kind"]) == len(group["abs_error"])
