from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from softmax_layernorm.engine import Engine, compute_from_scratch
from softmax_layernorm.errors import BusinessError
from softmax_layernorm.jsonutil import sha256_json
from softmax_layernorm.models import CommandSpec


def _command(index: int, **raw):
    return CommandSpec(index=index, command_id=f"cmd-{index}", raw=raw)


def test_tiny_budget_returns_pending_without_waiting_for_completion(make_run, tmp_path: Path) -> None:
    loaded, _, _, _ = make_run(
        rows=320,
        cols=220,
        config={
            "interaction_budget_ms": 0.0,
            "thread_pool_workers": 4,
            "snapshot_interval_commands": 1000,
            "layernorm_scale": 1.0,
            "snapshot_root": str(tmp_path / "budget-output"),
            "resume": True,
        },
    )
    command = _command(0, type="set_stabilize", enabled=False)
    engine = Engine(
        loaded,
        [command],
        sha256_json([command.raw]),
        output_dir=tmp_path / "budget-output",
        fresh=True,
    )
    response = engine.handle_command(command)
    assert response["status"] == "pending"
    assert "E_INTERACT_BUDGET_PENDING" in response["business_codes"]
    assert response["response_latency_ms"] < 500.0
    assert response["pending_queue"] == ["cmd-0"]
    report = engine.finalize()
    assert report["commands"][0]["status"] == "ok"
    assert report["business_code_counts"].get("E_INTERACT_BUDGET_PENDING", 0) >= 1
    final_calls, final_tensor = engine.export_final()
    full = compute_from_scratch(final_calls, final_tensor, loaded.config)
    assert np.array_equal(engine.states[0].softmax_result, full.states[0].softmax_result)


def test_resume_continues_from_committed_command_cursor(make_run, tmp_path: Path) -> None:
    loaded, _, _, _ = make_run(rows=6, cols=5)
    commands = [
        _command(0, type="set_epsilon", epsilon=1e-3),
        _command(1, type="set_stabilize", enabled=False),
        _command(
            2,
            type="replace_rows",
            start=2,
            end=4,
            values=[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
        ),
        _command(3, type="set_layernorm_axis", axis=0),
    ]
    digest = sha256_json([item.raw for item in commands])
    output = tmp_path / "resume-output"
    first = Engine(loaded, commands, digest, output_dir=output, fresh=True)
    first.handle_command(commands[0])
    first.handle_command(commands[1])
    first.finalize()
    assert first.next_command_index == 2

    resumed = Engine(loaded, commands, digest, output_dir=output, fresh=False)
    assert resumed.next_command_index == 2
    for command in commands[2:]:
        assert resumed.handle_command(command)["status"] == "ok"
    report = resumed.finalize()
    final_calls, final_tensor = resumed.export_final()
    full = compute_from_scratch(
        final_calls,
        final_tensor,
        loaded.config,
        affected_rows=report["summary"]["affected_rows"],
    )
    assert np.array_equal(resumed.states[0].softmax_result, full.states[0].softmax_result)
    assert np.array_equal(resumed.states[0].layernorm_result, full.states[0].layernorm_result)
    assert report["failures"] == [failure.to_dict() for failure in full.failures]


def test_config_mismatch_refuses_old_snapshot(make_run, tmp_path: Path) -> None:
    loaded, _, _, _ = make_run(rows=3, cols=3)
    commands = [_command(0, type="set_epsilon", epsilon=1e-3)]
    output = tmp_path / "mismatch-output"
    engine = Engine(
        loaded,
        commands,
        sha256_json([item.raw for item in commands]),
        output_dir=output,
        fresh=True,
    )
    engine.finalize()

    changed_config = loaded.config.__class__(
        **{**loaded.config.to_dict(), "thread_pool_workers": 3}
    )
    changed_loaded = loaded.__class__(
        config=changed_config,
        tensor_spec=loaded.tensor_spec,
        tensor=loaded.tensor,
        calls=loaded.calls,
        config_hash="different-config-hash",
        source_input_fingerprint=loaded.source_input_fingerprint,
        current_input_fingerprint=loaded.current_input_fingerprint,
        call_description_hash=loaded.call_description_hash,
        tensor_description_hash=loaded.tensor_description_hash,
    )
    with pytest.raises(BusinessError) as error:
        Engine(
            changed_loaded,
            commands,
            sha256_json([item.raw for item in commands]),
            output_dir=output,
            fresh=False,
        )
    assert error.value.failures[0].code == "E_CONFIG_MISMATCH"


def test_snapshot_manifest_and_stage_versions_are_complete(make_run, tmp_path: Path) -> None:
    loaded, _, _, _ = make_run(rows=3, cols=3)
    commands = [_command(0, type="set_epsilon", epsilon=1e-3)]
    output = tmp_path / "manifest-output"
    engine = Engine(
        loaded,
        commands,
        sha256_json([item.raw for item in commands]),
        output_dir=output,
        fresh=True,
    )
    engine.handle_command(commands[0])
    engine.finalize()
    current = json.loads((output / "current.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / current["manifest"]).read_text(encoding="utf-8"))
    assert current["snapshot_version"] == manifest["snapshot_version"]
    assert manifest["config_hash"] == loaded.config_hash
    assert manifest["source_input_fingerprint"] == loaded.source_input_fingerprint
    assert set(manifest["stages"]) >= {"input", "results", "state", "reuse_recompute", "failures"}
    for stage in manifest["stages"].values():
        assert stage["version"] == manifest["global_version"]
        assert len(stage["sha256"]) == 64


def test_corrupt_stage_is_rejected(make_run, tmp_path: Path) -> None:
    loaded, _, _, _ = make_run(rows=3, cols=3)
    commands = [_command(0, type="set_epsilon", epsilon=1e-3)]
    digest = sha256_json([item.raw for item in commands])
    output = tmp_path / "corrupt-output"
    engine = Engine(loaded, commands, digest, output_dir=output, fresh=True)
    engine.finalize()
    current = json.loads((output / "current.json").read_text(encoding="utf-8"))
    manifest_path = output / current["manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    (manifest_path.parent / manifest["stages"]["results"]["file"]).write_bytes(b"corrupt")
    with pytest.raises(BusinessError) as error:
        Engine(loaded, commands, digest, output_dir=output, fresh=False)
    assert error.value.failures[0].code == "E_CONFIG_MISMATCH"
