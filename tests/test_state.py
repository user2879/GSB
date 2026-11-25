from __future__ import annotations

import copy

import pytest

from log_sampler.state import CommandError, SessionStore, apply_command


def test_session_commands_are_idempotent_and_ordered(sample_bundle, tmp_path):
    store = SessionStore(tmp_path / "state")
    store.initialize(
        sample_bundle["config"], sample_bundle["batches"], sample_bundle["injections"]
    )
    first = copy.deepcopy(sample_bundle["commands"][0])
    store.apply_commands([first])
    version = store.load()
    assert len(version["applied_commands"]) == 1
    store.apply_commands([first])
    assert len(store.load()["applied_commands"]) == 1

    conflict = copy.deepcopy(first)
    conflict["id"] = first["id"]
    conflict["batch"] = copy.deepcopy(conflict["batch"])
    conflict["batch"]["records"][0]["metric"] = 0.99999
    with pytest.raises(CommandError):
        store.apply_commands([conflict])


def test_delete_log_and_invalid_command_errors(sample_bundle, tmp_path):
    config = copy.deepcopy(sample_bundle["config"])
    batches = copy.deepcopy(sample_bundle["batches"])
    apply_command(
        config,
        batches,
        {"type": "delete_log", "log_id": "log-0001"},
    )
    assert all(
        record["log_id"] != "log-0001"
        for batch in batches
        for record in batch["records"]
    )
    with pytest.raises(CommandError):
        apply_command(config, batches, {"type": "delete_log", "log_id": "missing"})
    with pytest.raises(CommandError):
        apply_command(
            config,
            batches,
            {"type": "set_bucket_capacity", "device_id": "edge-a", "severity": "info", "capacity": 0},
        )

