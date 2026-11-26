from __future__ import annotations

import json
from pathlib import Path

import pytest

from ckptsim.errors import CkptError
from ckptsim.spec import load_bundle, load_events, load_layers, load_shards


def _rewrite(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def test_duplicate_port_is_rejected_with_line(spec_files) -> None:
    data = json.loads(spec_files.shards.read_text(encoding="utf-8"))
    data["shards"][1]["port"] = data["shards"][0]["port"]
    _rewrite(spec_files.shards, data)
    with pytest.raises(CkptError) as caught:
        load_shards(spec_files.shards)
    error = caught.value
    assert error.code == "E_SPEC"
    assert error.line and error.line > 1
    assert error.item == "shard-b"
    assert "duplicate loopback port" in error.detail


def test_cycle_is_rejected(spec_files) -> None:
    data = json.loads(spec_files.layers.read_text(encoding="utf-8"))
    data["parameter_versions"][0]["layers"][0]["derived_from"] = "out"
    _rewrite(spec_files.layers, data)
    with pytest.raises(CkptError) as caught:
        load_layers(spec_files.layers)
    assert caught.value.code == "E_SPEC"
    assert "cycle" in caught.value.detail
    assert caught.value.line


def test_missing_derived_reference_is_rejected(spec_files) -> None:
    data = json.loads(spec_files.layers.read_text(encoding="utf-8"))
    data["parameter_versions"][0]["layers"][1]["derived_from"] = "missing"
    _rewrite(spec_files.layers, data)
    with pytest.raises(CkptError) as caught:
        load_layers(spec_files.layers)
    assert caught.value.code == "E_SPEC"
    assert "missing layer" in caught.value.detail
    assert caught.value.item == "mix"


def test_event_seq_must_be_nondecreasing(spec_files) -> None:
    data = json.loads(spec_files.events.read_text(encoding="utf-8"))
    data["events"][2]["seq"] = 1
    _rewrite(spec_files.events, data)
    with pytest.raises(CkptError) as caught:
        load_events(spec_files.events)
    assert caught.value.code == "E_SPEC"
    assert "nondecreasing" in caught.value.detail
    assert caught.value.line


def test_checkpoint_generation_must_increase(spec_files) -> None:
    data = json.loads(spec_files.checkpoints.read_text(encoding="utf-8"))
    data["checkpoints"][1]["generation"] = 1
    _rewrite(spec_files.checkpoints, data)
    from ckptsim.spec import load_checkpoints

    with pytest.raises(CkptError) as caught:
        load_checkpoints(spec_files.checkpoints)
    assert caught.value.code == "E_SPEC"
    assert "strictly increasing" in caught.value.detail


def test_cross_reference_parameter_version(spec_files) -> None:
    data = json.loads(spec_files.checkpoints.read_text(encoding="utf-8"))
    data["checkpoints"][0]["parameter_version"] = 99
    _rewrite(spec_files.checkpoints, data)
    with pytest.raises(CkptError) as caught:
        load_bundle(
            spec_files.shards,
            spec_files.layers,
            spec_files.checkpoints,
            spec_files.events,
        )
    assert caught.value.code == "E_SPEC"
    assert "missing parameter_version" in caught.value.detail

