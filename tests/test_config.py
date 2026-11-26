from __future__ import annotations

import copy

import numpy as np

from quantledger.config import load_base_config, validate_base_config, validate_commands

from conftest import base_config, free_port


def test_worker_and_layer_validation_codes_skip_bad_items(tmp_path):
    ports = [free_port(), free_port()]
    raw = base_config(ports)
    raw["workers"][1]["worker_id"] = "wa"
    raw["workers"][1]["port"] = ports[0]
    raw["workers"].append(
        {
            "worker_id": "duplicate-port",
            "port": ports[0],
            "concurrency_quota": 1,
            "lease_limit": 1,
            "frac_bits": 8,
            "quant_level": 8,
        }
    )
    raw["workers"].append(
        {
            "worker_id": "badquota",
            "port": free_port(),
            "concurrency_quota": 0,
            "lease_limit": 1,
            "frac_bits": 8,
            "quant_level": 8,
        }
    )
    raw["operator"]["layers"].append(
        {"name": "badgroup", "rows": 2, "cols": 2, "group_size": 3, "weight_seed": 999}
    )
    raw["operator"]["layers"].append(
        {"name": "nan", "rows": 2, "cols": 2, "group_size": 2, "weight": [[float("nan"), 0], [0, 1]]}
    )
    loaded = validate_base_config(raw, tmp_path / "config.json")
    codes = [item.code for item in loaded.issues]
    assert "E_DUP_WORKER_ID" in codes
    assert "E_DUP_PORT" in codes
    assert "E_BAD_QUOTA" in codes
    assert "E_GROUP_NOT_DIVISIBLE" in codes
    assert "E_NONFINITE_WEIGHT" in codes
    assert [worker.worker_id for worker in loaded.state.workers] == ["wa"]


def test_layer_dimension_chain_and_allowed_enums(local_fixture):
    loaded = load_base_config(local_fixture["config"])
    assert not loaded.issues
    raw = copy.deepcopy(loaded.raw)
    raw["operator"]["quant_level"] = 7
    raw["operator"]["frac_bits"] = 5
    raw["operator"]["layers"][1]["cols"] = 9
    loaded = validate_base_config(raw)
    codes = [item.code for item in loaded.issues]
    assert "E_BAD_QUANT_LEVEL" in codes
    assert "E_BAD_FRAC_BITS" in codes
    assert "E_BAD_DIMENSION" in codes


def test_invalid_commands_emit_codes_and_are_skipped(local_fixture):
    loaded = load_base_config(local_fixture["config"])
    stream = {
        "commands": [
            {"seq": 1, "type": "replace_weight", "layer": "missing", "weight_seed": 1},
            {"seq": 2, "type": "set_quant_level", "value": 99},
            {"seq": 3, "type": "replace_weight", "layer": "l1", "weight": np.zeros((99, 99)).tolist()},
            {"seq": 4, "type": "set_frac_bits", "value": 10},
        ]
    }
    accepted, issues = validate_commands(stream, loaded.state)
    assert [item["seq"] for item in accepted] == [4]
    codes = [item.code for item in issues]
    assert codes == ["E_UNKNOWN_LAYER", "E_BAD_QUANT_LEVEL", "E_BAD_DIMENSION"]
