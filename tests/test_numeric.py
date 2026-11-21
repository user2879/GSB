from __future__ import annotations

import numpy as np

from opchain.engine import OperatorChainEngine, compare_execution_artifacts
from opchain.numeric import activate
from opchain.config import parse_config


def test_all_activations_are_deterministic_and_nonfinite_safe(base_input, base_config, tmp_path):
    chain = {
        "input": "x",
        "outputs": ["tanh1"],
        "nodes": [
            {"id": "relu1", "op": "relu", "params": {}},
            {"id": "gelu1", "op": "gelu", "params": {"approximation": "erf"}},
            {"id": "sigmoid1", "op": "sigmoid", "params": {}},
            {"id": "tanh1", "op": "tanh", "params": {}},
        ],
        "edges": [
            {"from": "x", "to": "relu1"},
            {"from": "relu1", "to": "gelu1"},
            {"from": "gelu1", "to": "sigmoid1"},
            {"from": "sigmoid1", "to": "tanh1"},
        ],
    }
    first = OperatorChainEngine(chain, base_input, base_config, tmp_path / "first").run()
    second = OperatorChainEngine(chain, base_input, base_config, tmp_path / "second").run()
    assert first["status"] == second["status"] == "complete"
    assert compare_execution_artifacts(first, second)["equal"]

    for node in first["nodes"]:
        metadata = node["metadata"]
        assert metadata["approximation"]["exp"]
        assert metadata["approximation"]["sqrt"]
        assert "exp" in metadata["approximation"]["all_available"]
        assert "sqrt" in metadata["approximation"]["all_available"]


def test_precision_tiers_change_values_and_decimal_metadata(base_input, base_config, tmp_path):
    fp64_config = dict(base_config)
    fp64_config["precision_tier"] = "fp64"
    fp32_config = dict(base_config)
    fp32_config["precision_tier"] = "fp32"
    chain = {
        "input": "x",
        "outputs": ["sigmoid1"],
        "nodes": [{"id": "sigmoid1", "op": "sigmoid", "params": {}}],
        "edges": [{"from": "x", "to": "sigmoid1"}],
    }
    fp64 = OperatorChainEngine(chain, base_input, fp64_config, tmp_path / "fp64").run()
    fp32 = OperatorChainEngine(chain, base_input, fp32_config, tmp_path / "fp32").run()
    fp64_node = fp64["nodes"][0]
    fp32_node = fp32["nodes"][0]
    assert fp64_node["values_hash"] != fp32_node["values_hash"]
    assert fp64_node["metadata"]["precision"]["decimal_places"] == 9
    assert fp32_node["metadata"]["precision"]["decimal_places"] == 6
    assert fp64_node["metadata"]["precision"]["dtype"] == "<f8"
    assert fp32_node["metadata"]["precision"]["dtype"] == "<f4"


def test_relu_and_gelu_handle_infinities_without_nan_corruption():
    config = parse_config({"precision_tier": "fp64", "quantization_tier": "none"})
    values = np.array([-np.inf, np.inf, np.nan, -2.0, 2.0], dtype=np.float64)
    relu = activate(values, "relu", {"negative_slope": 0.0}, config)
    assert relu[0] == 0.0 and relu[1] == np.inf
    assert np.isnan(relu[2])
    for approximation in ("erf", "tanh"):
        gelu = activate(values, "gelu", {"approximation": approximation}, config)
        assert gelu[0] == 0.0
        assert np.isposinf(gelu[1])
        assert np.isnan(gelu[2])
