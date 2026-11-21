from __future__ import annotations

import numpy as np

from opchain.engine import OperatorChainEngine
from opchain.numeric import permute_values


def test_permutation_shape_and_lexicographic_layout():
    values = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    perm = (2, 0, 1)
    result = permute_values(values, perm)
    assert result.shape == (4, 2, 3)
    for output_index in np.ndindex(result.shape):
        source_index = tuple(output_index[perm.index(axis)] for axis in range(3))
        assert result[output_index] == values[source_index]


def test_inplace_conflict_records_pair_and_isolates_other_branch(base_input, base_config, tmp_path):
    chain = {
        "input": "x",
        "outputs": ["sigmoid_b"],
        "nodes": [
            {"id": "relu_a", "op": "relu", "params": {}, "inplace": True},
            {"id": "sigmoid_b", "op": "sigmoid", "params": {}},
        ],
        "edges": [
            {"from": "x", "to": "relu_a"},
            {"from": "x", "to": "sigmoid_b"},
        ],
    }
    result = OperatorChainEngine(chain, base_input, base_config, tmp_path / "conflict").run()
    assert result["status"] == "partial"
    assert result["nodes"][1]["id"] == "sigmoid_b"
    assert result["nodes"][1]["status"] == "computed"
    failure = result["failures"][0]
    assert failure["node_id"] == "relu_a"
    assert failure["code"] == "E_INPLACE_CONFLICT"
    assert failure["stage"] == "execute.inplace"
    assert failure["details"]["conflict_pairs"] == [["relu_a", "sigmoid_b"]]


def test_successful_inplace_writeback_updates_tensor_registry(base_input, base_config, tmp_path):
    chain = {
        "input": "x",
        "outputs": ["tanh1"],
        "nodes": [
            {"id": "relu1", "op": "relu", "params": {}, "inplace": True},
            {"id": "tanh1", "op": "tanh", "params": {}},
        ],
        "edges": [
            {"from": "x", "to": "relu1"},
            {"from": "relu1", "to": "tanh1"},
        ],
    }
    result = OperatorChainEngine(chain, base_input, base_config, tmp_path / "ok").run()
    assert result["status"] == "complete"
    registry = result["tensor_registry"]
    assert registry["relu1"]["producer"] == "relu1"
    assert registry["x"]["producer"] == "relu1"
    assert registry["x"]["writeback_of"] == "x"
    assert registry["x"]["values_hash"] == registry["relu1"]["values_hash"]
