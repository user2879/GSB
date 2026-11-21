from __future__ import annotations

from opchain.engine import OperatorChainEngine


def _failure_code(chain: dict, input_spec: dict, config: dict, workspace) -> str:
    result = OperatorChainEngine(chain, input_spec, config, workspace).run()
    assert result["status"] == "failed"
    return result["failures"][0]["code"]


def test_duplicate_node_id_is_chain_invalid(base_input, base_config, tmp_path):
    chain = {
        "input": "x",
        "nodes": [
            {"id": "same", "op": "relu", "params": {}},
            {"id": "same", "op": "tanh", "params": {}},
        ],
        "edges": [{"from": "x", "to": "same"}],
    }
    assert _failure_code(chain, base_input, base_config, tmp_path / "duplicate") == "E_CHAIN_INVALID"


def test_dangling_edge_is_chain_invalid(base_input, base_config, tmp_path):
    chain = {
        "input": "x",
        "nodes": [{"id": "relu1", "op": "relu", "params": {}}],
        "edges": [{"from": "missing_tensor", "to": "relu1"}],
    }
    assert _failure_code(chain, base_input, base_config, tmp_path / "dangling") == "E_CHAIN_INVALID"


def test_cycle_is_reported_separately(base_input, base_config, tmp_path):
    chain = {
        "input": "x",
        "nodes": [
            {"id": "a", "op": "relu", "params": {}},
            {"id": "b", "op": "tanh", "params": {}},
        ],
        "edges": [{"from": "a", "to": "b"}, {"from": "b", "to": "a"}],
    }
    assert _failure_code(chain, base_input, base_config, tmp_path / "cycle") == "E_CYCLE_DETECTED"


def test_permutation_rank_and_uniqueness_are_validated(base_input, base_config, tmp_path):
    rank_chain = {
        "input": "x",
        "nodes": [
            {"id": "relu1", "op": "relu", "params": {}},
            {"id": "perm1", "op": "permute", "params": {"perm": [0]}},
        ],
        "edges": [{"from": "x", "to": "relu1"}, {"from": "relu1", "to": "perm1"}],
    }
    assert _failure_code(rank_chain, base_input, base_config, tmp_path / "rank") == "E_PERM_INVALID"

    duplicate_chain = {
        "input": "x",
        "nodes": [
            {"id": "relu1", "op": "relu", "params": {}},
            {"id": "perm1", "op": "permute", "params": {"perm": [0, 0]}},
        ],
        "edges": [{"from": "x", "to": "relu1"}, {"from": "relu1", "to": "perm1"}],
    }
    assert _failure_code(duplicate_chain, base_input, base_config, tmp_path / "duplicate_perm") == "E_PERM_INVALID"


def test_activation_parameter_validation(base_input, base_config, tmp_path):
    chain = {
        "input": "x",
        "nodes": [{"id": "gelu1", "op": "gelu", "params": {"approximation": "unsupported"}}],
        "edges": [{"from": "x", "to": "gelu1"}],
    }
    assert _failure_code(chain, base_input, base_config, tmp_path / "params") == "E_CHAIN_INVALID"

