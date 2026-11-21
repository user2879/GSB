from __future__ import annotations

from opchain.edits import apply_edit
from opchain.engine import OperatorChainEngine, compare_execution_artifacts


def _run_full(chain, input_spec, config, workspace):
    result = OperatorChainEngine(chain, input_spec, config, workspace).run()
    assert result["status"] == "complete"
    return result


def test_parameter_edit_recomputes_only_node_and_downstream(sample_chain, sample_input, sample_config, tmp_path):
    workspace = tmp_path / "incremental"
    OperatorChainEngine(sample_chain, sample_input, sample_config, workspace).run()

    edited_chain, edited_config = apply_edit(
        sample_chain,
        sample_config,
        {"kind": "set_params", "node_id": "gelu1", "params": {"approximation": "tanh"}},
    )
    incremental = OperatorChainEngine(edited_chain, sample_input, edited_config, workspace).run()
    full = _run_full(edited_chain, sample_input, edited_config, tmp_path / "full-param")

    assert incremental["invalidated"] == ["gelu1", "sigmoid1", "tanh1", "perm2"]
    assert incremental["reused"] == ["relu1", "perm1"]
    assert incremental["recomputed"] == ["gelu1", "sigmoid1", "tanh1", "perm2"]
    assert compare_execution_artifacts(incremental, full)["equal"]
    assert incremental["canonical_signature"] == full["canonical_signature"]


def test_precision_switch_recomputes_all_and_matches_full(sample_chain, sample_input, sample_config, tmp_path):
    workspace = tmp_path / "incremental"
    OperatorChainEngine(sample_chain, sample_input, sample_config, workspace).run()
    edited_chain, edited_config = apply_edit(
        sample_chain,
        sample_config,
        {"kind": "set_precision", "precision_tier": "fp32"},
    )
    incremental = OperatorChainEngine(edited_chain, sample_input, edited_config, workspace).run()
    full = _run_full(edited_chain, sample_input, edited_config, tmp_path / "full-precision")
    assert incremental["invalidated"] == [node["id"] for node in incremental["nodes"]]
    assert incremental["reused"] == []
    assert compare_execution_artifacts(incremental, full)["equal"]


def test_node_order_edit_uses_stable_topology_and_matches_full(sample_chain, sample_input, sample_config, tmp_path):
    workspace = tmp_path / "incremental"
    OperatorChainEngine(sample_chain, sample_input, sample_config, workspace).run()
    moved_chain, moved_config = apply_edit(
        sample_chain,
        sample_config,
        {"kind": "move_node", "node_id": "sigmoid1", "target_index": 1},
    )
    incremental = OperatorChainEngine(moved_chain, sample_input, moved_config, workspace).run()
    full = _run_full(moved_chain, sample_input, moved_config, tmp_path / "full-order")
    assert incremental["invalidated"] == ["sigmoid1", "tanh1", "perm2"]
    assert incremental["reused"] == ["relu1", "perm1", "gelu1"]
    assert compare_execution_artifacts(incremental, full)["equal"]


def test_subgraph_replacement_matches_full(sample_chain, sample_input, sample_config, tmp_path):
    workspace = tmp_path / "incremental"
    OperatorChainEngine(sample_chain, sample_input, sample_config, workspace).run()
    edit = {
        "kind": "replace_subgraph",
        "remove": ["gelu1", "sigmoid1"],
        "add_nodes": [
            {"id": "gelu_new", "op": "gelu", "params": {"approximation": "tanh"}, "inputs": ["perm1"]},
            {"id": "sigmoid_new", "op": "sigmoid", "params": {}, "inputs": ["gelu_new"]},
        ],
        "rewire": {"gelu1": "gelu_new", "sigmoid1": "sigmoid_new"},
    }
    replaced_chain, replaced_config = apply_edit(sample_chain, sample_config, edit)
    incremental = OperatorChainEngine(replaced_chain, sample_input, replaced_config, workspace).run()
    full = _run_full(replaced_chain, sample_input, replaced_config, tmp_path / "full-replace")
    assert incremental["status"] == "complete"
    assert set(incremental["reused"]) == {"relu1", "perm1"}
    assert set(incremental["recomputed"]) == {"gelu_new", "sigmoid_new", "tanh1", "perm2"}
    assert compare_execution_artifacts(incremental, full)["equal"]


def test_edit_sequence_matches_fresh_full_rebuild_each_step(sample_chain, sample_input, sample_config, tmp_path):
    chain = sample_chain
    config = sample_config
    workspace = tmp_path / "incremental"

    edits = [
        {"kind": "set_params", "node_id": "gelu1", "params": {"approximation": "tanh"}},
        {"kind": "move_node", "node_id": "sigmoid1", "target_index": 1},
        {
            "kind": "replace_subgraph",
            "remove": ["gelu1", "sigmoid1"],
            "add_nodes": [
                {"id": "gelu_new", "op": "gelu", "params": {"approximation": "erf"}, "inputs": ["perm1"]},
                {"id": "sigmoid_new", "op": "sigmoid", "params": {}, "inputs": ["gelu_new"]},
            ],
            "rewire": {"gelu1": "gelu_new", "sigmoid1": "sigmoid_new"},
        },
        {"kind": "set_precision", "precision_tier": "fp32"},
        {"kind": "set_inplace", "node_id": "relu1", "inplace": True},
    ]

    for step, edit in enumerate(edits):
        chain, config = apply_edit(chain, config, edit)
        incremental = OperatorChainEngine(chain, sample_input, config, workspace).run()
        full = _run_full(chain, sample_input, config, tmp_path / f"full-step-{step}")
        comparison = compare_execution_artifacts(incremental, full)
        assert comparison["equal"], (step, edit, comparison, incremental["failures"], full["failures"])
        assert incremental["status"] == "complete"

