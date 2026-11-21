from __future__ import annotations

import numpy as np

from opchain.engine import OperatorChainEngine, compare_execution_artifacts
from opchain.edits import apply_edit


def test_fusion_changes_values_and_records_accumulation_metadata(relu_perm_chain, base_input, base_config, tmp_path):
    unfused_config = dict(base_config)
    unfused_config["fusion_strategy"] = "none"
    fused_config = dict(base_config)
    fused_config["fusion_strategy"] = "auto"

    unfused = OperatorChainEngine(relu_perm_chain, base_input, unfused_config, tmp_path / "unfused").run()
    fused = OperatorChainEngine(relu_perm_chain, base_input, fused_config, tmp_path / "fused").run()
    assert unfused["status"] == fused["status"] == "complete"
    assert compare_execution_artifacts(unfused, fused)["equal"] is False
    assert fused["fusion"]["pairs"] == {"relu1": "perm1"}

    unfused_relu, unfused_perm = unfused["nodes"]
    fused_relu, fused_perm = fused["nodes"]
    assert unfused_relu["values_hash"] == fused_relu["values_hash"]
    assert unfused_perm["values_hash"] != fused_perm["values_hash"]

    plain_values = np.asarray(unfused_perm["values"], dtype=np.float64)
    fused_values = np.asarray(fused_perm["values"], dtype=np.float64)
    with np.errstate(invalid="ignore"):
        difference = np.nanmax(np.abs(plain_values - fused_values))
    assert float(difference) > 0.0
    # The deterministic special value straddles a q8 rounding boundary. The
    # non-fused path rounds fixed then decimal, while the fused path reverses it.
    assert plain_values[0, 0] == 0.0625
    assert fused_values[0, 0] == 0.0
    assert fused_perm["metadata"]["fusion"]["rounding_order"] == ["decimal", "fixed"]
    assert unfused_perm["metadata"]["rounding_order"] == ["fixed", "decimal"]
    assert fused_perm["metadata"]["round_count"][0][0] == 2
    assert unfused_perm["metadata"]["round_count"][0][0] == 4
    assert any("FUSED_ACT_PERM" in label for label in fused_perm["metadata"]["element_order"][0][0])
    assert any("PERM:perm1" in label for label in unfused_perm["metadata"]["element_order"][0][0])


def test_fused_permutation_edit_reuses_activation_raw_sidecar(base_input, base_config, tmp_path):
    input_spec = dict(base_input)
    input_spec["shape"] = [2, 3, 4]
    input_spec["special_values"] = []
    chain = {
        "input": "x",
        "outputs": ["perm1"],
        "nodes": [
            {"id": "relu1", "op": "relu", "params": {}},
            {"id": "perm1", "op": "permute", "params": {"perm": [1, 0, 2]}},
        ],
        "edges": [{"from": "x", "to": "relu1"}, {"from": "relu1", "to": "perm1"}],
    }
    workspace = tmp_path / "incremental"
    OperatorChainEngine(chain, input_spec, base_config, workspace).run()
    edited_chain, edited_config = apply_edit(
        chain,
        base_config,
        {"kind": "set_params", "node_id": "perm1", "params": {"perm": [2, 1, 0]}},
    )
    incremental = OperatorChainEngine(edited_chain, input_spec, edited_config, workspace).run()
    full = OperatorChainEngine(edited_chain, input_spec, edited_config, tmp_path / "full").run()
    assert incremental["invalidated"] == ["perm1"]
    assert incremental["reused"] == ["relu1"]
    assert incremental["recomputed"] == ["perm1"]
    assert compare_execution_artifacts(incremental, full)["equal"]
