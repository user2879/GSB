from __future__ import annotations

from opchain.engine import OperatorChainEngine, compare_execution_artifacts
from opchain.util import atomic_write_json, read_json


def _linear_chain(length: int = 4) -> dict:
    nodes = []
    edges = []
    previous = "x"
    for index in range(length):
        node_id = f"n{index}"
        op = ["relu", "sigmoid", "tanh", "gelu"][index % 4]
        params = {"approximation": "erf"} if op == "gelu" else {}
        nodes.append({"id": node_id, "op": op, "params": params})
        edges.append({"from": previous, "to": node_id})
        previous = node_id
    return {"input": "x", "outputs": [previous], "nodes": nodes, "edges": edges}


def test_budget_pending_persists_and_reused_artifacts_do_not_change(base_input, tmp_path):
    chain = _linear_chain(4)
    config = {
        "precision_tier": "fp64",
        "quantization_tier": "q8",
        "thread_pool_size": 2,
        "recompute_budget": 1,
        "snapshot_interval": 1,
        "tolerance": 1e-9,
        "fusion_strategy": "none",
    }
    workspace = tmp_path / "budget"
    first = OperatorChainEngine(chain, base_input, config, workspace).run()
    assert first["status"] == "partial"
    assert first["recomputed"] == ["n0"]
    assert first["pending"] == ["n1", "n2", "n3"]
    assert any(item["code"] == "E_RECOMPUTE_BUDGET_PENDING" for item in first["diagnostics"])
    first_hash = first["nodes"][0]["values_hash"]

    second = OperatorChainEngine(chain, base_input, config, workspace).run()
    assert second["reused"] == ["n0"]
    assert second["recomputed"] == ["n1"]
    assert second["pending"] == ["n2", "n3"]
    assert second["nodes"][0]["values_hash"] == first_hash

    larger_budget = dict(config)
    larger_budget["recompute_budget"] = 10
    completed = OperatorChainEngine(chain, base_input, larger_budget, workspace).run()
    assert completed["status"] == "complete"
    assert completed["reused"] == ["n0", "n1"]
    assert completed["recomputed"] == ["n2", "n3"]
    assert completed["nodes"][0]["values_hash"] == first_hash
    assert any(item["code"] == "E_SNAPSHOT_STALE" for item in completed["diagnostics"])


def test_snapshot_resume_after_stop_and_stale_detection(sample_chain, sample_input, sample_config, tmp_path):
    workspace = tmp_path / "snapshot"
    interrupted = OperatorChainEngine(sample_chain, sample_input, sample_config, workspace).run(stop_after=2)
    assert interrupted["status"] == "partial"
    assert interrupted["snapshot"]["status"] == "partial"
    assert interrupted["pending"]

    resumed = OperatorChainEngine(sample_chain, sample_input, sample_config, workspace).run(
        resume=True,
        strict_resume=True,
    )
    assert resumed["status"] == "complete"
    assert any(item["code"] == "SNAPSHOT_RESUMED" for item in resumed["diagnostics"])

    manifest_path = workspace / "snapshots" / "current" / "manifest.json"
    manifest = read_json(manifest_path)
    manifest["config_hash"] = "forced-mismatch"
    atomic_write_json(manifest_path, manifest)
    stale = OperatorChainEngine(sample_chain, sample_input, sample_config, workspace).run(
        resume=True,
        strict_resume=True,
    )
    assert stale["status"] == "failed"
    assert stale["failures"][0]["code"] == "E_SNAPSHOT_STALE"


def test_thread_pool_and_serial_produce_identical_ordering_and_failures(sample_chain, sample_input, sample_config, tmp_path, monkeypatch):
    serial_config = dict(sample_config)
    serial_config["thread_pool_size"] = 1
    parallel_config = dict(sample_config)
    parallel_config["thread_pool_size"] = 4

    original_execute = OperatorChainEngine._execute_node

    def flaky_execute(self, node, *args, **kwargs):
        if node.id == "sigmoid1":
            raise RuntimeError("injected node failure")
        return original_execute(self, node, *args, **kwargs)

    monkeypatch.setattr(OperatorChainEngine, "_execute_node", flaky_execute)
    serial = OperatorChainEngine(sample_chain, sample_input, serial_config, tmp_path / "serial").run()
    parallel = OperatorChainEngine(sample_chain, sample_input, parallel_config, tmp_path / "parallel").run()

    assert serial["status"] == parallel["status"] == "partial"
    assert compare_execution_artifacts(serial, parallel)["equal"]
    assert [failure["node_id"] for failure in serial["failures"]] == [failure["node_id"] for failure in parallel["failures"]]
    assert serial["failures"][0]["node_id"] == "sigmoid1"
    assert serial["failures"][0]["code"] == "E_ACTIVATION_FAILED"
    assert serial["failures"][0]["stage"] == "activation"
    assert any(node["id"] == "relu1" and node["status"] == "computed" for node in serial["nodes"])
    assert any(node["id"] == "tanh1" and node["status"] == "failed" for node in serial["nodes"])


def test_serial_and_parallel_success_paths_match(sample_chain, sample_input, sample_config, tmp_path):
    serial_config = dict(sample_config)
    serial_config["thread_pool_size"] = 1
    parallel_config = dict(sample_config)
    parallel_config["thread_pool_size"] = 4
    serial = OperatorChainEngine(sample_chain, sample_input, serial_config, tmp_path / "serial-ok").run()
    parallel = OperatorChainEngine(sample_chain, sample_input, parallel_config, tmp_path / "parallel-ok").run()
    assert serial["status"] == parallel["status"] == "complete"
    assert compare_execution_artifacts(serial, parallel)["equal"]
    assert serial["canonical_signature"] == parallel["canonical_signature"]


def test_permute_runtime_exception_is_isolated(relu_perm_chain, base_input, base_config, tmp_path, monkeypatch):
    original_execute = OperatorChainEngine._execute_node

    def flaky_execute(self, node, *args, **kwargs):
        if node.id == "perm1":
            raise RuntimeError("injected permutation failure")
        return original_execute(self, node, *args, **kwargs)

    monkeypatch.setattr(OperatorChainEngine, "_execute_node", flaky_execute)
    result = OperatorChainEngine(relu_perm_chain, base_input, base_config, tmp_path / "perm-failure").run()
    assert result["status"] == "partial"
    assert result["nodes"][0]["id"] == "relu1"
    assert result["nodes"][0]["status"] == "computed"
    assert result["failures"][0]["node_id"] == "perm1"
    assert result["failures"][0]["stage"] == "permutation"
    assert result["failures"][0]["code"] == "E_PERMUTE_FAILED"
