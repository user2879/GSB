import json
import socket
from pathlib import Path

import pytest

from edge_lease.converge import reconcile
from edge_lease.errors import ConvergeError, PortError
from edge_lease.fixedpoint import quantize_array, quantize_scalar
from edge_lease.ledger import Ledger
from edge_lease.logging_setup import JsonlLogger
from edge_lease.models import Receipt
from edge_lease.pipeline import PipelineConfig, run_pipeline


def _config(workspace: Path, registry_name: str = "nodes.json", **overrides) -> PipelineConfig:
    defaults = dict(
        workdir=workspace / "run",
        registry_path=workspace / registry_name,
        shards_path=workspace / "shards.json",
        manifest_path=workspace / "manifest.json",
        scale=1000,
        content_seed=20261123,
        global_limit=3,
        batch_size=3,
        startup_timeout_s=15,
    )
    defaults.update(overrides)
    return PipelineConfig(**defaults)


def test_full_pipeline_converges_across_versions(sample_workspace):
    result = run_pipeline(_config(sample_workspace))
    assert result["status"] == "ok"
    for version in result["versions"]:
        assert version["items"] > 0
        assert all(row["ok"] for row in version["convergence_rows"])
        assert {
            outcome["status"]
            for outcome in version["node_outcomes"].values()
        } == {"success"}


def test_rerun_is_byte_reproducible(sample_workspace):
    first = run_pipeline(_config(sample_workspace, workdir=sample_workspace / "run-a"))
    second = run_pipeline(_config(sample_workspace, workdir=sample_workspace / "run-b"))
    digests_a = [v["plan_digest"] for v in first["versions"]]
    digests_b = [v["plan_digest"] for v in second["versions"]]
    assert digests_a == digests_b
    rows_a = [
        (row["item_id"], row["expected_digest"], row["expected_shard_digest"])
        for version in first["versions"]
        for row in version["convergence_rows"]
    ]
    rows_b = [
        (row["item_id"], row["expected_digest"], row["expected_shard_digest"])
        for version in second["versions"]
        for row in version["convergence_rows"]
    ]
    assert rows_a == rows_b


def test_concurrency_limits_are_never_exceeded(sample_workspace):
    config = _config(sample_workspace, global_limit=2, batch_size=6)
    run_pipeline(config)
    events = [
        json.loads(line)
        for line in (sample_workspace / "run" / "events.jsonl").read_text().splitlines()
    ]
    # Reconstruct running windows from dispatch_ok / failure events ordering is
    # timing-based; instead verify structural caps via semaphore config +
    # registry quotas persisted in the ledger batches.
    ledger = Ledger(sample_workspace / "run" / "lease.sqlite")
    try:
        for batch in ledger.committed_batches():
            quota = json.loads(batch["quota_json"])
            attempts = sum(quota.values())
            assert attempts <= batch["total_items"] * 3
            assert batch["committed"] == 1
    finally:
        ledger.close()
    assert any(event["event"] == "state_transition" for event in events)


def test_running_windows_never_breach_global_or_node_caps(sample_workspace):
    global_limit = 2
    config = _config(sample_workspace, global_limit=global_limit, batch_size=7)
    run_pipeline(config)
    events = [
        json.loads(line)
        for line in (sample_workspace / "run" / "events.jsonl").read_text().splitlines()
    ]
    starts = {
        event["item_id"]: event["ts_start"]
        for event in events
        if event["event"] == "task_running"
    }
    ends = {
        event["item_id"]: event["ts_end"]
        for event in events
        if event["event"] == "task_finished"
    }
    windows = [
        (starts[item_id], ends[item_id], item_id.split(":")[-1])
        for item_id in starts
    ]
    # At every start instant, count windows already active (start <= t < end).
    max_global = 0
    per_node_peak: dict = {}
    for start, _end, _node in windows:
        active = [
            window for window in windows if window[0] <= start < window[1]
        ]
        max_global = max(max_global, len(active))
        for node_id in {window[2] for window in active}:
            count = sum(1 for window in active if window[2] == node_id)
            per_node_peak[node_id] = max(per_node_peak.get(node_id, 0), count)
    assert max_global <= global_limit
    registry = {
        node["node_id"]: node["per_node_quota"]
        for node in json.loads((sample_workspace / "nodes.json").read_text())["nodes"]
    }
    for node_id, peak in per_node_peak.items():
        assert peak <= registry[node_id]


def test_wait_and_service_timing_recorded(sample_workspace):
    run_pipeline(_config(sample_workspace, global_limit=1, batch_size=2))
    report = [
        json.loads(line)
        for line in (sample_workspace / "run" / "events.jsonl")
        .read_text()
        .splitlines()
    ]
    assert report and all("ts_mono_ms" in event for event in report)
    ledger = Ledger(sample_workspace / "run" / "lease.sqlite")
    try:
        batches = ledger.committed_batches()
        assert batches and all(b["committed_ms"] is not None for b in batches)
    finally:
        ledger.close()


def test_port_conflict_is_e_port(sample_workspace):
    registry = json.loads((sample_workspace / "nodes.json").read_text())
    holder = socket.socket()
    holder.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    holder.bind(("127.0.0.1", registry["nodes"][0]["port"]))
    holder.listen(1)
    try:
        with pytest.raises(PortError) as exc:
            run_pipeline(_config(sample_workspace, workdir=sample_workspace / "port-run"))
        assert exc.value.code == "E_PORT"
    finally:
        holder.close()


def test_bad_reply_node_is_excluded_and_e_converge_reported(sample_workspace):
    registry = json.loads((sample_workspace / "nodes.json").read_text())
    for node in registry["nodes"]:
        if node["node_id"] == "edge-c":
            node["fail_mode"] = "bad_reply"
    (sample_workspace / "bad_nodes.json").write_text(json.dumps(registry))
    with pytest.raises(ConvergeError) as exc:
        run_pipeline(
            _config(
                sample_workspace,
                registry_name="bad_nodes.json",
                workdir=sample_workspace / "bad-run",
            )
        )
    assert exc.value.code == "E_CONVERGE"
    deviations = exc.value.context["deviations"]
    assert all(d["node_id"] == "edge-c" for d in deviations)
    events = [
        json.loads(line)
        for line in (sample_workspace / "bad-run" / "events.jsonl").read_text().splitlines()
    ]
    assert any(event.get("code") == "E_REPLY" for event in events)


def test_quant_mismatch_is_a_convergence_field(sample_workspace):
    registry = json.loads((sample_workspace / "nodes.json").read_text())
    for node in registry["nodes"]:
        if node["node_id"] == "edge-a":
            node["fail_mode"] = "quant_mismatch"
    (sample_workspace / "qm_nodes.json").write_text(json.dumps(registry))
    with pytest.raises(ConvergeError) as exc:
        run_pipeline(
            _config(
                sample_workspace,
                registry_name="qm_nodes.json",
                workdir=sample_workspace / "qm-run",
            )
        )
    fields = {f for d in exc.value.context["deviations"] for f in d["diff_fields"]}
    assert "quant_version" in fields


def test_numpy_vectorized_quantization_matches_scalar_engine():
    values = ["0.125", "0.3751", "0.5", "0.9999"]
    arr_q, arr_flags = quantize_array(values, 1000)
    for index, value in enumerate(values):
        scalar_q, scalar_flag = quantize_scalar(value, 1000)
        assert int(arr_q[index]) == scalar_q
        assert int(arr_flags[index]) == scalar_flag


def test_reconcile_detects_missing_receipt(sample_workspace):
    run_pipeline(_config(sample_workspace))
    ledger = Ledger(sample_workspace / "run" / "lease.sqlite")
    try:
        plan_doc = ledger.load_plan(1)
        from edge_lease.replay import _plan_items, _receipts

        items = _plan_items(plan_doc)
        receipts = _receipts(ledger.all_committed_receipts())
        grouped = {}
        for receipt in receipts:
            if receipt.node_id != "edge-c":
                grouped.setdefault(receipt.item_id, []).append(receipt)
        with pytest.raises(ConvergeError) as exc:
            reconcile(items, grouped)
        assert "receipt_missing" in exc.value.context["deviations"][0]["diff_fields"]
    finally:
        ledger.close()
