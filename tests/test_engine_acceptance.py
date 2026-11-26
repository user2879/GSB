from __future__ import annotations

from quantledger.engine import LedgerEngine, compare_ledgers
from quantledger.ledger import LedgerStore
from quantledger.receipts import arbitrate_receipts, load_receipts
from quantledger.util import atomic_write_json, read_json


def test_staged_incremental_and_blank_rebuild_are_identical(local_fixture):
    root = local_fixture["root"]
    incremental_dir = root / "incremental"
    rebuild_dir = root / "rebuild"
    raw_commands = read_json(local_fixture["commands"])["commands"]
    prefix_path = root / "prefix.json"
    atomic_write_json(prefix_path, {"commands": raw_commands[:2]})

    first = LedgerEngine(
        local_fixture["config"], commands_path=prefix_path, runtime_dir=incremental_dir, request_timeout_s=10
    ).run()
    second = LedgerEngine(
        local_fixture["config"], commands_path=local_fixture["commands"], runtime_dir=incremental_dir, request_timeout_s=10
    ).run()
    rebuilt = LedgerEngine(
        local_fixture["config"], commands_path=local_fixture["commands"], runtime_dir=rebuild_dir, request_timeout_s=10
    ).run(rebuild=True)
    incremental_ledger = LedgerStore(incremental_dir / "ledger.json").load()
    rebuilt_ledger = LedgerStore(rebuild_dir / "ledger.json").load()
    comparison = compare_ledgers(incremental_ledger, rebuilt_ledger)
    assert comparison["equal"], comparison["differences"]
    assert first["command_count"] == 2
    assert second["command_count"] == 4
    assert rebuilt["snapshot_digest"] == second["snapshot_digest"]
    assert all(item["conclusion_unchanged"] for item in incremental_ledger["roundtrip_checks"])

    receipts_dir = incremental_dir / "receipts"
    receipts, issues = load_receipts(
        [receipts_dir / "q1.json", receipts_dir / "q2.json", receipts_dir / "q3.json"],
        sidecar_path=receipts_dir / "q3_to_q1_sidecar.json",
    )
    assert not issues
    counts = {version: sum(1 for receipt in receipts if receipt.format == version) for version in (1, 2, 3)}
    assert len(set(counts.values())) == 1
    arbitration = arbitrate_receipts(receipts)
    assert arbitration["authority"]
    assert all(
        arbitration["authority"][batch_id]["decision_digest"]
        == incremental_ledger["authority"][batch_id]["decision_digest"]
        for batch_id in arbitration["authority"]
    )


def test_single_layer_replacement_reuses_upstream_prefix(local_fixture):
    root = local_fixture["root"]
    command_path = root / "single_replace.json"
    atomic_write_json(command_path, {"commands": [{"seq": 1, "type": "replace_weight", "layer": "l2", "weight_seed": 999}]})
    result = LedgerEngine(
        local_fixture["config"], commands_path=command_path, runtime_dir=root / "reuse", request_timeout_s=10
    ).run()
    ledger = LedgerStore(root / "reuse" / "ledger.json").load()
    reuse = [event for event in ledger["events"] if event["kind"] == "reuse"]
    invalidated = [event for event in ledger["events"] if event["kind"] == "invalidate" and event["entity"] == "layer"]
    assert len(reuse) == 2
    assert {event["layer"] for event in reuse} == {"l1"}
    assert {event["layer"] for event in invalidated} == {"l2"}
    assert result["recovery_report"]["reuse_entries"] == 2
