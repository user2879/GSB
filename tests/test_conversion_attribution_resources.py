from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from gsb_ledger.attribution import attribute_receipts
from gsb_ledger.config import load_config, load_requests
from gsb_ledger.conversions import convert_receipt, roundtrip_via_sidecar
from gsb_ledger.receipts import load_receipts
from gsb_ledger.resources import compute_resource


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "sample"


def _sample_receipts():
    config = load_config(SAMPLE / "config.json")
    requests = load_requests(SAMPLE / "requests.ndjson", batch_size=config.batch_size)
    receipts = load_receipts(config, project_root=SAMPLE)
    return config, requests, receipts


def test_r3_to_r1_roundtrip_restores_every_bypassed_field():
    _, _, receipt_load = _sample_receipts()
    original = next(item for item in receipt_load.receipts if item.key == "b-alpha-001/r-001")
    down, up, sidecar = roundtrip_via_sidecar(original, "r1")
    assert down.receipt.version == "r1"
    assert set(down.preserved_fields) == {
        "param_version",
        "duration_ms",
        "worker_id",
        "rank",
        "covered_refs",
        "param_digest",
    }
    assert down.lost_fields == ()
    assert up.lost_fields == ()
    for field in (
        "batch_id",
        "request_id",
        "output_digest",
        "param_version",
        "duration_ms",
        "worker_id",
        "rank",
        "covered_refs",
        "param_digest",
    ):
        assert getattr(up.receipt, field) == getattr(original, field)
    assert sidecar.get(original.key) is not None


def test_r1_upgrade_without_sidecar_marks_every_missing_field():
    _, _, receipt_load = _sample_receipts()
    r1 = next(item for item in receipt_load.receipts if item.version == "r1")
    upgraded = convert_receipt(r1, "r3")
    assert upgraded.direction == "upgrade"
    assert set(upgraded.lost_fields) == {
        "param_version",
        "duration_ms",
        "worker_id",
        "rank",
        "covered_refs",
        "param_digest",
    }
    assert set(upgraded.receipt.missing_fields) == set(upgraded.lost_fields)


def test_coverage_chain_and_version_tiebreak():
    config, requests, receipt_load = _sample_receipts()
    result = attribute_receipts(receipt_load.receipts, requests, config)
    by_request = {decision.request_id: decision for decision in result.decisions}
    assert by_request["r-003"].authority.batch_id == "b-alpha-002"
    assert by_request["r-003"].rule == "coverage_ref"
    statuses = {delivery.receipt.key: delivery.status for delivery in by_request["r-003"].deliveries}
    assert statuses["b-alpha-002/r-003"] == "authority"
    assert statuses["b-alpha-001/r-003"] == "covered"
    assert by_request["r-002"].rule == "param_version_desc_batch_id_asc"
    assert by_request["r-002"].authority.batch_id == "b-alpha-001"
    assert len(by_request["r-002"].deliveries) == 2
    legacy = next(batch for batch in result.batches if batch.batch_id == "b-alpha-legacy")
    assert legacy.invalidated
    assert legacy.invalidation_reason.startswith("param_version_stale")


def test_fixed_point_width_changes_integer_results_and_retained_digits():
    config, requests, receipt_load = _sample_receipts()
    result = attribute_receipts(receipt_load.receipts, requests, config)
    batch = next(item for item in result.batches if item.batch_id == "b-alpha-001")
    metrics = [compute_resource(batch, replace(config, fixed_point_bits=bits)) for bits in (16, 24, 32, 48)]
    assert len({item.fixed_macs for item in metrics}) == 4
    assert len({item.utilization_units for item in metrics}) == 4
    assert len({item.utilization_scale for item in metrics}) == 4
    assert len({item.retained_decimal_digits for item in metrics}) == 4
    expected_per_request = 8 * 32 * 64 + 4 * 32 * 32 + 2 * 64
    assert metrics[2].ops_per_request == expected_per_request
    assert metrics[2].raw_ops == 3 * expected_per_request

