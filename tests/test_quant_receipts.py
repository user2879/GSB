from __future__ import annotations

import numpy as np

from quantledger.config import LayerSpec
from quantledger.quant import evaluate_chain, layer_cache_keys, quantize_fixed
from quantledger.receipts import (
    Receipt,
    arbitrate_receipts,
    downgrade_q3_to_q1,
    parse_receipt,
    receipt_roundtrip,
    upgrade_to_q3,
)


def test_fixed_quantization_and_chain_are_deterministic():
    samples = np.array([[0.11, -0.27], [0.42, 0.08]])
    layer = LayerSpec("l", 2, 2, 2, np.array([[0.4, -0.2], [0.1, 0.3]]))
    output_a, records_a, _, meta_a = evaluate_chain(
        [layer], samples, revision=2, frac_bits=8, quant_level=8, batch_id="b"
    )
    output_b, records_b, _, meta_b = evaluate_chain(
        [layer], samples, revision=2, frac_bits=8, quant_level=8, batch_id="b"
    )
    assert output_a.tolist() == output_b.tolist()
    assert records_a[0]["integer_accumulator"] == records_b[0]["integer_accumulator"]
    assert meta_a["ledger_cache_key"] == meta_b["ledger_cache_key"]
    quantized = quantize_fixed(samples, frac_bits=8, quant_level=8)
    assert quantized.values.min() >= -128
    assert quantized.values.max() <= 127


def test_cache_key_changes_with_weight_and_quant_level():
    layer = LayerSpec("l", 2, 2, 2, np.array([[0.4, -0.2], [0.1, 0.3]]))
    activation = np.array([[0.1, 0.2]])
    base = layer_cache_keys(
        layer, activation=activation, revision=2, frac_bits=8, quant_level=8, batch_ids=["b"]
    )
    changed_layer = layer.clone()
    changed_layer.weight = changed_layer.weight + 0.1
    changed_weight = layer_cache_keys(
        changed_layer, activation=activation, revision=2, frac_bits=8, quant_level=8, batch_ids=["b"]
    )
    changed_level = layer_cache_keys(
        layer, activation=activation, revision=2, frac_bits=8, quant_level=16, batch_ids=["b"]
    )
    assert base["dequant"] != changed_weight["dequant"]
    assert base["dequant"] != changed_level["dequant"]


def test_q1_missing_markers_and_q3_q1_roundtrip():
    q1 = parse_receipt({"q": 1, "batch_id": "b", "layer_name": "l", "output_summary": "out"})
    upgraded = upgrade_to_q3(q1)
    assert upgraded.completeness["label"] == "q1_upgraded_partial"
    assert set(upgraded.completeness["missing_fields"]) >= {"revision", "duration_ms", "worker_id"}

    q3 = Receipt(
        3,
        "b",
        "l",
        "out",
        revision=7,
        duration_ms=5,
        worker_id="w1",
        stable_rank=1,
        superseded_refs=["old"],
        quant_params_summary={"digest": "q"},
    )
    roundtrip = receipt_roundtrip(q3)
    assert roundtrip["conclusion_unchanged"] is True
    assert roundtrip["restored_q3"]["superseded_refs"] == ["old"]
    q1_raw, sidecar = downgrade_q3_to_q1(q3)
    restored = parse_receipt(q1_raw, sidecars={sidecar["sidecar_key"]: sidecar})
    assert restored.revision == 7
    assert restored.superseded_refs == ["old"]


def test_arbitration_uses_revision_then_duration_then_worker():
    low_revision = Receipt(2, "b", "l", "low", revision=1, duration_ms=99, worker_id="w1")
    slow = Receipt(2, "b", "l", "slow", revision=2, duration_ms=20, worker_id="w2")
    fast = Receipt(2, "b", "l", "fast", revision=2, duration_ms=10, worker_id="w1")
    decision = arbitrate_receipts([low_revision, slow, fast])["decisions"][0]
    assert decision["winner"]["output_summary"] == "slow"
    assert decision["winner"]["revision"] == 2
    assert decision["rationale"]["order"][:3] == ["revision desc", "duration_ms desc", "worker_id asc"]
