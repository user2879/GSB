from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from edge_ledger.canonical import canonical_dumps
from edge_ledger.errors import FormatError
from edge_ledger.formats import (
    DecodeContext,
    decode_receipt_line,
    decode_receipts,
    encode_v1,
    verify_normalized_receipt,
)
from edge_ledger.integrity import compute_shard_proof


def test_three_formats_normalize_with_explicit_provenance(spec):
    ctx = DecodeContext(spec=spec)
    v1 = decode_receipts((spec.root / "receipts_v1.txt").read_text().splitlines(), ctx, "v1")[0]
    v2 = decode_receipts((spec.root / "receipts_v2.jsonl").read_text().splitlines(), ctx, "v2")[0]
    v3 = decode_receipts((spec.root / "receipts_v3.jsonl").read_text().splitlines(), ctx, "v3")
    assert v1.source_format == "v1"
    assert {"shard_seq", "sample_bytes", "node_id", "content_bytes_digest"} <= set(v1.filled_fields)
    assert dict(v1.fill_basis)["shard_seq"] == "derived:shard_id_suffix"
    assert {"lease_version", "quota_version", "content_bytes_digest"} <= set(v2.filled_fields)
    assert v3[0].invalid_ref_id == "ref-invalid-shard-0002"
    for receipt in (v1, v2, *v3):
        ok, reason = verify_normalized_receipt(receipt, spec)
        assert ok, reason


def test_v1_width_error_locates_line_and_field(spec):
    good = (spec.root / "receipts_v1.txt").read_text().rstrip("\n")
    with pytest.raises(FormatError) as exc:
        decode_receipt_line(good[:-3], 7)
    assert exc.value.code == "E_FORMAT"
    assert exc.value.line == 7
    assert exc.value.field == "digest"


def test_shard_sequence_mismatch_is_eformat(spec):
    line = (spec.root / "receipts_v2.jsonl").read_text().strip()
    obj = json.loads(line)
    obj["shard_seq"] = 0
    with pytest.raises(FormatError) as exc:
        decode_receipt_line(json.dumps(obj), 3, DecodeContext(spec=spec), "v2")
    assert exc.value.field == "shard_seq"
    assert exc.value.line == 3


def test_dangling_invalid_ref_is_eformat(spec):
    line = (spec.root / "receipts_v3.jsonl").read_text().splitlines()[0]
    obj = json.loads(line)
    obj["invalid_shard_ref"]["ref_id"] = "missing"
    with pytest.raises(FormatError) as exc:
        decode_receipt_line(json.dumps(obj), 11, DecodeContext(spec=spec), "v3")
    assert "invalid_shard_ref" in (exc.value.field or "")
    assert exc.value.line == 11


def test_canonical_order_is_hash_stable():
    left = canonical_dumps({"b": 2, "a": [{"z": 1, "y": 2}]})
    right = canonical_dumps({"a": [{"y": 2, "z": 1}], "b": 2})
    assert left == right


def test_proofs_are_deterministic_under_permutation_and_threads(spec):
    content = spec.content
    expected = [compute_shard_proof(spec.node("node-a"), content, i) for i in range(content.shard_count)]
    assert [p.byte_start for p in expected] == [0, 512, 1024, 1536]
    assert all(p.byte_end - p.byte_start == 512 for p in expected)
    assert all(p.sample_count == 64 for p in expected)

    ordered = [3, 0, 2, 1]
    permuted = [compute_shard_proof(spec.node("node-a"), content, i) for i in ordered]
    for seq, proof in zip(ordered, permuted):
        assert proof.digest == expected[seq].digest
        assert proof.content_bytes_digest == expected[seq].content_bytes_digest

    with ThreadPoolExecutor(max_workers=8) as pool:
        threaded = list(pool.map(lambda i: compute_shard_proof(spec.node("node-a"), content, i).digest, list(range(4)) * 20))
    for index, digest in enumerate(threaded):
        assert digest == expected[index % 4].digest
