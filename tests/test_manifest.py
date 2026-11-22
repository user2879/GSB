from __future__ import annotations

import json

import pytest

from snapdiag.bitmap import bitmap_from_indices
from snapdiag.errors import DiagnosticError, E_FORMAT
from snapdiag.manifest import (
    FORMAT_VERSIONS,
    DictManifestResolver,
    build_manifest_model,
    decode_manifest,
    encode_manifest,
    encode_manifest_text,
    make_snapshot_manifest,
)


def _model():
    refs0 = {0: "a" * 64, 2: "b" * 64}
    refs1 = {5: "c" * 64}
    first = make_snapshot_manifest(
        snapshot_id="s0",
        block_count=12,
        bitmap=bytes(bitmap_from_indices(12, refs0)),
        digest_table=refs0,
        block_refs=refs0,
        parent_snapshot_id=None,
        snapshot_version=3,
    )
    second = make_snapshot_manifest(
        snapshot_id="s1",
        block_count=12,
        bitmap=bytes(bitmap_from_indices(12, refs1)),
        digest_table=refs1,
        block_refs=refs1,
        parent_snapshot_id="s0",
        snapshot_version=4,
    )
    model = build_manifest_model(block_size=512, block_count=12, snapshots=[first, second])
    resolver = DictManifestResolver(
        {
            "s0": {
                "digest_table": refs0,
                "block_refs": refs0,
                "parent_snapshot_id": None,
                "snapshot_version": 3,
            },
            "s1": {
                "digest_table": refs1,
                "block_refs": refs1,
                "parent_snapshot_id": "s0",
                "snapshot_version": 4,
            },
        }
    )
    return model, resolver


def test_all_historical_formats_normalize_to_the_same_model() -> None:
    model, resolver = _model()
    fingerprints = set()
    for version in FORMAT_VERSIONS:
        decoded = decode_manifest(encode_manifest_text(model, version), resolver=resolver)
        fingerprints.add(decoded.fingerprint())
        assert decoded.source_format == version
        if version == "m1":
            assert "snapshots.s0.digest_table" in decoded.filled_fields
            assert decoded.fill_basis["snapshots.s0.digest_table"] == "sqlite_snapshot_blocks"
        if version == "m2":
            assert "snapshots.s0.content_digest" in decoded.filled_fields
        if version == "m3":
            assert decoded.filled_fields == ()
    assert len(fingerprints) == 1


def test_m2_bitmap_checksum_mismatch_reports_snapshot_and_field() -> None:
    model, resolver = _model()
    document = encode_manifest(model, "m2")
    document["snapshots"][0]["bitmap_checksum"] = "0" * 64
    with pytest.raises(DiagnosticError) as caught:
        decode_manifest(json.dumps(document), resolver=resolver)
    assert caught.value.code == E_FORMAT
    assert caught.value.snapshot_id == "s0"
    assert caught.value.field == "bitmap_checksum"


def test_bitmap_length_and_count_are_validated() -> None:
    model, resolver = _model()
    document = encode_manifest(model, "m3")
    document["snapshots"][0]["bitmap_length"] = 3
    with pytest.raises(DiagnosticError) as caught:
        decode_manifest(json.dumps(document), resolver=resolver)
    assert caught.value.code == E_FORMAT
    assert caught.value.field == "bitmap_length"
    assert caught.value.snapshot_id == "s0"


def test_digest_table_must_match_set_bits() -> None:
    model, resolver = _model()
    document = encode_manifest(model, "m3")
    document["snapshots"][0]["digest_table"].pop(0)
    with pytest.raises(DiagnosticError) as caught:
        decode_manifest(json.dumps(document), resolver=resolver)
    assert caught.value.code == E_FORMAT
    assert caught.value.field == "digest_table"
    assert caught.value.snapshot_id == "s0"


def test_unknown_manifest_version_is_e_format() -> None:
    model, _ = _model()
    document = encode_manifest(model, "m3")
    document["format_version"] = "m9"
    with pytest.raises(DiagnosticError) as caught:
        decode_manifest(document)
    assert caught.value.code == E_FORMAT
    assert caught.value.field == "format_version"


def test_truncated_manifest_retains_snapshot_hint() -> None:
    model, resolver = _model()
    data = encode_manifest_text(model, "m3")
    cut = data[: data.index('"snapshot_id"') + 30]
    with pytest.raises(DiagnosticError) as caught:
        decode_manifest(cut, resolver=resolver)
    assert caught.value.code == E_FORMAT
    assert caught.value.field == "json"
    assert caught.value.snapshot_id == "s0"


def test_m3_content_digest_tamper_is_rejected() -> None:
    model, resolver = _model()
    document = encode_manifest(model, "m3")
    document["snapshots"][1]["content_digest"] = "f" * 64
    with pytest.raises(DiagnosticError) as caught:
        decode_manifest(json.dumps(document), resolver=resolver)
    assert caught.value.code == E_FORMAT
    assert caught.value.snapshot_id == "s1"
    assert caught.value.field == "content_digest"


def test_missing_parent_is_rejected() -> None:
    from dataclasses import replace

    model, resolver = _model()
    broken = build_manifest_model(
        block_size=model.block_size,
        block_count=model.block_count,
        snapshots=[
            model.snapshots[0],
            replace(model.snapshots[1], parent_snapshot_id="missing"),
        ],
    )
    with pytest.raises(DiagnosticError) as caught:
        decode_manifest(encode_manifest(broken, "m3"), resolver=resolver)
    assert caught.value.code == E_FORMAT
    assert caught.value.field == "parent_snapshot_id"


def test_m1_without_disk_resolver_is_rejected_as_lossy() -> None:
    model, _ = _model()
    with pytest.raises(DiagnosticError) as caught:
        decode_manifest(encode_manifest_text(model, "m1"))
    assert caught.value.code == E_FORMAT
    assert caught.value.field == "digest_table"


def test_m1_may_omit_bitmap_when_disk_state_can_fill_it() -> None:
    model, resolver = _model()
    document = encode_manifest(model, "m1")
    for entry in document["snapshots"]:
        entry.pop("bitmap")
    decoded = decode_manifest(json.dumps(document), resolver=resolver)
    assert decoded.fingerprint() == model.fingerprint()
