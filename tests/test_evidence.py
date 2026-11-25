from __future__ import annotations

import json

from log_sampler.evidence import (
    EvidenceSlot,
    convert_evidence,
    parse_evidence,
    serialize_evidence,
)


def _slots() -> list[EvidenceSlot]:
    return [
        EvidenceSlot(
            bucket_id="edge-a|critical",
            slot_no=1,
            weight=125,
            admitted=1,
            arrival_ordinal=1,
            random_digest="11" * 32,
            replaced_record_ref=None,
            bucket_count_snapshot=1,
            log_id="log-0001",
            arrival_seq=1,
            record_fingerprint="22" * 32,
            raw_metric=0.125,
        ),
        EvidenceSlot(
            bucket_id="edge-b|info",
            slot_no=2,
            weight=910,
            admitted=1,
            arrival_ordinal=4,
            random_digest="33" * 32,
            replaced_record_ref="log-0002",
            bucket_count_snapshot=4,
            log_id="log-0008",
            arrival_seq=2,
            record_fingerprint="44" * 32,
            raw_metric=0.91,
        ),
        EvidenceSlot(
            bucket_id="edge-b|info",
            slot_no=0,
            weight=200,
            admitted=0,
            arrival_ordinal=5,
            random_digest="55" * 32,
            replaced_record_ref=None,
            bucket_count_snapshot=5,
            log_id="log-0009",
            arrival_seq=2,
            record_fingerprint="66" * 32,
            raw_metric=0.2,
        ),
    ]


def test_p1_fixed_width_parses_required_columns():
    slots = _slots()
    payload, sidecar = serialize_evidence(slots, "p1")
    lines = payload.splitlines()
    assert all(len(line) == 78 for line in lines)
    parsed = parse_evidence(payload, "p1", sidecar)
    assert not parsed.issues
    assert [(item.bucket_id, item.slot_no, item.weight, item.admitted) for item in parsed.slots] == [
        ("edge-a|critical", 1, 125, 1),
        ("edge-b|info", 2, 910, 1),
        ("edge-b|info", 0, 200, 0),
    ]
    assert parsed.slots[1].arrival_ordinal == 4
    assert parsed.slots[1].replaced_record_ref == "log-0002"
    assert sidecar["rows"][1]["bucket_count_snapshot"] == 4


def test_p2_and_p3_preserve_full_internal_representation():
    slots = _slots()
    for version in ("p2", "p3"):
        payload, sidecar = serialize_evidence(slots, version)
        parsed = parse_evidence(payload, version, sidecar)
        assert not parsed.issues
        assert [item.to_dict() for item in parsed.slots] == [
            item.to_dict() for item in slots
        ]


def test_downgrade_upgrade_uses_sidecar_not_recomputation():
    slots = _slots()
    p3, _ = serialize_evidence(slots, "p3")
    p1 = convert_evidence(p3, "p3", "p1")
    assert not p1.issues
    p3_back = convert_evidence(p1.payload, "p1", "p3", p1.sidecar)
    assert not p3_back.issues
    parsed = parse_evidence(p3_back.payload, "p3")
    assert not parsed.issues
    assert [item.to_dict() for item in parsed.slots] == [item.to_dict() for item in slots]

    p2 = convert_evidence(p3, "p3", "p2")
    p1_chain = convert_evidence(p2.payload, "p2", "p1", p2.sidecar)
    p3_chain = convert_evidence(p1_chain.payload, "p1", "p3", p1_chain.sidecar)
    assert not p3_chain.issues
    assert [item.to_dict() for item in parse_evidence(p3_chain.payload, "p3").slots] == [
        item.to_dict() for item in slots
    ]


def test_missing_sidecar_metadata_is_reported_on_upgrade():
    p1, _ = serialize_evidence(_slots(), "p1")
    upgraded = convert_evidence(p1, "p1", "p3")
    assert upgraded.issues
    assert all(item.code == "E_SAMPLE_METADATA" for item in upgraded.issues)
    assert {item.field for item in upgraded.issues} >= {
        "arrival_ordinal",
        "random_digest",
    }


def test_p1_crc_error_skips_only_bad_row():
    slots = _slots()
    payload, sidecar = serialize_evidence(slots, "p1")
    lines = payload.splitlines()
    corrupted_first = bytearray(lines[0])
    corrupted_first[68] = ord("0") if lines[0][68] == ord("1") else ord("1")
    corrupted = (
        bytes(corrupted_first)
        + b"\n"
        + lines[1]
        + b"\n"
        + lines[2]
        + b"\n"
    )
    parsed = parse_evidence(corrupted, "p1", sidecar)
    assert len(parsed.slots) == 2
    assert len(parsed.issues) == 1
    assert parsed.issues[0].row == 0


def test_p3_integrity_failure_is_reported():
    payload, _ = serialize_evidence(_slots(), "p3")
    corrupted = payload[:-1] + bytes([payload[-1] ^ 0x01])
    parsed = parse_evidence(corrupted, "p3")
    assert not parsed.slots
    assert parsed.issues
    assert parsed.issues[0].field == "sha256"


def test_p2_missing_field_skips_bad_row_only():
    payload, sidecar = serialize_evidence(_slots(), "p2")
    document = json.loads(payload)
    del document["rows"][0]["random_digest"]
    malformed = json.dumps(document).encode("utf-8")
    parsed = parse_evidence(malformed, "p2", sidecar)
    assert len(parsed.slots) == 2
    assert any(item.row == 0 and "random_digest" in item.message for item in parsed.issues)
