from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest
from dnslib import DNSRecord, QTYPE

from dns_local.errors import BusinessError
from dns_local.wire import (
    AA,
    QR,
    RA,
    RD,
    RCODE_NOERROR,
    Question,
    ResourceRecord,
    build_response,
    encode_message,
    frame_tcp,
    make_query,
    maybe_truncate_udp,
    parse_message,
    unframe_tcp,
)


def test_manifest_files_match_digests(root: Path) -> None:
    import hashlib

    manifest = json.loads((root / "data" / "wire" / "manifest.json").read_text(encoding="utf-8"))
    for sample in manifest["samples"]:
        payload = (root / "data" / "wire" / sample["file"]).read_bytes()
        assert len(payload) == sample["bytes"]
        assert hashlib.sha256(payload).hexdigest() == sample["sha256"]


def test_query_round_trip_and_dnslib_cross_check(root: Path) -> None:
    raw = (root / "data" / "wire" / "query-a.dns").read_bytes()
    message = parse_message(raw)
    assert message.header.transaction_id == 0x1001
    assert message.header.qdcount == 1
    assert message.questions[0].name == "www.lab.test."
    reference = DNSRecord.parse(raw)
    assert str(reference.q.qname) == "www.lab.test."
    assert reference.q.qtype == QTYPE.A

    generated = make_query("WWW.Lab.Test.", "A", transaction_id=0x4321)
    parsed = parse_message(generated)
    assert parsed.questions[0].name == "www.lab.test."
    assert DNSRecord.parse(generated).header.id == 0x4321


def test_compressed_response_expands_and_cross_checks(root: Path) -> None:
    raw = (root / "data" / "wire" / "response-compressed.dns").read_bytes()
    message = parse_message(raw)
    assert [record.name for record in message.answers] == ["www.lab.test.", "www.lab.test."]
    assert [record.type_name for record in message.answers] == ["A", "AAAA"]
    assert b"\xc0\x0c" in raw
    reference = DNSRecord.parse(raw)
    assert len(reference.rr) == 2
    assert str(reference.rr[0].rname) == "www.lab.test."


@pytest.mark.parametrize("filename", [
    "pointer-loop.dns",
    "pointer-forward.dns",
    "pointer-out-of-bounds.dns",
    "bad-label-length.dns",
    "truncated-header.dns",
])
def test_malformed_samples_report_wire_offset(root: Path, filename: str) -> None:
    manifest = json.loads((root / "data" / "wire" / "manifest.json").read_text(encoding="utf-8"))
    sample = next(item for item in manifest["samples"] if item["file"] == filename)
    with pytest.raises(BusinessError) as caught:
        parse_message((root / "data" / "wire" / filename).read_bytes())
    assert caught.value.code == "E_WIRE"
    assert caught.value.offset == sample["offset"]


def test_tcp_framing_round_trip(root: Path) -> None:
    framed = (root / "data" / "wire" / "query-a.tcp.dns").read_bytes()
    payload = unframe_tcp(framed)
    assert len(payload) == struct.unpack("!H", framed[:2])[0]
    assert unframe_tcp(frame_tcp(payload)) == payload


def test_encoder_never_uses_a_forward_pointer() -> None:
    question = Question("www.lab.test.", 1)
    records = tuple(
        ResourceRecord("www.lab.test.", 1, 1, 60, f"192.0.2.{index}")
        for index in range(1, 20)
    )
    packet = encode_message(0x1234, QR | AA | RD | RA, (question,), records)
    position = 0
    while True:
        position = packet.find(b"\xc0", position)
        if position < 0:
            break
        assert position + 1 < len(packet)
        target = ((packet[position] & 0x3F) << 8) | packet[position + 1]
        assert target < position
        position += 1


def test_udp_truncation_sets_tc_and_drops_records() -> None:
    query_raw = make_query("bulk.lab.test.", "A", transaction_id=0x7777)
    query = parse_message(query_raw)
    answers = tuple(
        ResourceRecord("bulk.lab.test.", 1, 1, 60, f"192.0.2.{index}")
        for index in range(1, 50)
    )
    full = build_response(query, answers=answers, rcode=RCODE_NOERROR, authoritative=True)
    assert len(full) > 512
    truncated, did_truncate = maybe_truncate_udp(full, 512)
    parsed = parse_message(truncated)
    assert did_truncate is True
    assert parsed.header.flags & 0x0200
    assert parsed.header.ancount == 0
    assert len(truncated) <= 512


def test_trailing_bytes_are_rejected() -> None:
    raw = make_query("www.lab.test.", "A") + b"X"
    with pytest.raises(BusinessError) as caught:
        parse_message(raw)
    assert caught.value.code == "E_WIRE"
    assert caught.value.offset == len(raw) - 1

