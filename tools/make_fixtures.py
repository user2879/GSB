"""Regenerate the deterministic DNS wire samples used by tests and docs."""

from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dns_local.wire import (
    AA,
    QR,
    RA,
    RD,
    TYPE_BY_NAME,
    Question,
    ResourceRecord,
    encode_message,
    frame_tcp,
    make_query,
)

WIRE = ROOT / "data" / "wire"


def write(name: str, payload: bytes) -> dict[str, object]:
    path = WIRE / name
    path.write_bytes(payload)
    return {"file": name, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def main() -> int:
    WIRE.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "format": "deterministic-dns-wire-samples/1",
        "samples": [],
    }
    samples: list[dict[str, object]] = manifest["samples"]  # type: ignore[assignment]

    query = make_query("www.lab.test.", "A", transaction_id=0x1001)
    entry = write("query-a.dns", query)
    entry.update({"kind": "query", "expect": "OK", "description": "A/IN query for www.lab.test."})
    samples.append(entry)

    tcp_entry = write("query-a.tcp.dns", frame_tcp(query))
    tcp_entry.update(
        {
            "kind": "tcp-query",
            "expect": "OK",
            "description": "The same query with a two-byte TCP length prefix.",
        }
    )
    samples.append(tcp_entry)

    question = Question("www.lab.test.", TYPE_BY_NAME["A"])
    answers = (
        ResourceRecord("www.lab.test.", TYPE_BY_NAME["A"], 1, 120, "192.0.2.10"),
        ResourceRecord("www.lab.test.", TYPE_BY_NAME["AAAA"], 1, 240, "2001:db8::10"),
    )
    compressed = encode_message(0x1001, QR | AA | RA | RD, (question,), answers, compress=True)
    entry = write("response-compressed.dns", compressed)
    entry.update(
        {
            "kind": "response",
            "expect": "OK",
            "description": "Two answers whose owner names are compression pointers to the question.",
            "assertions": ["compression pointers point backwards", "all answer names decode"],
        }
    )
    samples.append(entry)

    header = struct.pack("!HHHHHH", 0x1002, RD, 1, 0, 0, 0)
    loop_name = b"\x01a\xc0\x0c"
    loop_packet = header + loop_name + struct.pack("!HH", 1, 1)
    entry = write("pointer-loop.dns", loop_packet)
    entry.update(
        {
            "kind": "query",
            "expect": "E_WIRE",
            "offset": 14,
            "description": "A backwards pointer at offset 14 targets offset 12, creating a two-step loop.",
        }
    )
    samples.append(entry)

    forward = header + b"\xc0\x0e" + struct.pack("!HH", 1, 1)
    entry = write("pointer-forward.dns", forward)
    entry.update(
        {
            "kind": "query",
            "expect": "E_WIRE",
            "offset": 12,
            "description": "The very first pointer targets its own next byte, i.e. a forward reference.",
        }
    )
    samples.append(entry)

    out_of_bounds = struct.pack("!HHHHHH", 0x1003, RD, 1, 0, 0, 0)
    out_of_bounds += b"\x01x\xc0\xff" + struct.pack("!HH", 1, 1)
    entry = write("pointer-out-of-bounds.dns", out_of_bounds)
    entry.update(
        {
            "kind": "query",
            "expect": "E_WIRE",
            "offset": 14,
            "description": "A pointer to offset 255 is rejected by the strictly-backwards rule.",
        }
    )
    samples.append(entry)

    reserved = header + b"\x40" + b"\x00" * 5
    entry = write("bad-label-length.dns", reserved)
    entry.update(
        {
            "kind": "query",
            "expect": "E_WIRE",
            "offset": 12,
            "description": "Reserved label-length bits 0b01 are set.",
        }
    )
    samples.append(entry)

    entry = write("truncated-header.dns", b"\x10\x01\x01\x00\x00\x01")
    entry.update(
        {
            "kind": "query",
            "expect": "E_WIRE",
            "offset": 0,
            "description": "Packet ends inside the fixed 12-byte DNS header.",
        }
    )
    samples.append(entry)

    (WIRE / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
