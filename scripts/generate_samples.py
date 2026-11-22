"""生成全部本地确定性样本（可重复运行，内容固定）。

输出到 dns_workbench/samples/：
- zone.json / zone_corrupt.json
- cache_v1.json / cache_v2.json / cache_v3.json / cache_bad_version.json
- upstream_answers.json / upstream_u1.json / upstream_u2.json
- wire_query.bin / wire_response.bin（DNS 线格式，含压缩指针）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dns_workbench.wire import (
    CLASS_IN,
    TYPE_A,
    TYPE_CNAME,
    TYPE_NS,
    DnsMessage,
    Question,
    ResourceRecord,
    build_response_flags,
    encode_message,
)

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "dns_workbench" / "samples"


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False) + "\n", encoding="utf-8")


def main() -> int:
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    zone = {
        "zone_version": 1,
        "origin": "example.test",
        "soa": {
            "mname": "ns1.example.test",
            "rname": "hostmaster.example.test",
            "serial": 2024010101,
            "refresh": 7200,
            "retry": 3600,
            "expire": 1209600,
            "minimum": 60,
        },
        "records": [
            {"name": "ns1.example.test", "qtype": TYPE_A, "rclass": CLASS_IN, "ttl": 3600,
             "value": "10.0.0.2", "rdata_hex": "0a000002"},
            {"name": "www.example.test", "qtype": TYPE_CNAME, "rclass": CLASS_IN, "ttl": 300,
             "value": "edge.example.test", "rdata_hex": ""},
            {"name": "edge.example.test", "qtype": TYPE_A, "rclass": CLASS_IN, "ttl": 300,
             "value": "10.0.0.5", "rdata_hex": "0a000005"},
            {"name": "edge.example.test", "qtype": TYPE_A, "rclass": CLASS_IN, "ttl": 300,
             "value": "10.0.0.6", "rdata_hex": "0a000006"},
            {"name": "mx.example.test", "qtype": TYPE_NS, "rclass": CLASS_IN, "ttl": 600,
             "value": "ns1.example.test", "rdata_hex": ""},
            {"name": "forward.example.test", "qtype": TYPE_A, "rclass": CLASS_IN, "ttl": 240,
             "value": "10.0.2.20", "rdata_hex": "0a000214"},
        ],
    }
    write_json(SAMPLE_DIR / "zone.json", zone)
    corrupt = dict(zone)
    corrupt["records"] = "not-a-list"
    write_json(SAMPLE_DIR / "zone_corrupt.json", corrupt)

    v1 = [
        {"version": "v1", "name": "legacy.example.test", "qtype": TYPE_A, "value": "A 10.0.1.10", "ttl": 120},
        {"version": "v1", "name": "old.example.test", "qtype": TYPE_A, "value": "A 10.0.1.11", "ttl": 60},
    ]
    v2 = [
        {"version": "v2", "name": "www.example.test", "qtype": TYPE_A, "value": "A 10.0.0.5",
         "ttl": 300, "rcode": 0, "aa": True, "source": "authoritative"},
        {"version": "v2", "name": "nodata.example.test", "qtype": TYPE_A, "value": "NEGATIVE rcode=3",
         "ttl": 60, "rcode": 3, "aa": True, "source": "authoritative"},
    ]
    v3 = [
        {"version": "v3", "name": "edge.example.test", "qtype": TYPE_A, "value": "A 10.0.0.6",
         "ttl": 180, "rcode": 0, "aa": False, "source": "upstream",
         "ecs_prefix": "192.0.2.0/24", "hits": 4, "seq": 100},
        {"version": "v3", "name": "edge.example.test", "qtype": TYPE_A, "value": "A 10.0.0.7",
         "ttl": 180, "rcode": 0, "aa": False, "source": "upstream",
         "ecs_prefix": "198.51.100.0/24", "hits": 1, "seq": 101},
    ]
    write_json(SAMPLE_DIR / "cache_v1.json", v1)
    write_json(SAMPLE_DIR / "cache_v2.json", v2)
    write_json(SAMPLE_DIR / "cache_v3.json", v3)
    write_json(SAMPLE_DIR / "cache_bad_version.json", [
        {"version": "v9", "name": "x.example.test", "qtype": TYPE_A, "value": "A 10.0.9.9", "ttl": 30}
    ])
    write_json(SAMPLE_DIR / "cache_seq_regression.json", [
        v3[0],
        {"version": "v3", "name": "edge.example.test", "qtype": TYPE_A, "value": "A 10.0.0.8",
         "ttl": 180, "rcode": 0, "aa": False, "source": "upstream",
         "ecs_prefix": None, "hits": 0, "seq": 50},
    ])

    upstream = [
        {"name": "www.example.test", "qtype": TYPE_A, "answers": [
            {"name": "www.example.test", "qtype": TYPE_CNAME, "ttl": 300, "value": "edge.example.test", "rdata_hex": ""},
            {"name": "edge.example.test", "qtype": TYPE_A, "ttl": 300, "value": "10.0.0.5", "rdata_hex": "0a000005"},
        ]},
        {"name": "edge.example.test", "qtype": TYPE_A, "answers": [
            {"name": "edge.example.test", "qtype": TYPE_A, "ttl": 180, "value": "10.0.0.6", "rdata_hex": "0a000006"},
        ]},
        {"name": "forward.example.test", "qtype": TYPE_A, "answers": [
            {"name": "forward.example.test", "qtype": TYPE_A, "ttl": 240, "value": "10.0.2.20", "rdata_hex": "0a000214"},
        ]},
    ]
    write_json(SAMPLE_DIR / "upstream_answers.json", upstream)
    write_json(SAMPLE_DIR / "upstream_u1.json", {
        "format": "u1", "txn_id": 0x1234,
        "question": {"name": "edge.example.test", "qtype": TYPE_A, "qclass": CLASS_IN},
        "answers": [{"name": "edge.example.test", "qtype": TYPE_A, "value": "A 10.0.0.6", "ttl": 180}],
    })
    write_json(SAMPLE_DIR / "upstream_u2.json", {
        "format": "u2", "txn_id": 0x1234,
        "question": {"name": "edge.example.test", "qtype": TYPE_A, "qclass": CLASS_IN},
        "answers": [{"name": "edge.example.test", "qtype": TYPE_A, "value": "A 10.0.0.6", "ttl": 180}],
        "edns": True, "dnssec_ok": True,
    })

    question = Question("www.example.test", TYPE_A)
    query = DnsMessage(0x1234, 0x0100, [question])
    (SAMPLE_DIR / "wire_query.bin").write_bytes(encode_message(query))
    response = DnsMessage(
        0x1234,
        build_response_flags(rcode=0, aa=True),
        [question],
        [
            ResourceRecord("www.example.test", TYPE_CNAME, CLASS_IN, 300, b"", "edge.example.test"),
            ResourceRecord("edge.example.test", TYPE_A, CLASS_IN, 300, bytes([10, 0, 0, 5]), "10.0.0.5"),
        ],
    )
    (SAMPLE_DIR / "wire_response.bin").write_bytes(encode_message(response))
    print(f"samples written to {SAMPLE_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
