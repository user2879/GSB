from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .formats import encode_v1, encode_v2, encode_v3
from .integrity import compute_shard_proof
from .netutil import free_loopback_port
from .spec import create_readonly_snapshot


def sample_spec_dict(*, allocate_ports: bool = True, quota_version: int = 1) -> dict[str, Any]:
    ports = [free_loopback_port() for _ in range(3)] if allocate_ports else [48621, 48622, 48623]
    return {
        "nodes": [
            {
                "node_id": "node-a",
                "host": "127.0.0.1",
                "port": ports[0],
                "capacity_slots": 2,
                "max_node_leases": 2,
                "seed": 1101,
            },
            {
                "node_id": "node-b",
                "host": "127.0.0.1",
                "port": ports[1],
                "capacity_slots": 2,
                "max_node_leases": 2,
                "seed": 2202,
            },
            {
                "node_id": "node-c",
                "host": "127.0.0.1",
                "port": ports[2],
                "capacity_slots": 1,
                "max_node_leases": 1,
                "seed": 3303,
            },
        ]
    }


CONTENT_DICT = {
    "contents": [
        {
            "content_key": "ck-demo-2026-1126",
            "total_bytes": 2048,
            "shard_count": 4,
            "sample_stride": 8,
            "integer_scale": 1000,
            "seed": 20261126,
            "invalid_refs": [
                {"ref_id": "ref-invalid-shard-0002", "shard_id": "shard-0002", "reason": "sample-declared-invalid-reference"}
            ],
        }
    ]
}


POLICY_DICT = {
    "quota_version": 1,
    "ttl_ms": 300,
    "max_renewals": 2,
    "global_max_active_leases": 2,
    "global_byte_budget": 16384,
    "reaper_interval_ms": 20,
    "queue_timeout_ms": 3000,
    "socket_timeout_ms": 1500,
    "legacy_node_id": "node-a",
}


EVENT_RECORDS = [
    {"seq": 1, "event_type": "declaration_loaded", "source": "nodes.json"},
    {"seq": 2, "event_type": "content_manifest_loaded", "content_key": "ck-demo-2026-1126"},
    {"seq": 3, "event_type": "policy_loaded", "quota_version": 1},
    {"seq": 4, "event_type": "receipt_replaced", "format": "v1", "shard_id": "shard-0000"},
    {"seq": 5, "event_type": "quota_revision_scheduled", "from_version": 1, "to_version": 2},
    {"seq": 6, "event_type": "recovery_rehearsal_scheduled", "node_id": "node-a"},
]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_sample_tree(
    root: str | Path,
    *,
    allocate_ports: bool = True,
    create_snapshot: bool = True,
    quota_version: int = 1,
) -> Path:
    target = Path(root)
    target.mkdir(parents=True, exist_ok=True)
    nodes = sample_spec_dict(allocate_ports=allocate_ports, quota_version=quota_version)
    policy = dict(POLICY_DICT)
    policy["quota_version"] = quota_version
    write_json(target / "nodes.json", nodes)
    write_json(target / "content_manifest.json", CONTENT_DICT)
    write_json(target / "lease_policy.json", policy)
    event_text = "".join(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n" for record in EVENT_RECORDS)
    (target / "event_stream.jsonl").write_text(event_text, encoding="utf-8")

    from .spec import load_spec_dir

    spec = load_spec_dir(target)
    content = spec.content
    proofs = {
        (node_id, seq): compute_shard_proof(spec.node(node_id), content, seq)
        for seq, node_id in enumerate(["node-a", "node-b", "node-c", "node-a"])
    }
    p0 = proofs[("node-a", 0)]
    p1 = proofs[("node-b", 1)]
    p2 = proofs[("node-c", 2)]
    p3 = proofs[("node-a", 3)]

    (target / "receipts_v1.txt").write_text(encode_v1(p0.shard_id, p0.content_key, p0.digest) + "\n", encoding="utf-8")
    v2_lines = [encode_v2(p1)]
    (target / "receipts_v2.jsonl").write_text("\n".join(v2_lines) + "\n", encoding="utf-8")
    invalid_ref = {"ref_id": "ref-invalid-shard-0002", "shard_id": "shard-0002"}
    v3_lines = [
        encode_v3(p2, lease_version=1, quota_version=quota_version, invalid_ref=invalid_ref),
        encode_v3(p3, lease_version=1, quota_version=quota_version),
    ]
    (target / "receipts_v3.jsonl").write_text("\n".join(v3_lines) + "\n", encoding="utf-8")

    bad = dict(json.loads(encode_v3(p3, lease_version=1, quota_version=quota_version)))
    bad["digest"] = "0" * 64
    (target / "receipt_bad_digest_v3.json").write_text(json.dumps(bad, sort_keys=True) + "\n", encoding="utf-8")

    replacement = dict(json.loads(encode_v2(p3)))
    (target / "receipt_replacement_v2.json").write_text(json.dumps(replacement, sort_keys=True) + "\n", encoding="utf-8")

    if create_snapshot:
        create_readonly_snapshot(target, target / "raw_snapshot")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic local edge-ledger samples")
    parser.add_argument("--out", default="samples")
    parser.add_argument("--fixed-ports", action="store_true")
    args = parser.parse_args(argv)
    path = write_sample_tree(args.out, allocate_ports=not args.fixed_ports)
    print(f"sample spec written to {path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
