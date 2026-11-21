"""Generate deterministic local fixture data (manifest, nodes, requests)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

KEYS = ["cache_ttl", "max_conn", "batch_size", "timeout_ms", "retry_limit"]


def build_manifest(count: int) -> dict:
    revisions = []
    parent = 0
    for seq in range(1, count + 1):
        key = KEYS[(seq - 1) % len(KEYS)]
        changes = {key: 100 + seq * 7, f"rev_marker_{seq}": seq}
        revisions.append({
            "rev_seq": seq,
            "parent": parent,
            "changes": changes,
        })
        parent = seq
    return {"revisions": revisions}


def build_nodes(node_count: int, base_port: int) -> dict:
    return {
        "nodes": [
            {"node_id": f"node-{idx + 1}", "host": "127.0.0.1",
             "port": base_port + idx}
            for idx in range(node_count)
        ]
    }


def build_requests(revision_count: int, node_count: int) -> dict:
    requests = []
    for seq in range(1, revision_count + 1):
        node_id = f"node-{(seq - 1) % node_count + 1}"
        requests.append({
            "node_id": node_id,
            "rev_seq": seq,
            "low_seq": seq,
            "high_seq": seq,
        })
    return {"requests": requests}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="runtime/data")
    parser.add_argument("--revisions", type=int, default=8)
    parser.add_argument("--nodes", type=int, default=3)
    parser.add_argument("--base-port", type=int, default=49152)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(
        json.dumps(build_manifest(args.revisions), indent=2), encoding="utf-8")
    (out / "nodes.json").write_text(
        json.dumps(build_nodes(args.nodes, args.base_port), indent=2), encoding="utf-8")
    (out / "requests.json").write_text(
        json.dumps(build_requests(args.revisions, args.nodes), indent=2),
        encoding="utf-8")
    print(f"fixtures written to {out}")


if __name__ == "__main__":
    main()
