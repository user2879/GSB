"""独立验证器：只消费持久化目录，从链基重放并与在线演进逐版本比对。"""

from __future__ import annotations

from .engine import rebuild
from .store import Store


def verify(store_dir: str) -> dict:
    store = Store(store_dir)
    manifest = store.load_manifest()
    online = store.read_json("online_hashes.json")
    online_by_version = {entry["version"]: entry["state_hash"] for entry in online}

    replayed = rebuild(store)
    mismatches = []
    per_version = []
    for entry in replayed["per_version"]:
        version = entry["version"]
        expected = online_by_version.get(version)
        ok = expected == entry["state_hash"]
        per_version.append({
            "version": version,
            "replayed_hash": entry["state_hash"],
            "online_hash": expected,
            "match": ok,
        })
        if not ok:
            ops = store.load_ops(version) if store.has(f"logs/v{version}.jsonl") else []
            mismatches.append({
                "version": version,
                "replayed_hash": entry["state_hash"],
                "online_hash": expected,
                "participating_ops": [
                    {"event_id": op["event_id"], "peer_id": op["peer_id"],
                     "op_type": op["op_type"], "source_format": op["source_format"]}
                    for op in ops
                ],
            })
            break  # 定位首个不一致版本

    return {
        "head": manifest["head"],
        "base": replayed["base"],
        "versions_checked": len(per_version),
        "consistent": not mismatches,
        "first_mismatch": mismatches[0] if mismatches else None,
        "per_version": per_version,
    }
