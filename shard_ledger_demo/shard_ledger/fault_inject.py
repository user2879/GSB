"""Fault injection that only mutates input files or runtime parameters."""

import json
import os
import shutil

from .errors import SnapshotError


def inject_corrupt_manifest(source_manifest: str, workdir: str) -> str:
    target = os.path.join(workdir, "manifest_corrupt.json")
    shutil.copyfile(source_manifest, target)
    with open(target, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    data["nodes"][0]["concurrency"] = -1
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    handle = open(target, "a", encoding="utf-8")
    handle.write("\n{ broken json tail")
    handle.close()
    return target


def inject_cycle_manifest(source_manifest: str, workdir: str) -> str:
    target = os.path.join(workdir, "manifest_cycle.json")
    shutil.copyfile(source_manifest, target)
    with open(target, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    first = data["shards"][0]["shard_id"]
    last = data["shards"][-1]["shard_id"]
    deps = list(data["shards"][-1].get("deps", []))
    if first not in deps:
        deps.append(first)
    data["shards"][-1]["deps"] = deps
    if last not in data["shards"][0].get("deps", []):
        data["shards"][0]["deps"] = list(data["shards"][0].get("deps", [])) + [last]
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    return target


def inject_snapshot_mismatch(ledger_path: str):
    import sqlite3

    if not os.path.exists(ledger_path):
        raise SnapshotError("ledger file missing for snapshot injection", entity=ledger_path, field="file")
    conn = sqlite3.connect(ledger_path)
    row = conn.execute("SELECT seq, digest FROM records ORDER BY seq LIMIT 1 OFFSET 1").fetchone()
    if row is None:
        row = conn.execute("SELECT seq, digest FROM records ORDER BY seq LIMIT 1").fetchone()
    seq, _digest = row
    conn.execute("UPDATE records SET digest=? WHERE seq=?", ("tampered" + "0" * 55, seq))
    conn.commit()
    conn.close()
    return seq
