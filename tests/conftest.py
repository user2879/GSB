import json
from pathlib import Path

import pytest

from kvdiag.samples import generate_keyspace, generate_renewals


def write_tree(root, *, seed=42, shard_count=4, rows_per_shard=6, ttl_ms=120,
               renew_interval_ms=40, max_inflight=2, tolerance=0):
    root = Path(root)
    (root / "samples").mkdir(parents=True, exist_ok=True)
    decl, samples = generate_keyspace(
        seed=seed, shard_count=shard_count, rows_per_shard=rows_per_shard,
        ttl_ms=ttl_ms, renew_interval_ms=renew_interval_ms,
        max_inflight=max_inflight, tolerance=tolerance)
    (root / "keyspace.json").write_text(json.dumps(decl, indent=2),
                                        encoding="utf-8")
    for shard_id, doc in samples.items():
        (root / "samples" / ("%s.json" % shard_id)).write_text(
            json.dumps(doc, indent=2), encoding="utf-8")
    (root / "renewals.json").write_text(
        json.dumps(generate_renewals(decl), indent=2), encoding="utf-8")
    return decl


@pytest.fixture
def tree(tmp_path):
    root = tmp_path / "ex"
    decl = write_tree(root)
    return {"root": root, "spec": root / "keyspace.json",
            "renewals": root / "renewals.json", "decl": decl,
            "db": tmp_path / "t.sqlite"}
