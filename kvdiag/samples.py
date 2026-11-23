"""Deterministic sample and declaration generation from a fixed seed."""

import json
import random
from pathlib import Path

ALPHABET = ["alpha", "beta", "gamma", "delta", "omega", "kappa", "sigma", "theta"]


def generate_keyspace(seed=42, shard_count=4, rows_per_shard=8,
                      key_space=1 << 16, ttl_ms=300, renew_interval_ms=100,
                      max_inflight=2, tolerance=0):
    """Build the declaration dict plus per-shard sample docs (in memory)."""
    width = key_space // shard_count
    shards = []
    samples = {}
    for i in range(shard_count):
        shard_id = "shard-%d" % i
        lo, hi = i * width, (i + 1) * width - 1
        rng = random.Random(seed * 100003 + i)
        keys = sorted(rng.sample(range(lo, hi + 1), rows_per_shard))
        rows = []
        for key in keys:
            rows.append({
                "key": key,
                "value": "%s-%04d" % (rng.choice(ALPHABET), rng.randint(0, 9999)),
                "score": rng.randint(0, 99),
            })
        shards.append({
            "shard_id": shard_id,
            "range": [lo, hi],
            "sample_file": "samples/%s.json" % shard_id,
            "index_rules": [
                {"column": "value", "kind": "prefix", "prefix_len": 2},
                {"column": "score", "kind": "range", "width": 10},
            ],
            "lease": {"ttl_ms": ttl_ms, "renew_interval_ms": renew_interval_ms,
                      "max_inflight_renewals": max_inflight},
            "tolerance": tolerance,
        })
        samples[shard_id] = {"shard_id": shard_id, "rows": rows}
    declaration = {
        "format_version": 1,
        "seed": seed,
        "mode": "high_byte",
        "shards": shards,
    }
    return declaration, samples


def generate_renewals(declaration):
    """Default renewal plan: renew the first lease of each shard once."""
    renewals = []
    for shard in declaration["shards"]:
        renewals.append({"shard_id": shard["shard_id"], "lease_ordinal": 0,
                         "expected_version": 1})
    return {"renewals": renewals}


def generate_injection_manifest():
    """Parameter list for the three supported fault injections."""
    return {
        "injections": [
            {"name": "sample-corrupt", "target": "samples/shard-1.json",
             "expects": "E_SPEC",
             "args": {"shard_id": "shard-1"}},
            {"name": "lease-version-regression", "target": "renewals.json",
             "expects": "E_LEASE",
             "args": {"lease_ordinal": 0}},
            {"name": "ledger-digest-mismatch", "target": "session_snapshot.json",
             "expects": "E_SNAPSHOT",
             "args": {}},
        ]
    }


def write_example_tree(out_dir, seed=42):
    """Write declaration, samples, renewal plan and injection manifest."""
    out = Path(out_dir)
    (out / "samples").mkdir(parents=True, exist_ok=True)
    declaration, samples = generate_keyspace(seed=seed)
    (out / "keyspace.json").write_text(
        json.dumps(declaration, indent=2, sort_keys=True), encoding="utf-8")
    for shard_id, doc in samples.items():
        (out / "samples" / ("%s.json" % shard_id)).write_text(
            json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")
    (out / "renewals.json").write_text(
        json.dumps(generate_renewals(declaration), indent=2, sort_keys=True),
        encoding="utf-8")
    (out / "injections.json").write_text(
        json.dumps(generate_injection_manifest(), indent=2, sort_keys=True),
        encoding="utf-8")
    return declaration
