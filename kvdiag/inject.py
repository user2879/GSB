"""Fault injection: only input files or parameters are altered, never code.

Three injection kinds map to business codes:
  sample-corrupt            -> E_SPEC      (rewrites a shard sample file)
  lease-version-regression  -> E_LEASE     (rewrites the renewal plan file)
  ledger-digest-mismatch    -> E_SNAPSHOT  (rewrites the session snapshot)
"""

import json
from pathlib import Path


def inject_sample_corrupt(spec_path, shard_id):
    """Move the first sample key of shard_id outside its declared range."""
    spec_path = Path(spec_path)
    decl = json.loads(spec_path.read_text(encoding="utf-8"))
    shard = next(s for s in decl["shards"] if s["shard_id"] == shard_id)
    sample_path = spec_path.parent / shard["sample_file"]
    doc = json.loads(sample_path.read_text(encoding="utf-8"))
    lo, hi = shard["range"]
    doc["rows"][0]["key"] = hi + 1  # outside [lo, hi]
    sample_path.write_text(json.dumps(doc, indent=2, sort_keys=True),
                           encoding="utf-8")
    return {"target": str(sample_path), "mutated_key": hi + 1}


def inject_lease_version_regression(renewals_path):
    """Regress the first renewal's expected version (optimistic check fails)."""
    path = Path(renewals_path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    first = doc["renewals"][0]
    first["expected_version"] = max(0, first["expected_version"] - 1)
    path.write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")
    return {"target": str(path), "expected_version": first["expected_version"]}


def inject_ledger_digest_mismatch(session_path):
    """Corrupt the session snapshot content digest."""
    path = Path(session_path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["content_digest"] = "0" * 64
    path.write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")
    return {"target": str(path), "content_digest": doc["content_digest"]}
