"""Ledger assembly, versioning and atomic persistence.

Layout under <out>/ledger:
  ledger.json              committed current ledger (atomic replace)
  HEAD                     committed version stamp (atomic replace)
  snapshots/<version>.json immutable content-addressed snapshots

The version stamp is derived from the ledger content itself, so identical
inputs always produce byte-identical ledger files. Replacement decisions
compare version + hash; *.tmp files are never valid state and are removed
on startup, so recovery after a kill only ever sees committed ledgers.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import util

SCHEMA = "codecq-ledger/1"


def stamp_version(ledger_body: dict) -> str:
    """Deterministic content-addressed version stamp."""
    return "v1-" + util.hash_obj(ledger_body)[:16]


def finalize(ledger_body: dict) -> dict:
    """Attach per-section hashes, overall hash and the version stamp."""
    body = dict(ledger_body)
    body["schema"] = SCHEMA
    hashes = {
        "config": util.hash_obj(body["config"]),
        "source": util.hash_obj(body["source"]),
        "encodes": util.hash_obj(body["encodes"]),
        "isolation": util.hash_obj(body["isolation"]),
    }
    hashes["overall"] = util.hash_obj({k: body[k] for k in sorted(body)} | {"hashes.partial": hashes})
    body["hashes"] = hashes
    body["version"] = stamp_version(body)
    return body


def commit(ledger: dict, ledger_dir) -> dict:
    """Atomically commit the ledger; returns a commit report."""
    ledger_dir = Path(ledger_dir)
    util.clean_stale_tmp(ledger_dir)
    snapshots = ledger_dir / "snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)

    version = ledger["version"]
    data = (util.canonical_json(ledger) + "\n").encode("utf-8")
    digest = util.sha256_hex(data)

    snapshot_path = snapshots / f"{version}.json"
    if snapshot_path.exists():
        if util.sha256_hex(snapshot_path.read_bytes()) != digest:
            raise RuntimeError(f"snapshot hash collision/corruption at {snapshot_path}")
        snapshot_action = "kept"
    else:
        util.atomic_write(snapshot_path, data)
        snapshot_action = "created"

    head_path = ledger_dir / "HEAD"
    previous = head_path.read_text(encoding="utf-8").strip() if head_path.exists() else None
    util.atomic_write(ledger_dir / "ledger.json", data)
    util.atomic_write(head_path, (version + "\n").encode("ascii"))
    return {
        "version": version,
        "sha256": digest,
        "previous_version": previous,
        "replaced": previous is not None and previous != version,
        "snapshot": snapshot_action,
    }


def load_committed(ledger_dir) -> dict:
    """Load the committed ledger, verifying its version stamp and hash chain.

    Only ledger.json + HEAD are consulted; stray *.tmp files are ignored and
    removed. Raises if the committed content does not match its version stamp.
    """
    ledger_dir = Path(ledger_dir)
    util.clean_stale_tmp(ledger_dir)
    head = (ledger_dir / "HEAD").read_text(encoding="utf-8").strip()
    raw = (ledger_dir / "ledger.json").read_bytes()
    ledger = json.loads(raw.decode("utf-8"))
    if ledger.get("version") != head:
        raise RuntimeError(f"HEAD {head} does not match ledger version {ledger.get('version')}")
    body = {k: v for k, v in ledger.items() if k != "version"}
    if stamp_version(body) != head:
        raise RuntimeError("ledger content does not match its version stamp (tampered?)")
    return ledger
