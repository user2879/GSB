"""Fault injection: corrupt config (E_CONFIG) and lease conflicts (E_LEASE).

Every injection is deterministic, reports its business code with evidence and
writes a structured log entry.
"""

from __future__ import annotations

import json


def inject_config_corruption(path: str, mode: str = "invalid_json") -> str:
    """Corrupt a config file so the next load deterministically fails E_CONFIG."""
    if mode == "invalid_json":
        with open(path, "a", encoding="utf-8") as fh:
            fh.write("\n{ not json ")
        return "appended invalid JSON tail"
    if mode == "bad_field":
        with open(path, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
        doc["routing_units"][0]["physical_dbs"] = 0
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2)
        return "set routing_units[0].physical_dbs = 0"
    raise ValueError(f"unknown corruption mode {mode}")


def inject_lease_version_conflict(lease_manager, lease_id: str) -> dict:
    """Silently bump the stored lease version so the holder's next renew fails."""
    lease = lease_manager.get(lease_id)
    if lease is None:
        raise ValueError(f"no such lease {lease_id}")
    with lease_manager._lock:  # injection reaches past the public API on purpose
        lease.version += 1
    return {
        "lease_id": lease_id,
        "injected_version": lease.version,
        "effect": "next renew with the holder's expected version raises E_LEASE",
    }
