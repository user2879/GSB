"""Fault injection that only mutates input files / parameters, never source."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from .spec import canonical_json
from hashlib import sha256


def inject_bad_spec(src: Path, dst: Path) -> Path:
    """Corrupt a config declaration: negative TTL and unknown subscribed key."""
    data = json.loads(src.read_text(encoding="utf-8"))
    data["configs"][0]["ttl_seconds"] = -1
    data["clients"][0]["subscribed_keys"].append("does.not.exist")
    dst.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return dst


def inject_bad_lease(src: Path, dst: Path) -> Path:
    """Corrupt a lease record: wrong value digest and dangling parent ref."""
    data = json.loads(src.read_text(encoding="utf-8"))
    for record in data:
        if record.get("format") == "r3":
            record["value_digest"] = sha256(b"tampered").hexdigest()
            record["parent_lease_id"] = "L:nonexistent:0000"
    dst.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return dst


def inject_cursor_ahead(db_path: Path, client_id: str, jump: int = 999) -> None:
    """Force a cursor beyond any ledger-recorded push (parameter-level fault)."""
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    row = conn.execute(
        "SELECT MAX(config_version) AS m FROM ledger WHERE event_type='config_push' "
        "AND client_id=?", (client_id,)).fetchone()
    current = int(row[0] or 0)
    conn.execute(
        "INSERT INTO client_cursors(client_id,seen_config_version) VALUES(?,?) "
        "ON CONFLICT(client_id) DO UPDATE SET seen_config_version=excluded."
        "seen_config_version", (client_id, current + jump))
    conn.commit()
    conn.close()


def copy_dir(src: Path, dst: Path) -> Path:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return dst
