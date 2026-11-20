"""Fault injection for the three anomaly classes.

Each fault only mutates input files or parameters; the normal pipeline then
reports the corresponding business code (E_SPEC / E_SEED / E_SNAPSHOT) and
writes a structured log record.
"""

import json
import shutil
import sqlite3


def corrupt_spec(spec_path):
    """Fault 1: truncate the declaration file -> E_SPEC on next load."""
    backup = str(spec_path) + ".bak"
    shutil.copyfile(spec_path, backup)
    with open(spec_path, "r", encoding="utf-8") as fh:
        content = fh.read()
    with open(spec_path, "w", encoding="utf-8") as fh:
        fh.write(content[: max(1, len(content) // 2)])  # invalid JSON
    return {"fault": "corrupt-spec", "backup": backup}


def seed_mismatch(spec_path):
    """Fault 2: change the seed parameter -> timeline digest no longer
    matches the ledger -> E_SEED on verify."""
    backup = str(spec_path) + ".bak"
    shutil.copyfile(spec_path, backup)
    with open(spec_path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    doc["seed"] = doc.get("seed", 0) + 1
    with open(spec_path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True)
    return {"fault": "seed-mismatch", "backup": backup, "new_seed": doc["seed"]}


def ledger_digest_mismatch(ledger_path):
    """Fault 3: tamper a ledger *data* row without refreshing the content
    digest -> E_SNAPSHOT on next open."""
    conn = sqlite3.connect(str(ledger_path))
    try:
        row = conn.execute("SELECT MIN(seq) FROM injections").fetchone()
        if row and row[0] is not None:
            conn.execute("UPDATE injections SET planned_start_ms"
                         " = planned_start_ms + 1 WHERE seq=?", (row[0],))
        else:
            conn.execute(
                "INSERT INTO injections(seq,stable_key,itype,target,"
                " params_json,planned_start_ms,state,committed)"
                " VALUES(-1,'tampered','latency','none','{}',0,'failed',1)")
        conn.commit()
    finally:
        conn.close()
    return {"fault": "ledger-digest", "ledger": str(ledger_path)}
