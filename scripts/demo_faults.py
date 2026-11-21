"""Inject the three required fault classes and report their business codes."""

import json
import os
import sqlite3
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from hotgray.errors import ReplayError, SnapshotError
from hotgray.gateway import Gateway
from hotgray.samples import build_sample_declaration
from hotgray.snapshot import authoritative_recompute

DECL_PATH = os.path.join(ROOT, "samples", "gateway_declaration.json")


def fresh_gateway(db_path):
    if os.path.exists(db_path):
        os.remove(db_path)
    declaration = build_sample_declaration()
    with open(DECL_PATH, "w", encoding="utf-8") as handle:
        json.dump(declaration, handle, ensure_ascii=False, indent=2)
    gateway = Gateway(DECL_PATH, db_path)
    gateway.start()
    for request in declaration["request_stream"]:
        gateway.admit_request(request)
    gateway.flush_all()
    gateway.rebuild_snapshot()
    return gateway, declaration


def inject_and_verify(fault, db_path):
    gateway, declaration = fresh_gateway(db_path)
    conn = sqlite3.connect(db_path)
    if fault == "corrupt_reply":
        row = conn.execute(
            "SELECT ledger_seq FROM hits ORDER BY ledger_seq LIMIT 1"
        ).fetchone()
        conn.execute("UPDATE hits SET output_digest=? WHERE ledger_seq=?",
                     ("0" * 64, row[0]))
        expected = SnapshotError
    elif fault == "missing_param_version":
        row = conn.execute(
            "SELECT backend_id, backend_seq FROM param_versions "
            "ORDER BY backend_seq DESC LIMIT 1"
        ).fetchone()
        conn.execute("DELETE FROM param_versions WHERE backend_id=? "
                     "AND backend_seq=?", row)
        expected = ReplayError
    else:
        row = conn.execute(
            "SELECT ledger_seq FROM hits ORDER BY ledger_seq LIMIT 1"
        ).fetchone()
        conn.execute("UPDATE hits SET content_digest=? WHERE ledger_seq=?",
                     ("f" * 64, row[0]))
        expected = ReplayError
    conn.commit()
    conn.close()
    try:
        authoritative_recompute(declaration, gateway.store)
        raise AssertionError(f"{fault} not detected")
    except expected as exc:
        return {"injected": fault, "code": exc.code, "field": exc.field,
                "message": str(exc)}
    finally:
        gateway.stop()


def main():
    os.makedirs(os.path.join(ROOT, "runtime"), exist_ok=True)
    results = []
    for fault in ("corrupt_reply", "missing_param_version", "ledger_digest"):
        db_path = os.path.join(ROOT, "runtime", f"fault_{fault}.sqlite3")
        results.append(inject_and_verify(fault, db_path))
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
