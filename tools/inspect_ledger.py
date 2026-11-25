"""Read-only ledger inspection using the stdlib sqlite3 module (no CLI needed)."""

import argparse
import json
import sqlite3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("db_path")
    args = parser.parse_args()
    conn = sqlite3.connect(args.db_path)
    conn.row_factory = sqlite3.Row
    print("== meta ==")
    for row in conn.execute("SELECT key, value FROM meta ORDER BY key"):
        print(f"{row['key']} = {row['value']}")
    for table in ("events", "shard_outcomes", "resource_usage", "lease_records",
                  "freeze_records", "revoke_records", "progress"):
        count = conn.execute(f"SELECT COUNT(*) AS c FROM {table}").fetchone()["c"]
        print(f"{table}: {count} rows")
    print("== outcomes ==")
    for row in conn.execute("SELECT shard_id, status, node_id, lease_version, fixed_total FROM shard_outcomes ORDER BY shard_id"):
        print(json.dumps(dict(row), sort_keys=True))
    conn.close()


if __name__ == "__main__":
    main()
