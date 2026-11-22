"""SQLite-persisted block store with checkpointed, resumable format upgrade.

Layout inside a data directory:
  blocks.db       SQLite database
    blocks(block_id PK, format, payload)          current block payloads
    blocks_backup(block_id, format, payload, ...) append-only read-only backups
  checkpoint.json upgrade checkpoint: {"current_format": "b3",
                                       "upgraded": [block_id, ...]}

Upgrade protocol (per block, in sorted block_id order):
  1. skip block_ids already listed in the checkpoint (never reprocessed,
     even after a forced kill);
  2. decode + validate the old payload (E_FORMAT on violation);
  3. upgrade in memory to CURRENT_FORMAT;
  4. in ONE SQLite transaction: insert the old payload into blocks_backup
     (INSERT OR IGNORE; backups are never updated or deleted) and replace
     the row in blocks;
  5. only then append the block_id to checkpoint.json (atomic os.replace).
A kill between (4) and (5) reprocesses exactly one block on restart; the
backup keeps the first (original) payload thanks to INSERT OR IGNORE.
"""

import json
import os
import sqlite3

from .formats import CURRENT_FORMAT, Block, decode_block, encode_block, upgrade_block

_SCHEMA = """
CREATE TABLE IF NOT EXISTS blocks(
  block_id TEXT PRIMARY KEY,
  format   TEXT NOT NULL,
  payload  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS blocks_backup(
  block_id     TEXT NOT NULL,
  format       TEXT NOT NULL,
  payload      TEXT NOT NULL,
  backed_up_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
  PRIMARY KEY(block_id, format)
);
"""


class BlockStore:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.db_path = os.path.join(data_dir, "blocks.db")
        self.checkpoint_path = os.path.join(data_dir, "checkpoint.json")
        self.conn = sqlite3.connect(self.db_path)
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def put_block(self, block, version=None):
        payload = encode_block(block, version)
        fmt = version or block.format_version
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO blocks(block_id, format, payload) VALUES(?,?,?)",
                (block.block_id, fmt, payload))

    def get_payload(self, block_id):
        row = self.conn.execute(
            "SELECT format, payload FROM blocks WHERE block_id=?",
            (block_id,)).fetchone()
        if row is None:
            raise KeyError(block_id)
        return row[0], row[1]

    def get_block(self, block_id):
        return decode_block(self.get_payload(block_id)[1])

    def list_block_ids(self):
        return [r[0] for r in self.conn.execute(
            "SELECT block_id FROM blocks ORDER BY block_id")]

    def list_backups(self):
        return [r[0] for r in self.conn.execute(
            "SELECT DISTINCT block_id FROM blocks_backup ORDER BY block_id")]

    def read_checkpoint(self):
        if not os.path.exists(self.checkpoint_path):
            return {"current_format": CURRENT_FORMAT, "upgraded": []}
        with open(self.checkpoint_path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def _write_checkpoint(self, state):
        tmp = self.checkpoint_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=1, sort_keys=True)
            fh.write("\n")
        os.replace(tmp, self.checkpoint_path)

    def upgrade_all(self, on_upgrade=None):
        """Upgrade every stale block to CURRENT_FORMAT. Resumable via checkpoint.

        Returns the list of block_ids upgraded during THIS call.
        """
        state = self.read_checkpoint()
        done = set(state.get("upgraded", []))
        upgraded_now = []
        for block_id in self.list_block_ids():
            if block_id in done:
                continue  # already upgraded before the interruption
            fmt, payload = self.get_payload(block_id)
            block = decode_block(payload)  # E_FORMAT on invalid old blocks
            if block.format_version == CURRENT_FORMAT:
                done.add(block_id)
                state["upgraded"] = sorted(done)
                self._write_checkpoint(state)
                continue
            upgraded = upgrade_block(block)
            new_payload = encode_block(upgraded, CURRENT_FORMAT)
            with self.conn:  # atomic: backup + replace in one transaction
                self.conn.execute(
                    "INSERT OR IGNORE INTO blocks_backup(block_id, format, payload)"
                    " VALUES(?,?,?)", (block_id, fmt, payload))
                self.conn.execute(
                    "UPDATE blocks SET format=?, payload=? WHERE block_id=?",
                    (CURRENT_FORMAT, new_payload, block_id))
            done.add(block_id)
            state["upgraded"] = sorted(done)
            self._write_checkpoint(state)  # checkpoint AFTER the block commits
            upgraded_now.append(block_id)
            if on_upgrade:
                on_upgrade(block_id)
        return upgraded_now

    def close(self):
        self.conn.close()


def block_from_columns(block_id, columns, quant_scale_exp, batch_max_rows,
                       format_version=CURRENT_FORMAT):
    """columns: list of (name, dtype, null_bitmap, values)."""
    from .formats import Column
    cols = [Column(name=n, dtype=t, null_bitmap=b, values=list(v))
            for n, t, b, v in columns]
    row_count = len(columns[0][3]) if columns else 0
    return Block(block_id=block_id, row_count=row_count, columns=cols,
                 quant_scale_exp=quant_scale_exp, batch_max_rows=batch_max_rows,
                 format_version=format_version)
