"""Simulate a forced kill mid-batch: begin a batch, write partial state
WITHOUT committing, then os._exit. The ledger must later classify this batch
as unknown and require a rerun; no exit report is produced.
"""

import os
import sqlite3
import sys


def main():
    path, spec_version, digest = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    conn = sqlite3.connect(path)
    cursor = conn.execute(
        "INSERT INTO batches(spec_version, format_version, content_digest, committed,"
        " started_ms) VALUES (?, 1, ?, 0, 0)",
        (spec_version, digest),
    )
    batch_id = cursor.lastrowid
    conn.commit()  # batch row itself is visible, still uncommitted
    conn.execute(
        "INSERT OR REPLACE INTO stage_summaries(stage_id, window_id, batch_id,"
        " summary, null_count, null_rule) VALUES ('partial', 'w0', ?, 'deadbeef', 0, 'x')",
        (batch_id,),
    )
    # Killed before the batch commit transaction: partial row is rolled back.
    os._exit(137)


if __name__ == "__main__":
    main()
