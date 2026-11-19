"""Fault injection harness: mutates only input files/parameters.

Three supported injections, each mapped to its mandated error code:

inject_corrupt_event_file  -> E_SPEC   (malformed NDJSON line)
inject_timestamp_regression -> E_ORDER (regression beyond allowance)
inject_snapshot_digest      -> E_SNAPSHOT (content digest mismatch)

No implementation source file is modified; only copies under the runtime
directory are rewritten or a verification flag is toggled.
"""
from __future__ import annotations

import json
import os
import shutil

from .snapshot import read_snapshot


def inject_corrupt_event_file(source_ndjson: str, target_ndjson: str,
                              *, corrupt_line: int = 1) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(target_ndjson)),
                exist_ok=True)
    with open(source_ndjson, "r", encoding="utf-8") as handle:
        lines = handle.readlines()
    if corrupt_line < 1 or corrupt_line > len(lines):
        corrupt_line = 1
    lines[corrupt_line - 1] = '{"partition": "p0", "seq": 0, "ts_ms": }\n'
    with open(target_ndjson, "w", encoding="utf-8") as handle:
        handle.writelines(lines)
    return target_ndjson


def inject_timestamp_regression(source_ndjson: str, target_ndjson: str,
                                *, partition: str, seq: int,
                                new_ts_ms: int) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(target_ndjson)),
                exist_ok=True)
    changed = False
    with open(source_ndjson, "r", encoding="utf-8") as handle:
        lines = handle.readlines()
    with open(target_ndjson, "w", encoding="utf-8") as handle:
        for line in lines:
            if not line.strip():
                handle.write(line)
                continue
            event = json.loads(line)
            if (event.get("partition") == partition
                    and event.get("seq") == seq):
                event["ts_ms"] = new_ts_ms
                line = json.dumps(event, separators=(",", ":")) + "\n"
                changed = True
            handle.write(line)
    if not changed:
        raise ValueError(
            f"injection target {partition}#seq={seq} not found")
    return target_ndjson


def inject_snapshot_digest(source_snapshot: str,
                           target_snapshot: str) -> str:
    payload = read_snapshot(source_snapshot)
    os.makedirs(os.path.dirname(os.path.abspath(target_snapshot)),
                exist_ok=True)
    shutil.copyfile(source_snapshot, target_snapshot)
    # Mutate persisted content without recomputing the digest.
    payload["partition_views"][0]["finalized_count"] += 1
    payload["content_digest"] = "tampered-" + payload["content_digest"][:24]
    with open(target_snapshot, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True,
                  separators=(",", ":"))
        handle.write("\n")
    return target_snapshot
