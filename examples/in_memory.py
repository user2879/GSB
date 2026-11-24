"""Minimal library integration example using the committed local sample."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from colmv.changes import load_changes
from colmv.engine import MaterializationEngine
from colmv.replay import split_changes
from colmv.schema import load_config

def run(path_mode: str) -> MaterializationEngine:
    config = load_config(
        ROOT / "samples" / "base" / "catalog.json",
        ROOT / "samples" / "views.json",
    )
    changes = load_changes(ROOT / "samples" / "changes.ndjson", config.table)
    engine = MaterializationEngine(config, budget=1, path_mode=path_mode)
    for batch_no, batch in enumerate(split_changes(changes, [3, 6, 3, 6, 2]), start=1):
        engine.process_batch(batch, batch_no=batch_no)
    return engine


def main() -> None:
    resident = run("resident")
    rescan = run("rescan")
    if resident.materialized_bytes() != rescan.materialized_bytes():
        raise SystemExit("E_DIVERGE: paths are not byte-identical")
    print(f"byte_length={len(resident.materialized_bytes())}")
    print(f"digest={resident.materialized_digest()}")
    print(f"resident hits={resident.total_hits} replacements={resident.total_replacements}")
    print(f"rescan hits={rescan.total_hits} rescans={rescan.total_rescans}")
    print(f"groups={len(resident.group_state_payload())}")


if __name__ == "__main__":
    main()
