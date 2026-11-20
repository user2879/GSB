"""Child process that leaves an uncommitted staging directory then hard-exits."""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from pointcloud_studio.persistence import AtomicStateStore


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: crash_child.py WORKSPACE")
    root = Path(sys.argv[1])
    store = AtomicStateStore(root)
    store.open()
    store.load_current()
    staging = store.staging_root / f"partial-operation-{os.getpid()}-{uuid.uuid4().hex[:10]}"
    staging.mkdir(parents=True, exist_ok=False)
    (staging / "points.bin.tmp").write_bytes(b"uncommitted-point-cloud-cache")
    (staging / "ledger.json.tmp").write_bytes(b'{"uncommitted":true}\n')
    # Deliberately do not call close(): the next opener must detect a forced exit.
    os._exit(23)


if __name__ == "__main__":
    main()
