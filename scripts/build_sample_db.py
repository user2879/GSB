"""Build the deterministic sample database (entry-point wrapper)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pool_workbench.sample_db import build_sample_db  # noqa: E402


def main() -> int:
    workspace = Path(__file__).resolve().parent.parent / "workspace"
    db_path = workspace / "sample.sqlite"
    meta = build_sample_db(db_path)
    print(f"sample database built at {db_path}")
    print(meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
