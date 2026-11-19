from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from srebudget.samples import write_samples


def main() -> int:
    chain_path, trace_path = write_samples("data")
    print(
        json.dumps(
            {
                "ok": True,
                "chain": str(chain_path),
                "trace": str(trace_path),
                "note": "samples are deterministic and generated locally",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
