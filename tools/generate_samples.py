"""Materialize deterministic sample JSON files from samples/spec.json."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bucketpack.canonical import write_json
from bucketpack.source import DATA_FORMAT_VERSION, generate_rows
from bucketpack.spec import load_spec


def main() -> int:
    spec = load_spec(ROOT / "samples" / "spec.json")
    for source in spec.sources:
        if source.data_file is None:
            continue
        payload = {
            "format_version": DATA_FORMAT_VERSION,
            "source_id": source.id,
            "seed": source.seed,
            "rows": generate_rows(source),
        }
        target = Path(spec.base_dir) / source.data_file
        write_json(target, payload)
        print(f"wrote {target.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
