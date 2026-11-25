#!/usr/bin/env python3
"""Generate deterministic local sample table artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from category_cross_diagnostic.config import load_bundle  # noqa: E402
from category_cross_diagnostic.sample_data import generate_table  # noqa: E402


def main() -> int:
    features_path = ROOT / "examples" / "feature_table.yaml"
    strategy_path = ROOT / "examples" / "strategy.yaml"
    output_dir = ROOT / "artifacts" / "sample_data"
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_bundle(features_path, strategy_path)
    table = generate_table(bundle)

    csv_path = output_dir / "sample_table.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(table.column_names())
        for row_index in range(table.row_count):
            writer.writerow(
                [
                    "" if table.values[name][row_index] is None else table.values[name][row_index]
                    for name in table.column_names()
                ]
            )

    (output_dir / "missing_bitmap.json").write_text(
        json.dumps(table.missing_bitmap, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "column_fingerprints.json").write_text(
        json.dumps(
            {
                "table_fingerprint": table.table_fingerprint,
                "columns": table.column_fingerprints,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {csv_path.relative_to(ROOT)} and JSON sidecars to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
