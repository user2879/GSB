"""Generate deterministic reference/current sample data from a YAML config.

Usage:
    python scripts/generate_sample_data.py --config configs/default_monitor.yaml \
        --tier small --output output/sample
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drift_monitor.config import load_config
from drift_monitor.datagen import expanded_columns, generate_pair


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--tier", default="small")
    parser.add_argument("--output", default="output/sample")
    args = parser.parse_args()

    config = load_config(args.config)
    tier = next(t for t in config.data.size_tiers if t.name == args.tier)
    reference, current = generate_pair(config.data, tier)
    columns = expanded_columns(config.data, tier)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    for period, dataset in (("reference", reference), ("current", current)):
        rows = []
        for idx in range(dataset.n_samples):
            row = {"row_id": idx}
            for column in dataset.columns:
                row[column.name] = column.values[idx]
                row[f"{column.name}__missing"] = bool(column.missing_mask[idx])
            rows.append(row)
        (out_dir / f"{period}_{args.tier}.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "tier": tier.name,
        "samples": tier.samples,
        "feature_multiplier": tier.feature_multiplier,
        "columns": [c.name for c in columns],
        "missing_rate": config.data.missing_rate,
        "seed": config.data.seed,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
