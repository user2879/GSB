"""Run the shipped example and write reports/latest.json."""

from __future__ import annotations

import json
from pathlib import Path

from multimodal_fusion import run_pipeline
from scripts.generate_demo_data import main as generate_demo_data


ROOT = Path(__file__).resolve().parent


def main() -> dict:
    input_dir = ROOT / "examples" / "input"
    if not (input_dir / "texts.jsonl").exists() or not any((input_dir / "images").glob("*.png")):
        generate_demo_data(input_dir)
    report = run_pipeline(
        config_path=ROOT / "examples" / "config.yaml",
        input_dir=input_dir,
        state_dir=ROOT / ".fusion_state",
        report_path=ROOT / "reports" / "latest.json",
    )
    print(json.dumps({
        "global_version": report["global_version"],
        "scored_samples": report["summary"]["scored_samples"],
        "boundary_samples": report["boundary_samples"],
        "failure_count": len(report["failure_list"]),
        "report": "reports/latest.json",
    }, sort_keys=True))
    return report


if __name__ == "__main__":
    main()
