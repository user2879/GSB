"""Generate the deterministic demo workspace used by tests and acceptance runs.

Outputs under ``demo/``:

* ``config.json``                 - labels, features, classifiers, thresholds,
                                   quantization, concurrency and budget;
* ``samples/``                    - generated PNG + annotation JSON pairs;
* ``edit_plan.json``              - incremental batches covering every edit op;
* ``injections.json``             - validation-stage input fault injection;
* ``budget_injections.json``      - smaller helper config set used in tests.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mlimgedit.config import load_config_dict  # noqa: E402
from mlimgedit.datagen import generate_dataset  # noqa: E402

LABELS = ("rectangle", "ellipse", "triangle", "circle_red")
SAMPLE_COUNT = 12
WIDTH, HEIGHT = 160, 120
BINS = 6
HOG_CELL = 8


def build_config(*, threads: int = 4, budget: int = 50000, snapshot_every: int = 1) -> dict:
    return {
        "labels": list(LABELS),
        "positive_ratios": {
            "rectangle": 0.8,
            "ellipse": 0.7,
            "triangle": 0.6,
            "circle_red": 0.5,
        },
        "feature": {"mode": "color_hog", "bins": BINS, "hog_cell": HOG_CELL},
        "classifiers": {
            "logreg": {"learning_rate": 0.12, "epochs": 20, "l2": 0.001},
            "centroid": {"distance": "euclidean"},
            "knn": {"k": 3, "vote": "distance", "distance": "euclidean"},
        },
        "threshold": 0.5,
        "label_thresholds": {},
        "quantization": {"quant_level": "Q1.15", "frac_bits": 15, "rounding": "HALF_UP"},
        "runtime": {
            "threads": threads,
            "feature_point_budget": budget,
            "snapshot_every": snapshot_every,
        },
        "image_size": [WIDTH, HEIGHT],
        "seed": 20261123,
    }


def build_edit_plan() -> dict:
    return {
        "base_config": "config.json",
        "base_sample_dir": "samples",
        "batches": [
            {
                "id": "batch_00_initial",
                "edits": [],
            },
            {
                "id": "batch_01_add_sample",
                "edits": [
                    {
                        "op": "add_sample",
                        "sample": {"name": "sample_added_000", "index": 101, "variant": 0},
                    }
                ],
            },
            {
                "id": "batch_02_replace_sample",
                "edits": [
                    {
                        "op": "replace_sample",
                        "sample": {"name": "sample_003", "index": 3, "variant": 7},
                    }
                ],
            },
            {
                "id": "batch_03_add_label",
                "edits": [{"op": "add_label", "label": "diagonal_blue", "ratio": 0.45}],
            },
            {
                "id": "batch_04_threshold_and_hyperparams",
                "edits": [
                    {
                        "op": "update_config",
                        "config": {
                            "threshold": 0.55,
                            "classifiers": {
                                "logreg": {"learning_rate": 0.08, "epochs": 30, "l2": 0.002}
                            },
                        },
                    }
                ],
            },
            {
                "id": "batch_05_remove_label",
                "edits": [{"op": "remove_label", "label": "circle_red"}],
            },
            {
                "id": "batch_06_switch_feature",
                "edits": [
                    {"op": "update_config", "config": {"feature": {"mode": "hog"}}}
                ],
            },
            {
                "id": "batch_07_delete_and_label_threshold",
                "edits": [
                    {"op": "delete_sample", "sample": "sample_added_000"},
                    {
                        "op": "update_config",
                        "config": {"label_thresholds": {"triangle": 0.6}},
                    },
                ],
            },
        ],
    }


def build_injections() -> dict:
    return {
        "injections": [
            {
                "id": "INJ-001",
                "sample": "sample_001",
                "stage": "validation",
                "fault": "truncate_png",
                "enabled": True,
            },
            {
                "id": "INJ-002",
                "sample": "sample_004",
                "stage": "validation",
                "fault": "unknown_label",
                "enabled": True,
            },
            {
                "id": "INJ-003",
                "sample": "sample_006",
                "stage": "validation",
                "fault": "bbox_out_of_range",
                "enabled": True,
            },
            {
                "id": "INJ-004",
                "sample": "sample_008",
                "stage": "validation",
                "fault": "drop_label",
                "enabled": True,
            },
        ]
    }


def main(output_dir: str = "demo") -> Path:
    root = ROOT / output_dir
    samples_dir = root / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    config_dict = build_config()
    config = load_config_dict(config_dict)
    (root / "config.json").write_text(
        json.dumps(config_dict, sort_keys=True, indent=2), encoding="utf-8"
    )
    names = generate_dataset(samples_dir, config, SAMPLE_COUNT)
    (root / "edit_plan.json").write_text(
        json.dumps(build_edit_plan(), sort_keys=True, indent=2), encoding="utf-8"
    )
    (root / "injections.json").write_text(
        json.dumps(build_injections(), sort_keys=True, indent=2), encoding="utf-8"
    )
    print(f"generated {len(names)} samples under {samples_dir}")
    print(f"demo root: {root}")
    return root


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "demo")
