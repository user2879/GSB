"""Generate deterministic local YAML config and vibration descriptions."""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vibdet.config import DEFAULT_CONFIG


def build_descriptions():
    devices = {
        "DEV-A": {
            "seed": 1001,
            "segments": [
                {"id": "A-seg0", "length": 1024, "seed": 11,
                 "rpm_hz": 25.0, "harmonics": [1.0, 0.5, 0.25],
                 "noise_sigma": 0.05,
                 "missing": {"count": 12},
                 "resample": {"indices": [30, 640]},
                 "impulse": {"count": 3, "amplitude": 3.0}},
                {"id": "A-seg1", "length": 1024, "seed": 12,
                 "rpm_hz": 27.0, "harmonics": [1.0, 0.6, 0.3],
                 "noise_sigma": 0.08,
                 "fault": {"amplitude": 1.6},
                 "impulse": {"count": 2, "amplitude": 4.0}},
                {"id": "A-seg2", "length": 768, "seed": 13,
                 "rpm_hz": 24.0, "harmonics": [0.9, 0.4],
                 "noise_sigma": 0.04,
                 "missing": {"count": 6}},
            ],
        },
        "DEV-B": {
            "seed": 2002,
            "segments": [
                {"id": "B-seg0", "length": 1536, "seed": 21,
                 "rpm_hz": 30.0, "harmonics": [1.2, 0.7, 0.35, 0.15],
                 "noise_sigma": 0.06,
                 "resample": {"indices": [128, 700, 1200]},
                 "impulse": {"count": 4, "amplitude": 3.5}},
                {"id": "B-seg1", "length": 1024, "seed": 22,
                 "rpm_hz": 29.0, "harmonics": [1.1, 0.5],
                 "noise_sigma": 0.05,
                 "missing": {"count": 10}},
            ],
        },
        "DEV-C": {
            "seed": 3003,
            "segments": [
                {"id": "C-seg0", "length": 896, "seed": 31,
                 "rpm_hz": 22.0, "harmonics": [0.8, 0.3],
                 "noise_sigma": 0.05,
                 "missing": {"count": 5},
                 "impulse": {"count": 2, "amplitude": 2.5}},
                {"id": "C-seg1", "length": 896, "seed": 32,
                 "rpm_hz": 23.0, "harmonics": [0.85, 0.32, 0.12],
                 "noise_sigma": 0.07,
                 "fault": {"amplitude": 2.0}},
            ],
        },
        "DEV-BAD-FINITE": {
            "seed": 4004,
            "segments": [
                {"id": "BAD-seg0", "length": 512, "seed": 41,
                 "rpm_hz": 20.0, "force_nonfinite": True},
            ],
        },
        "DEV-SHORT": {
            "seed": 5005,
            "segments": [
                {"id": "SHORT-seg0", "length": 128, "seed": 51,
                 "rpm_hz": 20.0},
            ],
        },
    }
    return {"version": 1, "devices": devices}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    args = parser.parse_args()
    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    with (workspace / "config.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(DEFAULT_CONFIG, fh, sort_keys=True, allow_unicode=True)
    with (workspace / "descriptions.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(build_descriptions(), fh, sort_keys=True,
                       allow_unicode=True)
    print(f"sample workspace written to {workspace}")


if __name__ == "__main__":
    main()
