#!/usr/bin/env python3
"""Generate the deterministic local dataset (images + descriptions + labels)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from workbench.config import Config
from workbench.samplegen import build_dataset


def main():
    root = Path(__file__).resolve().parent.parent
    config = Config.load(root / "config.yaml")
    samples = build_dataset(config, root / "data")
    print("generated samples:", ", ".join(samples))


if __name__ == "__main__":
    main()
