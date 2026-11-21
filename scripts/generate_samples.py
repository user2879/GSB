#!/usr/bin/env python3
"""Convenience wrapper for deterministic sample generation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from workbench.generator import generate_dataset


if __name__ == "__main__":
    generate_dataset("samples", "config", seed=20261122, sample_count=8)
