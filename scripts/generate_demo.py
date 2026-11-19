"""Convenience wrapper for deterministic demo data generation."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vad.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["generate-demo", *sys.argv[1:]]))
