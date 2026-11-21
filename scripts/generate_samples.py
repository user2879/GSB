#!/usr/bin/env python3
"""Convenience wrapper around ``python -m remote_seg generate``."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from remote_seg.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["generate", *sys.argv[1:]]))
