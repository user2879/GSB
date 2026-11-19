"""Convenience Windows-friendly demo entry point."""

from __future__ import annotations

import sys

from level_workbench.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["demo", *sys.argv[1:]]))
