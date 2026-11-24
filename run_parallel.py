"""Concurrent entry point used for serial/parallel consistency checks."""

from __future__ import annotations

import sys

from chart_workbench.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["run", *sys.argv[1:]]))
