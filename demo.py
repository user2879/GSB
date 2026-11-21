"""Convenience entry point for ``python demo.py`` on Windows."""

import sys

from mlpipe_workbench.cli import main

raise SystemExit(main(["demo", *sys.argv[1:]]))
