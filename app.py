"""Convenience entry point for ``python app.py`` on Windows."""

import sys

from mlpipe_workbench.cli import main

raise SystemExit(main(["ui", *sys.argv[1:]]))
