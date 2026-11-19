#!/usr/bin/env python3
"""Convenience launcher: no arguments opens the Tkinter workbench."""

import sys

from vecflow.cli import main


argv = sys.argv[1:] or ["gui"]
raise SystemExit(main(argv))
