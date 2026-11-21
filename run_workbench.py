#!/usr/bin/env python3
"""Convenience launcher: python run_workbench.py <subcommand> ..."""
import sys

from render_workbench.cli import main

if __name__ == "__main__":
    sys.exit(main())
