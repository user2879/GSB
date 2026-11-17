#!/usr/bin/env python3
"""MTU fragmentation scenario wrapper."""

import sys

from run_demo import main

raise SystemExit(main(["--scenario", "fragment", *sys.argv[1:]]))
