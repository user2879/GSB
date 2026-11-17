#!/usr/bin/env python3
"""Over-quota queueing scenario wrapper."""

import sys

from run_demo import main

raise SystemExit(main(["--scenario", "queue", *sys.argv[1:]]))
