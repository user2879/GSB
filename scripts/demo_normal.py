#!/usr/bin/env python3
"""Normal-loopback scenario wrapper."""

import sys

from run_demo import main

raise SystemExit(main(["--scenario", "normal", *sys.argv[1:]]))
