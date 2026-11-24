#!/usr/bin/env python3
"""Repeatable demo entry point.

Examples:
  python demo.py --sample samples/demo --workdir work/demo --gui
  python demo.py --sample samples/demo --workdir work/crash --crash-after-commits 2
  python demo.py --sample samples/demo --workdir work/crash
  python demo.py verify --baseline work/full --candidate work/crash
"""
from quota_ledger.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
