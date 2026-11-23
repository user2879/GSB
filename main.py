#!/usr/bin/env python3
"""Convenience entry point; equivalent to ``python -m rerank_lab``."""

from rerank_lab.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
