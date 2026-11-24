"""Convenience entry point equivalent to `python -m fppdiag.cli` (Windows friendly)."""
from fppdiag.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
