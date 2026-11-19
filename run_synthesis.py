"""Convenience entry point: ``python run_synthesis.py <manifest.json> [--serial]``."""

from haptic_engine.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
