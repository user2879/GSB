"""Convenience wrapper: ``python demo.py``."""

from speaker_diag.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["demo", "--root", "."]))

