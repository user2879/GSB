"""Standalone environment readiness check for the local Windows workbench."""

from autoscale_diag.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["doctor"]))

