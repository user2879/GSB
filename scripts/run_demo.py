"""End-to-end deterministic demo: generate sample, run concurrent + serial."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ab_workbench.cli import main as cli_main
from ab_workbench.generator import generate

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SAMPLE = os.path.join(ROOT, "data", "sample")


def main():
    paths = generate(SAMPLE)
    common = [
        "--log", paths["log"],
        "--config", paths["config"],
        "--attributes", paths["attributes"],
        "--injections", paths["injections"],
        "--snapshot-dir", os.path.join(ROOT, "snapshots", "demo"),
        "--cache-dir", os.path.join(ROOT, "tmp", "demo-cache"),
        "--tmp-dir", os.path.join(ROOT, "tmp", "demo-tmp"),
        "--report", os.path.join(ROOT, "tmp", "report-concurrent.json"),
    ]
    rc = cli_main(["run", *common])
    common[-1] = os.path.join(ROOT, "tmp", "report-serial.json")
    rc_serial = cli_main(["serial", *common])
    print("concurrent rc =", rc, " serial rc =", rc_serial)


if __name__ == "__main__":
    main()
