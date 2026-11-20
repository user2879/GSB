from __future__ import annotations

import argparse

from isolation_lab.demo import run_demo


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the deterministic four-scenario demo")
    parser.add_argument("--workdir", default="runtime/demo")
    args = parser.parse_args()
    print(run_demo(args.workdir), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
