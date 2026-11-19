"""Print the local Python and dependency versions used by the tool."""

from __future__ import annotations

import sys

import numpy
import yaml


def main() -> int:
    print(f"python={sys.version.split()[0]}")
    print(f"executable={sys.executable}")
    print(f"numpy={numpy.__version__}")
    print(f"pyyaml={yaml.__version__}")
    if sys.version_info < (3, 11):
        print("ERROR: Python 3.11 or newer is required", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

