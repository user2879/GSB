"""Windows-friendly wrapper: ``python run_demo.py``."""

import sys

from glyphforge.__main__ import main


if __name__ == "__main__":
    raise SystemExit(main(["demo", *sys.argv[1:]]))
