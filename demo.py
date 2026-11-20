from __future__ import annotations

import sys

from ckptmeta.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["demo", *sys.argv[1:]]))
