import sys

from imv.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["gui", *sys.argv[1:]]))
