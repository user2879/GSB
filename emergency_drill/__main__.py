"""Allow ``python -m emergency_drill`` to use the CLI."""

import sys

from .cli import main


if __name__ == "__main__":
    sys.exit(main())
