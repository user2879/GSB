"""Allow ``python -m vad`` as a CLI alias."""

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
