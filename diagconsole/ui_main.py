import argparse

from .ui import HeadlessError, launch


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Tkinter diagnostics console")
    parser.add_argument("--samples", default="samples")
    parser.add_argument("--data", default="data")
    parser.add_argument("--local", action="store_true")
    args = parser.parse_args(argv)
    try:
        launch(args.samples, args.data, use_services=not args.local)
    except HeadlessError as exc:
        print(str(exc))
        return 3
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
