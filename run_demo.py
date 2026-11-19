"""Convenience one-click entry: equivalent to `python -m syncpipe.cli demo`.

Run with the project virtualenv interpreter, e.g. on Windows:

    .venv\\Scripts\\python run_demo.py --workspace workspace --seed 20261119
"""

from syncpipe.cli import main

if __name__ == "__main__":
    raise SystemExit(main([
        "demo",
        "--workspace", "workspace",
        "--seed", "20261119",
    ]))
