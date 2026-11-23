#!/usr/bin/env python3
"""Generate the deterministic example tree (declaration + samples + plans)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kvdiag.samples import write_example_tree  # noqa: E402


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "examples"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    decl = write_example_tree(out, seed=seed)
    print("wrote %s with %d shards (seed=%d)" % (out, len(decl["shards"]), seed))


if __name__ == "__main__":
    main()
