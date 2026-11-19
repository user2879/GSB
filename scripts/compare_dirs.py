"""Compare two generated directories byte-for-byte."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path


def tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: compare_dirs.py LEFT RIGHT", file=sys.stderr)
        return 2
    left = tree_hashes(Path(sys.argv[1]))
    right = tree_hashes(Path(sys.argv[2]))
    if left != right:
        all_names = sorted(set(left) | set(right))
        for name in all_names:
            if left.get(name) != right.get(name):
                print(f"DIFF {name}: {left.get(name)} != {right.get(name)}")
        return 1
    print(f"IDENTICAL {len(left)} files")
    for name, digest in left.items():
        print(f"{digest}  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

