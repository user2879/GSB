"""Generate the deterministic local samples."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opexlab.samplegen import build_sample_files


def main() -> None:
    paths = build_sample_files(ROOT)
    for name, path in sorted(paths.items()):
        print(f"{name}: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
