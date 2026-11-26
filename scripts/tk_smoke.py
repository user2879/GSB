import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rope_lab.gui import smoke_gui  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(smoke_gui(".", 100))
