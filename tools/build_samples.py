"""Regenerate the checked-in deterministic sample JSON."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from exp_tracker.samples import legacy_library_document  # noqa: E402


def main() -> None:
    target = ROOT / "samples" / "legacy_experiment.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            legacy_library_document(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(target)


if __name__ == "__main__":
    main()

