from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from image_edit_workbench.samples import generate_sample_workspace  # noqa: E402


if __name__ == "__main__":
    result = generate_sample_workspace(ROOT)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))

