"""Validate the checked-in YAML sample without rendering."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from renderlab.config import load_workbench  # noqa: E402


def main() -> int:
    result = load_workbench(ROOT / "configs" / "workbench.yaml")
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    if result.can_render:
        assert result.config is not None
        assert len(result.noise_table) == result.config.render.denoise_steps
        assert len(result.guidance_table) == result.config.render.denoise_steps
        print("VALIDATION_OK")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
