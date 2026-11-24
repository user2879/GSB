from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mmledger.demo import generate_workspace


def generated_workspace(root: Path) -> Path:
    generate_workspace(root, use_subprocess_workers=False, clean_outputs=True)
    return root


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
