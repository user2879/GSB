from __future__ import annotations

import json
import shutil
from pathlib import Path

from arbitration_workbench.numeric import read_json, write_json


ROOT = Path(__file__).resolve().parents[1]


def copy_sample_workspace(target: Path) -> Path:
    target.mkdir(parents=True, exist_ok=True)
    sample_dir = target / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    for name in ("rules.json", "config.json", "evidence.jsonl", "exception_injections.json"):
        shutil.copy2(ROOT / "samples" / name, sample_dir / name)
    (target / "tmp").mkdir(parents=True, exist_ok=True)
    return target


def rewrite_config(root: Path, mutate) -> dict:
    path = root / "samples" / "config.json"
    raw = read_json(path)
    mutate(raw)
    write_json(path, raw, temp_dir=root / "tmp")
    return raw


def read_evidence_rows(root: Path) -> list[dict]:
    rows = []
    for line in (root / "samples" / "evidence.jsonl").read_text(encoding="utf-8").splitlines():
        rows.append(json.loads(line))
    return rows
