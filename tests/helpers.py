from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dbml.database import write_json_atomic
from dbml.manifest import load_manifest
from dbml.migrations import MigrationRunner
from dbml.sample import build_sample_database


MANIFEST_PATH = ROOT / "data" / "migrations" / "orders_migration.json"


def sample_payload() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def build_base(directory: Path) -> Path:
    base = directory / "sample_base.sqlite3"
    build_sample_database(base, seed=20261118, rows=48, overwrite=True)
    return base


def working_copy(base: Path, directory: Path, name: str = "orders.sqlite3") -> Path:
    target = directory / name
    shutil.copy2(base, target)
    return target


def write_manifest(directory: Path, payload: dict, name: str = "manifest.json") -> Path:
    path = directory / name
    write_json_atomic(path, payload)
    return path


def migrate_sample(
    directory: Path,
    *,
    batch_size: int = 7,
):
    base = build_base(directory)
    working = working_copy(base, directory)
    manifest = load_manifest(MANIFEST_PATH, working)
    runner = MigrationRunner(
        working,
        manifest,
        ledger_dir=directory / "ledger",
        batch_size=batch_size,
    )
    result = runner.run()
    return base, working, runner, result

