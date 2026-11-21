from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

from mlpipe_workbench.demo_data import build_dataset, build_pipeline
from mlpipe_workbench.loader import prepare_from_objects

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "sample.json"
PIPELINE = ROOT / "data" / "pipeline.json"


def base_objects() -> tuple[dict, dict]:
    return copy.deepcopy(build_dataset()), copy.deepcopy(build_pipeline())


def prepared_base():
    data, pipeline = base_objects()
    return prepare_from_objects(data, pipeline)


def write_json(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")
    return path


class TempDirectoryMixin:
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.tempdir = Path(self._temp.name)

    def tearDown(self) -> None:
        self._temp.cleanup()
