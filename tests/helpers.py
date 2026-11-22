from __future__ import annotations

import copy
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from feature_pipeline import mini_yaml
from feature_pipeline.config import load_pipeline_config
from feature_pipeline.pipeline import run_pipeline
from feature_pipeline.sample import prepare_sample

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_TMP = PROJECT_ROOT / ".tmp" / "tests"
TEST_TMP.mkdir(parents=True, exist_ok=True)


class TestWorkspace:
    def __init__(self) -> None:
        self._temporary = tempfile.TemporaryDirectory(prefix="workspace-", dir=TEST_TMP)
        self.root = Path(self._temporary.name)
        (self.root / "configs").mkdir(parents=True, exist_ok=True)
        (self.root / "data").mkdir(parents=True, exist_ok=True)
        (self.root / "reports").mkdir(parents=True, exist_ok=True)
        self.config: dict[str, Any] = copy.deepcopy(
            mini_yaml.load(PROJECT_ROOT / "configs" / "pipeline.yaml")
        )
        self.columns: dict[str, Any] = copy.deepcopy(
            mini_yaml.load(PROJECT_ROOT / "configs" / "columns.yaml")
        )
        self.config_path = self.root / "configs" / "pipeline.yaml"
        self.columns_path = self.root / "configs" / "columns.yaml"
        self.write_columns()
        self.write_config()

    def close(self) -> None:
        self._temporary.cleanup()

    def write_columns(self, columns: dict[str, Any] | None = None) -> None:
        if columns is not None:
            self.columns = copy.deepcopy(columns)
        self.columns_path.write_text(
            json.dumps(self.columns, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

    def write_config(self, config: dict[str, Any] | None = None) -> None:
        if config is not None:
            self.config = copy.deepcopy(config)
        self.config["paths"] = {
            "sample": "data/sample.json",
            "sample_manifest": "data/sample_manifest.json",
            "overrides": "data/overrides",
            "cache": self.config["paths"].get("cache", "cache"),
            "report": self.config["paths"].get("report", "reports/diagnostic_report.json"),
        }
        self.config_path.write_text(
            json.dumps(self.config, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

    def load_config(self):
        return load_pipeline_config(self.config_path, root=self.root)

    def prepare(self, *, force: bool = False) -> dict[str, Any]:
        return prepare_sample(self.load_config(), force=force)

    def run(self) -> dict[str, Any]:
        return run_pipeline(self.config_path, root=self.root)

    def clone(self, *, cache: str = "cache", report: str = "reports/diagnostic_report.json") -> "TestWorkspace":
        other = TestWorkspace()
        other.columns = copy.deepcopy(self.columns)
        other.write_columns()
        config = copy.deepcopy(self.config)
        config["paths"]["cache"] = cache
        config["paths"]["report"] = report
        config["paths"]["overrides"] = "data/overrides"
        other.write_config(config)
        source_overrides = self.root / "data" / "overrides"
        if source_overrides.exists():
            destination = other.root / "data" / "overrides"
            shutil.copytree(source_overrides, destination)
        return other

    def replace_override(self, name: str, values: list[Any]) -> Path:
        directory = self.root / "data" / "overrides"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{name}.json"
        path.write_text(
            json.dumps({"schema_version": 1, "column": name, "values": values}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return path


def report_for(workspace: TestWorkspace) -> dict[str, Any]:
    return json.loads((workspace.root / workspace.config["paths"]["report"]).read_text(encoding="utf-8"))


def ledger_is_sorted(report: dict[str, Any], key: str) -> bool:
    stage_rank = {
        "cross_encoding": 0,
        "standardization": 1,
        "dimensionality_reduction": 2,
        "metrics": 3,
    }
    entries = report["diagnostics"][key]
    keys = [(stage_rank[item["stage"]], item["sort_key"], item.get("action", "")) for item in entries]
    return keys == sorted(keys)
