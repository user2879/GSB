from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any, Callable

from intentlab.pipeline import PipelinePaths, run_pipeline


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "input"


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def prepare_project(
    tmp: str | Path,
    *,
    include_edits: bool = False,
    mutate_config: Callable[[dict[str, Any]], None] | None = None,
    mutate_queries: Callable[[dict[str, Any]], None] | None = None,
    mutate_rules: Callable[[dict[str, Any]], None] | None = None,
) -> PipelinePaths:
    root = Path(tmp)
    input_dir = root / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "vocabulary.json", "queries.json", "rules.json"):
        shutil.copy2(INPUT / name, input_dir / name)
    if include_edits:
        shutil.copy2(INPUT / "edits.json", input_dir / "edits.json")
    if mutate_config is not None:
        value = read_json(input_dir / "config.json")
        mutate_config(value)
        write_json(input_dir / "config.json", value)
    if mutate_queries is not None:
        value = read_json(input_dir / "queries.json")
        mutate_queries(value)
        write_json(input_dir / "queries.json", value)
    if mutate_rules is not None:
        value = read_json(input_dir / "rules.json")
        mutate_rules(value)
        write_json(input_dir / "rules.json", value)
    return PipelinePaths(
        config_path=input_dir / "config.json",
        vocabulary_path=input_dir / "vocabulary.json",
        queries_path=input_dir / "queries.json",
        rules_path=input_dir / "rules.json",
        edits_path=input_dir / "edits.json" if include_edits else None,
        snapshot_root=root / "snapshots",
        report_root=root / "reports",
    )


def run(paths: PipelinePaths):
    return run_pipeline(paths)


def conclusion_signature(report: dict[str, Any], *, include_reuse: bool = False) -> dict[str, Any]:
    queries = []
    for item in report.get("queries", []):
        entry = {
            "query_id": item["query_id"],
            "match": item["match"],
            "score": item["score"],
            "segment_id": item["segment_id"],
            "diagnostics": item["diagnostics"],
        }
        if include_reuse:
            entry["match_reused"] = item["match_reused"]
            entry["score_reused"] = item["score_reused"]
        queries.append(copy.deepcopy(entry))
    return {
        "queries": queries,
        "segments": copy.deepcopy(report.get("segments", [])),
        "rules": copy.deepcopy(report.get("rules", {}).get("triggers", [])),
        "pending": copy.deepcopy(report.get("pending", [])),
        "failures": copy.deepcopy(report.get("failures", [])),
    }

