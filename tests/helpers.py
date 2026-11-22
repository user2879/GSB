from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from feature_backfill.config_loader import load_feature_file, load_plan, parse_config
from feature_backfill.data_loader import load_samples
from feature_backfill.injections import load_injections
from feature_backfill.models import AppConfig, PlanBatch


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
INJECTIONS = ROOT / "injections"


def load_fixture(
    directory: Path | None = None,
    *,
    config_path: Path | None = None,
    plan_path: Path | None = None,
) -> dict[str, Any]:
    directory = directory or (DATA / "default")
    config = parse_config(config_path or (directory / "config.yaml"))
    features, raw_fields, feature_failures = load_feature_file(
        directory / "features.json"
    )
    assert not feature_failures, feature_failures
    rows, row_failures = load_samples(
        directory / "samples.jsonl",
        config=config,
        raw_fields=raw_fields,
    )
    plan, plan_failures = load_plan(plan_path or (directory / "plan.json"))
    assert not plan_failures, plan_failures
    return {
        "config": config,
        "features": features,
        "raw_fields": raw_fields,
        "rows": rows,
        "row_failures": row_failures,
        "plan": plan,
    }


def config_variant(**changes: Any) -> AppConfig:
    base = load_fixture()["config"]
    return replace(base, **changes)


def empty_plan() -> list[PlanBatch]:
    return load_plan(DATA / "scenarios" / "empty_plan.json")[0]


def load_injection_set(name: str):
    injections, failures = load_injections(INJECTIONS / f"{name}.json")
    assert not failures, failures
    return injections


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
