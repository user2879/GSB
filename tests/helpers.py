from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from lateness_ledger.spec import SessionBundle, load_bundle

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"


def sample_bundle(strategy: str) -> SessionBundle:
    return load_bundle(
        [SAMPLES / f"sources_{strategy}.json"],
        SAMPLES / "window_rules.json",
        SAMPLES / "derivation_budget.json",
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )


def mutate_sample(
    directory: Path,
    *,
    strategy: str = "source_max",
    source_mutator=None,
    rules_mutator=None,
    budget_mutator=None,
) -> tuple[list[Path], Path, Path]:
    source_doc = read_json(SAMPLES / f"sources_{strategy}.json")
    rules_doc = read_json(SAMPLES / "window_rules.json")
    budget_doc = read_json(SAMPLES / "derivation_budget.json")
    if source_mutator:
        source_mutator(source_doc)
    if rules_mutator:
        rules_mutator(rules_doc)
    if budget_mutator:
        budget_mutator(budget_doc)
    source_path = directory / "sources.json"
    rules_path = directory / "rules.json"
    budget_path = directory / "budgets.json"
    write_json(source_path, source_doc)
    write_json(rules_path, rules_doc)
    write_json(budget_path, budget_doc)
    return [source_path], rules_path, budget_path


def make_small_sources(directory: Path, *, event_count: int = 40) -> tuple[list[Path], Path, Path]:
    def mutator(doc: dict[str, Any]) -> None:
        for source in doc["sources"]:
            source["event_count"] = event_count

    return mutate_sample(directory, source_mutator=mutator)


def temp_dir() -> tempfile.TemporaryDirectory[str]:
    return tempfile.TemporaryDirectory(prefix="lateness-ledger-test-")
