from __future__ import annotations

import copy
from pathlib import Path

from ner_editor.engine import LocalNerEngine
from ner_editor.loader import load_config, load_corpus, load_model
from ner_editor.models import CommandSpec

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"


def load_base(*, budget_ms: float = 60000.0, workers: int = 4, sample_limit: int | None = None):
    samples, corpus_failures = load_corpus(SAMPLES / "corpus.jsonl")
    model, model_failures = load_model(SAMPLES / "model.json")
    config, config_failures = load_config(SAMPLES / "config.json", base_dir=ROOT)
    assert not model_failures and not config_failures
    selected = dict(sorted(samples.items())[:sample_limit] if sample_limit else samples)
    config = config.clone()
    config.latency_budget_ms = budget_ms
    config.thread_pool_workers = workers
    config.snapshot_commit_interval_commands = 999
    return selected, model, config, corpus_failures


def make_engine(*, budget_ms: float = 60000.0, workers: int = 4, sample_limit: int | None = None):
    samples, model, config, failures = load_base(
        budget_ms=budget_ms,
        workers=workers,
        sample_limit=sample_limit,
    )
    return LocalNerEngine(samples, model, config, load_failures=failures)


def command(position: int, command_type: str, **payload):
    return CommandSpec(
        position=position,
        command_type=command_type,
        payload={"type": command_type, **payload},
        sequence=position,
    )


def ledger_statuses(engine: LocalNerEngine) -> dict[str, dict[str, str]]:
    ledger = engine.build_ledger()
    result = {}
    for sample_id, value in ledger["samples"].items():
        result[sample_id] = {
            stage["stage"]: stage["status"] for stage in value["stages"]
        }
    return result
