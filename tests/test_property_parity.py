from __future__ import annotations

import copy
import json
import random
from dataclasses import replace
from pathlib import Path

import pytest

from ner_review.config import RuntimeConfig
from ner_review.engine import IncrementalEngine
from ner_review.labels import LABELS
from ner_review.synth import build_package, make_sample, write_demo_bundle
from ner_review.validation import load_package, parse_corpus


WORDS = [("LOC", "苏州"), ("LOC", "重庆"), ("PER", "屈原"), ("ORG", "岳麓书院"), ("TIME", "冬季")]


def make_engine(tmp_path: Path) -> IncrementalEngine:
    config = RuntimeConfig(seed=417, corpus_size=5)
    paths = write_demo_bundle(tmp_path / "data", config)
    package = load_package(paths["package"])["package"]
    engine = IncrementalEngine.from_bundle(paths["corpus"], paths["lexicon"], {"ok": True, "package": package}, config)
    engine.run_incremental()
    return engine


def assert_parity(engine: IncrementalEngine) -> None:
    fresh = IncrementalEngine(
        package=copy.deepcopy(engine.package),
        config=replace(engine.config),
        samples=copy.deepcopy(list(engine.samples.values())),
        lexicon=copy.deepcopy(engine.lexicon_list),
        corpus_failures=copy.deepcopy(engine.corpus_failures),
        lexicon_failures=copy.deepcopy(engine.lexicon_failures),
    )
    fresh.run_incremental()
    incremental_result = engine.authoritative_result()
    full_result = fresh.authoritative_result()
    assert incremental_result["cache_key_map"] == full_result["cache_key_map"]
    assert incremental_result["live_cache_keys"] == full_result["live_cache_keys"]
    assert incremental_result == full_result


@pytest.mark.parametrize("seed", [11, 29, 47])
def test_random_command_sequences_match_full_at_every_stable_point(tmp_path: Path, seed: int):
    engine = make_engine(tmp_path)
    rng = random.Random(seed)
    custom_entries: list[str] = []

    for step in range(9):
        op = rng.choice(
            [
                "upsert_lexicon",
                "update_lexicon",
                "remove_lexicon",
                "replace_sample",
                "delete_sample",
                "emission",
                "transition",
                "quant",
            ]
        )
        command: dict
        if op == "upsert_lexicon":
            entry_id = f"lex-random-{seed}-{step}"
            entity_type, text = rng.choice(WORDS)
            custom_entries.append(entry_id)
            command = {"op": "upsert_lexicon", "entry": {"id": entry_id, "text": text, "type": entity_type}}
        elif op == "update_lexicon":
            entry_id = rng.choice(list(engine.lexicon))
            entity_type, text = rng.choice(WORDS)
            command = {"op": "upsert_lexicon", "entry": {"id": entry_id, "text": text, "type": entity_type}}
        elif op == "remove_lexicon":
            removable = custom_entries or list(engine.lexicon)
            entry_id = rng.choice(removable)
            command = {"op": "remove_lexicon", "entry_id": entry_id}
            if entry_id in custom_entries:
                custom_entries.remove(entry_id)
        elif op == "replace_sample":
            sample_id = rng.choice(engine.sample_ids)
            row = make_sample(9000 + seed, rng.randint(1, 50))
            row["id"] = sample_id
            command = {"op": "replace_sample", "sample": row}
        elif op == "delete_sample":
            if len(engine.samples) <= 2:
                continue
            command = {"op": "delete_sample", "sample_id": rng.choice(engine.sample_ids)}
        elif op == "emission":
            command = {
                "op": "set_emission_weight",
                "label": rng.choice(LABELS),
                "feature_index": rng.randrange(engine.package["feature_dim"]),
                "bucket": rng.randrange(engine.package["bucket_count"]),
                "value": round(rng.uniform(-0.8, 0.8), 4),
            }
        elif op == "transition":
            command = {
                "op": "set_transition_weight",
                "from": rng.choice(LABELS),
                "to": rng.choice(LABELS),
                "value": round(rng.uniform(-2.0, 2.5), 4),
            }
        else:
            command = {
                "op": "set_quant",
                "decimal_bits": rng.choice([6, 8, 10, 12]),
                "quant_level": rng.choice([0, 1, 2]),
            }

        result = engine.apply_command(command)
        assert result["ok"], result
        engine.run_incremental()
        assert_parity(engine)


def test_blank_full_uses_same_package_constructor():
    config = RuntimeConfig(corpus_size=3)
    package = build_package(config)
    rows = [make_sample(config.seed, i) for i in range(1, 4)]
    samples = parse_corpus([json.dumps(row, ensure_ascii=False) for row in rows])["samples"]
    engine = IncrementalEngine(package=package, config=config, samples=samples)
    engine.run_incremental()
    assert engine.authoritative_result()["sample_count"] == 3
