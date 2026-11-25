from __future__ import annotations

import random

import pytest

from cross_coding.config import load_bundle_from_docs
from cross_coding.pipeline import comparison_view, run_pipeline
from cross_coding.replay import apply_edit


COLUMNS = ("a", "b", "c", "d")
PAIRS = [("a", "b"), ("a", "c"), ("a", "d"), ("b", "c"), ("b", "d"), ("c", "d")]


def _bundle(feature_doc, policy_doc):
    loaded, report = load_bundle_from_docs(feature_doc, policy_doc)
    assert report.ok, report.errors
    return loaded


def _random_edit(rng: random.Random) -> dict:
    kind = rng.randrange(9)
    if kind == 0:
        return {"type": "add_pair", "pair": list(rng.choice(PAIRS))}
    if kind == 1:
        return {"type": "remove_pair", "pair": list(rng.choice(PAIRS))}
    if kind == 2:
        return {"type": "set_max_order", "value": rng.choice([1, 2, 3, 4])}
    if kind == 3:
        return {"type": "set_cardinality_budget", "value": rng.choice([0, 2, 5, 12, 30])}
    if kind == 4:
        return {"type": "set_point_budget", "value": rng.choice([0, 240, 960, 5000, 10_000_000])}
    if kind == 5:
        return {"type": "set_encoding_width", "value": rng.choice([4, 8, 16, 32])}
    if kind == 6:
        return {"type": "set_quantization_levels", "value": rng.choice([8, 16, 32, 64])}
    if kind == 7:
        return {"type": "set_fixed_point_bits", "value": rng.choice([8, 16, 24, 32])}
    return {"type": "replace_column", "column": rng.choice(COLUMNS), "replacement_seed": rng.randrange(1, 100_000)}


@pytest.mark.parametrize("sequence_seed", [101, 202, 303])
def test_multiple_edit_sequences_match_blank_rebuild(
    sequence_seed, feature_doc, policy_doc, tmp_path
) -> None:
    policy_doc["candidate_pairs"] = [["a", "b"], ["b", "c"], ["c", "d"]]
    policy_doc["max_order"] = 3
    incremental_bundle = _bundle(feature_doc, policy_doc)
    full_bundle = _bundle(feature_doc, policy_doc)
    rng = random.Random(sequence_seed)
    state = tmp_path / "incremental"

    for cursor in range(1, 9):
        edit = _random_edit(rng)
        incremental_bundle = apply_edit(incremental_bundle, edit)
        full_bundle = apply_edit(full_bundle, edit)
        incremental = run_pipeline(
            incremental_bundle, str(state), command_cursor=cursor, parallel=True
        )
        full = run_pipeline(
            full_bundle,
            str(tmp_path / f"blank-{sequence_seed}-{cursor}"),
            command_cursor=cursor,
            parallel=False,
        )
        assert comparison_view(incremental) == comparison_view(full), (sequence_seed, cursor, edit)
