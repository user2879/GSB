import json

from conftest import ROOT
from qcalib.hashing import stable_json

from tests._helpers import execute


def test_single_layer_replacement_only_invalidates_that_layer(tmp_path):
    baseline = execute(tmp_path / "base", None)
    replace_stream = tmp_path / "replace.json"
    replace_stream.write_text(json.dumps({"commands": [
        {"op": "replace_weights", "layer": "retrieval_head",
         "source": str(ROOT / "samples" / "retrieval_head_replacement.json")}
    ]}), encoding="utf-8")
    updated = execute(tmp_path / "updated", replace_stream,
                      previous_cache=baseline["cache"])

    changed = [item for item in updated["reuse_ledger"]
               if not item["hit"] and item["layer"] == "retrieval_head"]
    untouched = [item for item in updated["reuse_ledger"]
                 if item["layer"] == "embedding_proj"]
    assert changed and all(
        item["reason"] == "layer_or_group_content_or_params_changed"
        or item["reason"].startswith("layer_or_group")
        for item in changed if item["stage"] == "quantile_calibration")
    assert untouched and all(item["hit"] for item in untouched)

    # Unchanged layer artifacts must be byte-for-byte identical.
    assert stable_json(updated["layers"]["embedding_proj"]) == stable_json(
        baseline["layers"]["embedding_proj"])
    assert stable_json(updated["layers"]["retrieval_head"]) != stable_json(
        baseline["layers"]["retrieval_head"])


def test_tier_switch_only_invalidates_distillation_of_affected_layer(tmp_path):
    baseline = execute(tmp_path / "base", None)
    tier_stream = tmp_path / "tier.json"
    tier_stream.write_text(json.dumps({"commands": [
        {"op": "set_tier", "layer": "embedding_proj", "tier": "large"}
    ]}), encoding="utf-8")
    updated = execute(tmp_path / "updated", tier_stream,
                      previous_cache=baseline["cache"])
    distill_misses = [
        item for item in updated["reuse_ledger"]
        if not item["hit"]
        and item["layer"] == "embedding_proj"
        and item["stage"] == "distillation_correction"]
    assert len(distill_misses) == 1
    group_hits = [
        item for item in updated["reuse_ledger"]
        if item["layer"] == "embedding_proj"
        and item["stage"] == "fixed_point_requantization"]
    assert all(item["hit"] for item in group_hits)
    assert updated["layers"]["embedding_proj"]["sample_refs"] != (
        baseline["layers"]["embedding_proj"]["sample_refs"])


def test_quant_level_change_invalidates_requantization_but_keeps_calibration(tmp_path):
    baseline = execute(tmp_path / "base", None)
    level_stream = tmp_path / "level.json"
    level_stream.write_text(json.dumps({"commands": [
        {"op": "set_layer", "layer": "score_mixer",
         "values": {"bits": 16, "level": "int16", "frac_bits": 10}}
    ]}), encoding="utf-8")
    updated = execute(tmp_path / "updated", level_stream,
                      previous_cache=baseline["cache"])
    cal_hits = [item for item in updated["reuse_ledger"]
                if item["layer"] == "score_mixer"
                and item["stage"] == "quantile_calibration"]
    qnt_misses = [item for item in updated["reuse_ledger"]
                  if item["layer"] == "score_mixer"
                  and item["stage"] == "fixed_point_requantization"]
    assert all(item["hit"] for item in cal_hits)
    assert all(not item["hit"] for item in qnt_misses)
