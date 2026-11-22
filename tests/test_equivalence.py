"""Requirement 5: incremental replay == full rebuild, field by field.

Each test runs a growing edit sequence against a persistent snapshot
(incremental mode) and compares the projected comparison view against a
fresh-from-empty full replay over the identical final inputs.
"""

import copy
import json

import yaml

from dialog_state_pipeline.config import load_config, load_fsm
from dialog_state_pipeline.injection import load_injection
from dialog_state_pipeline.pipeline import Pipeline
from dialog_state_pipeline.report import comparison_view


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _read_jsonl(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _run(events_path, fsm_path, config_path, snapshot_dir,
         injection_path=None, mode="incremental"):
    config = load_config(config_path)
    config.snapshot_dir = str(snapshot_dir)
    fsm = load_fsm(fsm_path)
    injections = load_injection(injection_path)
    pipeline = Pipeline(config, fsm, injections)
    return pipeline.run(events_path, report_path=None, mode=mode)


def _full_replay(events_path, fsm_path, config_path, workdir, injection_path=None):
    snapshot = workdir / "fresh_full"
    if snapshot.exists():
        import shutil
        shutil.rmtree(snapshot)
    return _run(events_path, fsm_path, config_path, snapshot,
                injection_path, mode="full")


def test_append_replace_delete_sequence(sample_dir, tmp_path):
    base_rows = _read_jsonl(sample_dir / "events_base.jsonl")
    edited_rows = _read_jsonl(sample_dir / "events.jsonl")

    # Phase A: seed the incremental snapshot from the base stream.
    base_path = tmp_path / "stream.jsonl"
    _write_jsonl(base_path, base_rows)
    inc_snapshot = tmp_path / "inc_snapshot"
    report_a = _run(base_path, sample_dir / "fsm.yaml",
                    sample_dir / "pipeline_config.json", inc_snapshot)

    # Phase B: apply the prepared append/replace/delete edit batch in place.
    _write_jsonl(base_path, edited_rows)
    report_b = _run(base_path, sample_dir / "fsm.yaml",
                    sample_dir / "pipeline_config.json", inc_snapshot)

    # Reuse must happen: edited turns sit at turn 4+, earlier turns cached.
    for sid, session in report_b["sessions"].items():
        if sid != "S001":  # S001 also has a turn-2 delete
            assert 3 in session["reused_turns"]
            assert session["replayed_turns"], f"{sid} should replay edited tail"

    full = _full_replay(base_path, sample_dir / "fsm.yaml",
                        sample_dir / "pipeline_config.json", tmp_path)
    assert comparison_view(report_b) == comparison_view(full)

    # And the seeded baseline itself matches a full run over the base stream.
    base_full = _full_replay(tmp_path / "base_only.jsonl"
                             if _write_base(tmp_path, base_rows) else base_path,
                             sample_dir / "fsm.yaml",
                             sample_dir / "pipeline_config.json", tmp_path)
    assert comparison_view(report_a) == comparison_view(base_full)


def _write_base(tmp_path, rows):
    _write_jsonl(tmp_path / "base_only.jsonl", rows)
    return True


def test_slot_schema_change_and_strategy_switch(sample_dir, tmp_path):
    edited_rows = _read_jsonl(sample_dir / "events.jsonl")
    stream_path = tmp_path / "stream.jsonl"
    _write_jsonl(stream_path, edited_rows)
    inc_snapshot = tmp_path / "inc_snapshot"

    fsm_path = tmp_path / "fsm.yaml"
    with open(sample_dir / "fsm.yaml", "r", encoding="utf-8") as handle:
        fsm_doc = yaml.safe_load(handle)
    with open(fsm_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(fsm_doc, handle, sort_keys=True)

    config_path = sample_dir / "pipeline_config.json"
    # First run establishes the cache.
    _run(stream_path, fsm_path, config_path, inc_snapshot)

    # Edit 1: switch a merge strategy (overwrite -> reject).
    edited_fsm = copy.deepcopy(fsm_doc)
    edited_fsm["slots"]["city"]["strategy"] = "reject"
    strategy_fsm_path = _persist(tmp_path, edited_fsm, "fsm_strategy.yaml")
    with open(fsm_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(edited_fsm, handle, sort_keys=True)
    report_after_strategy = _run(stream_path, fsm_path, config_path, inc_snapshot)

    # Edit 2: change a slot type constraint (schema change).
    edited_fsm["slot_types"]["score"]["max"] = 50
    schema_fsm_doc = copy.deepcopy(edited_fsm)
    with open(fsm_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(edited_fsm, handle, sort_keys=True)
    report_after_schema = _run(stream_path, fsm_path, config_path, inc_snapshot)

    full = _full_replay(stream_path, fsm_path, config_path, tmp_path)
    strategy_full = _full_replay(stream_path, strategy_fsm_path, config_path,
                                 tmp_path / "cmp1")
    assert comparison_view(report_after_strategy) == comparison_view(strategy_full)
    assert comparison_view(report_after_schema) == comparison_view(full)


def _persist(tmp_path, doc, name):
    path = tmp_path / name
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(doc, handle, sort_keys=True)
    return path
