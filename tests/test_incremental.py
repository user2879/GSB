from __future__ import annotations

import json
from dataclasses import replace

from dtw_diag.config import load_config
from dtw_diag.engine import (
    DiagnosticEngine,
    canonical_equivalence,
    full_rebuild,
    run_plan_reloading_state,
)
from dtw_diag.storage import load_edit_plan, load_library

from conftest import SAMPLE_ROOT, make_config, make_library, make_record


def _sample_inputs():
    config = load_config(SAMPLE_ROOT / "samples" / "config.yaml").config
    assert config is not None
    library = load_library(SAMPLE_ROOT / "samples" / "sequences.jsonl", config).library
    batches = load_edit_plan(SAMPLE_ROOT / "samples" / "edits.json").batches
    return config, library, batches


def _small_case():
    config = make_config(
        reference_sequence_id="R",
        bandwidth_permille=350,
        cell_recompute_budget=12,
        thread_pool_workers=2,
        snapshot_interval_batches=2,
    )
    library = make_library(
        [
            make_record("R", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], segment_ids=[0] * 7),
            make_record("Q1", [0.0, 0.12, 0.22, 0.31, 0.43, 0.52], segment_ids=[0] * 3 + [1] * 3),
            make_record("Q2", [0.05, 0.17, 0.27, 0.39, 0.49], segment_ids=[0] * 5),
            make_record("Q3", [0.03, 0.11, 0.24, 0.35], segment_ids=[0] * 4),
        ]
    )
    return config, library


def test_sample_incremental_equals_full_rebuild_from_persisted_state(tmp_path):
    config, library, batches = _sample_inputs()
    engine = DiagnosticEngine(
        config, library.copy(), [], workdir=tmp_path / "state", reset=True
    )
    engine, reports = run_plan_reloading_state(
        engine, batches, workdir=tmp_path / "state", auto_drain=True
    )
    incremental = engine.export_result()
    rebuilt = full_rebuild(engine.config, engine.library, engine.validation_failures)
    assert canonical_equivalence(incremental, rebuilt)["equal"]
    assert json.dumps(incremental, sort_keys=True) == json.dumps(rebuilt, sort_keys=True)
    assert any(
        report["budget"]["pending_sequence_count"] > 0 for report in reports
    )
    assert all(
        report["budget"]["consumed"] <= config.cell_recompute_budget
        for report in reports
    )
    assert engine.export_result()["pending_sequences"] == []
    assert (tmp_path / "state" / "snapshots" / "latest.json").exists()


def test_point_edit_invalidates_only_dependent_cells_and_downstream():
    config, library = _small_case()
    config = replace(config, cell_recompute_budget=10_000)
    engine = DiagnosticEngine(config, library, [], workdir="/tmp/dtwdiag_inval", reset=True)
    engine.run_batch(1, ())
    initial = engine.export_result()
    assert all(item["committed"] for item in engine.sequence_stats())

    engine.run_batch(
        2,
        (
            {
                "type": "point_update",
                "sequence_id": "Q1",
                "index": 2,
                "value": 0.99,
            },
        ),
    )
    batch = engine.batch_ledger[-1]
    stats = {item["sequence_id"]: item for item in batch["sequences"]}
    assert stats["Q1"]["target_changed"] is True
    assert stats["Q2"]["status"] == "RESULT_REUSED"
    assert stats["R"]["status"] == "RESULT_REUSED"
    assert stats["Q1"]["invalidated_cells"] > 0
    assert stats["Q1"]["reused_cells"] > 0

    plan_entry = next(
        item
        for item in reversed(engine.invalidation_ledger)
        if item["sequence_id"] == "Q1"
    )
    direct = [
        item
        for item in plan_entry["invalid_reasons"]
        if item["reason"] == "LOCAL_POINT_CHANGED"
    ]
    assert direct
    assert all(item["row"] == 2 or item["col"] == 2 for item in direct)
    assert any(
        item["reason"] == "DOWNSTREAM_OF_INVALID"
        for item in plan_entry["invalid_reasons"]
    )
    assert len(plan_entry["reused_cells"]) == plan_entry["reused_cell_count"]
    initial_q1 = next(
        item for item in initial["sequences"] if item["sequence_id"] == "Q1"
    )
    assert stats["Q1"]["recomputed_cells"] < len(initial_q1["cells"])


def test_budget_pending_uses_next_batch_priority_and_eventual_equivalence(tmp_path):
    config, library = _small_case()
    engine = DiagnosticEngine(
        config, library.copy(), [], workdir=tmp_path / "state", reset=True
    )
    first = engine.run_batch(1, ())
    assert first["pending_queue"]
    assert all(
        item["code"] == "E_CELL_BUDGET_PENDING" for item in first["failures"]
    )
    second = engine.run_batch(2, ())
    assert second["priority_sequences"] == first["pending_queue"]
    engine.run_plan([], auto_drain=True)
    incremental = engine.export_result()
    rebuilt = full_rebuild(engine.config, engine.library)
    assert canonical_equivalence(incremental, rebuilt)["equal"]
    assert max(item["consumed"] for item in engine.budget_ledger) <= config.cell_recompute_budget


def test_serial_and_parallel_results_and_ledgers_match(tmp_path):
    config, library = _small_case()
    batches = []
    from dtw_diag.models import EditBatch

    batches = [
        EditBatch(
            1,
            (
                {"type": "bandwidth_update", "bandwidth_permille": 500},
                {
                    "type": "point_update",
                    "sequence_id": "Q2",
                    "index": 1,
                    "value": 0.28,
                },
            ),
        ),
        EditBatch(2, ({"type": "step_mode_update", "step_mode": "asymmetric"},)),
    ]
    outputs = []
    for workers in (1, 4):
        workdir = tmp_path / f"w{workers}"
        engine = DiagnosticEngine(
            replace(config, thread_pool_workers=workers),
            library.copy(),
            [],
            workdir=workdir,
            reset=True,
        )
        engine.run_plan(batches, auto_drain=True)
        outputs.append(engine)
    assert canonical_equivalence(
        outputs[0].export_result(), outputs[1].export_result()
    )["equal"]
    assert outputs[0].budget_ledger == outputs[1].budget_ledger
    assert outputs[0].invalidation_ledger == outputs[1].invalidation_ledger
    assert outputs[0].reuse_ledger == outputs[1].reuse_ledger
