from __future__ import annotations

from dtw_diag.engine import DiagnosticEngine
from dtw_diag.errors import E_BANDWIDTH, E_INDEX_RANGE, E_STEP_MODE

from conftest import make_config, make_library, make_record


def test_invalid_edits_are_reported_and_do_not_change_state(tmp_path):
    config = make_config(reference_sequence_id="R")
    library = make_library(
        [make_record("R", [0.0, 0.1]), make_record("Q", [0.1, 0.2])]
    )
    engine = DiagnosticEngine(config, library, [], workdir=tmp_path, reset=True)
    failures = []
    failures += engine.apply_edit(
        {"type": "bandwidth_update", "bandwidth_permille": 1001}, batch=1, edit_index=0
    )
    failures += engine.apply_edit(
        {"type": "step_mode_update", "step_mode": "bad"}, batch=1, edit_index=1
    )
    failures += engine.apply_edit(
        {
            "type": "point_update",
            "sequence_id": "Q",
            "index": 9,
            "value": 0.4,
        },
        batch=1,
        edit_index=2,
    )
    assert [item.code for item in failures] == [E_BANDWIDTH, E_STEP_MODE, E_INDEX_RANGE]
    assert engine.config.bandwidth_permille == config.bandwidth_permille
    assert engine.config.step_mode == config.step_mode
    assert engine.library.records["Q"].observations == (0.1, 0.2)
