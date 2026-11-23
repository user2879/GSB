from __future__ import annotations

from dtw_diag.align import (
    compute_cell,
    feasible_coordinates,
    finalize_alignment,
)
from dtw_diag.band import build_band
from dtw_diag.errors import E_PATH_LENGTH_REJECTED, W_PATH_TRUNCATED
from dtw_diag.quantization import quantize_sequence

from conftest import make_config, make_record


def _align(query_obs, ref_obs, config):
    query = make_record("Q", query_obs, segment_ids=[0] * len(query_obs))
    reference = make_record("R", ref_obs, segment_ids=[0] * len(ref_obs))
    qq = quantize_sequence(query, config)
    rq = quantize_sequence(reference, config)
    band = build_band(len(query_obs), len(ref_obs), config.bandwidth_permille)
    cells = {}
    for row, col in feasible_coordinates(band):
        cells[(row, col)] = compute_cell("Q", row, col, qq, rq, band, config, cells)
    return finalize_alignment("Q", qq, rq, band, cells, config)


def test_symmetric_path_cells_and_tie_priority():
    config = make_config(bandwidth_permille=1000)
    result = _align([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], config)
    assert result.status == "ok"
    assert result.path[0] == (0, 0)
    assert result.path[-1] == (2, 3)
    assert len(result.path_key) == 64
    assert len(result.segment_key) == 64
    assert result.result_key != result.path_key
    assert set(result.path).issubset({(c.row, c.col) for c in result.cells})
    assert all(cell.backpointer is not None for cell in result.cells if cell.cost is not None)
    interior = next(cell for cell in result.cells if (cell.row, cell.col) == (1, 1))
    assert interior.backpointer == "DIAG"
    assert len(interior.predecessors) == 3


def test_asymmetric_uses_two_predecessors_and_reaches_endpoint():
    config = make_config(bandwidth_permille=1000, step_mode="asymmetric")
    result = _align([0.0, 0.2, 0.4], [0.0, 0.1, 0.2, 0.3], config)
    assert result.status == "ok"
    assert result.path[-1] == (2, 3)
    interior = next(cell for cell in result.cells if (cell.row, cell.col) == (2, 2))
    assert len(interior.predecessors) == 2
    assert interior.backpointer == "DIAG"


def test_path_length_reject_and_truncate():
    reject = _align(
        [0.1, 0.2, 0.3],
        [0.1, 0.2, 0.3, 0.4],
        make_config(bandwidth_permille=1000, max_path_length=2),
    )
    assert reject.status == "rejected"
    assert reject.failure_code == E_PATH_LENGTH_REJECTED
    assert reject.path == ()

    truncate = _align(
        [0.1, 0.2, 0.3],
        [0.1, 0.2, 0.3, 0.4],
        make_config(
            bandwidth_permille=1000,
            max_path_length=2,
            path_length_policy="truncate",
        ),
    )
    assert truncate.status == "ok"
    assert truncate.truncated is True
    assert truncate.path_length == 2
    assert truncate.warnings[0]["code"] == W_PATH_TRUNCATED


def test_segments_sum_in_fixed_order():
    query = make_record("Q", [0.0, 0.1, 0.2, 0.3], segment_ids=[0, 0, 1, 1])
    reference = make_record("R", [0.0, 0.2, 0.4, 0.6], segment_ids=[0, 1, 1, 2])
    config = make_config(bandwidth_permille=1000)
    qq = quantize_sequence(query, config)
    rq = quantize_sequence(reference, config)
    band = build_band(4, 4, 1000)
    cells = {}
    for row, col in feasible_coordinates(band):
        cells[(row, col)] = compute_cell("Q", row, col, qq, rq, band, config, cells)
    result = finalize_alignment("Q", qq, rq, band, cells, config)
    assert sum(item["local_cost"] for item in result.segments) == result.total_cost
    assert sum(item["cell_count"] for item in result.segments) == result.path_length
    assert all(item["sum_order"] == "row_asc_col_asc" for item in result.segments)
