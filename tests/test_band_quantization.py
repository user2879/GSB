from __future__ import annotations

from dtw_diag.band import RowBand, build_band
from dtw_diag.quantization import quantize_sequence

from conftest import make_config, make_record


def test_band_endpoints_and_radius():
    band = build_band(5, 7, 0)
    assert band.radius == 0
    assert band.contains(0, 0)
    assert band.contains(4, 6)
    assert all(not row.empty for row in band.rows)
    assert len(band.rows) == 5

    wide = build_band(5, 7, 1000)
    assert wide.radius == 7
    assert all(
        wide.contains(row, col)
        for row in range(5)
        for col in range(7)
    )


def test_empty_row_representation_is_explicit():
    row = RowBand(3, 4, 2, empty=True)
    assert not row.contains(3)
    assert row.to_dict()["empty"] is True


def test_quantization_repeatable_and_level_sensitive():
    record = make_record("Q", [0.12345, -1.23456, None, 2.71828])
    cfg4 = make_config(fixed_point={"fraction_bits": 8, "quantization_step": 4})
    cfg8 = make_config(fixed_point={"fraction_bits": 8, "quantization_step": 8})
    first = quantize_sequence(record, cfg4)
    second = quantize_sequence(record, cfg4)
    assert [point.to_dict() for point in first.points] == [
        point.to_dict() for point in second.points
    ]
    assert first.fingerprint == second.fingerprint
    other = quantize_sequence(record, cfg8)
    assert any(
        a.fixed_units != b.fixed_units or a.residual != b.residual
        for a, b in zip(first.points, other.points)
    )
    assert first.points[2].missing is True
    assert first.points[2].fixed_units is None
    assert first.points[2].residual is None
    assert first.points[0].residual.endswith("5")
