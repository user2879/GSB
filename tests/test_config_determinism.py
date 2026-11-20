from __future__ import annotations

from pathlib import Path

import pytest

from ckptmeta.config import load_config
from ckptmeta.errors import InteropError
from ckptmeta.ledger import build_ledger
from ckptmeta.samples import SAMPLE_CONFIG, generate_fixtures
from ckptmeta.view import view_bytes


def test_ledger_is_bit_deterministic():
    config = load_config(dict(SAMPLE_CONFIG))
    left = build_ledger(config)
    right = build_ledger(config)
    assert left.ledger_digest == right.ledger_digest
    for step in config.steps:
        assert view_bytes(left.entry(step).view) == view_bytes(right.entry(step).view)
        assert left.entry(step).view_digest == right.entry(step).view_digest


def test_fixture_generation_is_repeatable(tmp_path: Path):
    first = generate_fixtures(tmp_path / "first")
    second = generate_fixtures(tmp_path / "second")
    assert first.ledger.read_bytes() == second.ledger.read_bytes()
    assert first.direct_target.read_bytes() == second.direct_target.read_bytes()
    assert first.v1_target.read_bytes() == second.v1_target.read_bytes()
    assert first.v2_target.read_bytes() == second.v2_target.read_bytes()
    assert (first.content / "step-00000020" / "dataset_X.npy").read_bytes() == (
        second.content / "step-00000020" / "dataset_X.npy"
    ).read_bytes()


def test_partition_modes_change_data_cursor_and_view():
    modulo_config = load_config(dict(SAMPLE_CONFIG))
    contiguous_raw = dict(SAMPLE_CONFIG)
    contiguous_raw["partition_mode"] = "contiguous"
    contiguous_config = load_config(contiguous_raw)
    modulo = build_ledger(modulo_config)
    contiguous = build_ledger(contiguous_config)
    assert modulo.entry(20).view["partition_mode"] == "modulo"
    assert contiguous.entry(20).view["partition_mode"] == "contiguous"
    assert modulo.entry(20).view["shards"][0]["data_cursor"] != contiguous.entry(20).view["shards"][0]["data_cursor"]
    assert modulo.entry(20).view_digest != contiguous.entry(20).view_digest


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seed", True),
        ("shard_count", 0),
        ("partition_mode", "random"),
        ("sample_count", 2),
        ("input_dim", 0),
        ("hidden_dim", 5000),
        ("classes", 1),
        ("steps", [10, 20]),
        ("steps", [0, -1]),
    ],
)
def test_config_bounds_report_field(field: str, value):
    raw = dict(SAMPLE_CONFIG)
    raw[field] = value
    with pytest.raises(InteropError) as raised:
        load_config(raw)
    assert raised.value.code == "E_CONFIG"
    assert raised.value.path == field or raised.value.path.startswith("steps[")
    assert raised.value.details.get("field")
