import pytest

from drift_monitor.config import load_config
from drift_monitor.datagen import CUR, REF, expanded_columns, generate_pair
from drift_monitor.errors import ConfigInvalidError, DataGenError
from drift_monitor.windows import aligned_window_plan, slice_windows


def test_config_loads_defaults(config):
    assert config.window_count == 4
    assert len(config.tiers) == 3
    assert config.concurrency == 4
    assert config.active_tier == "fine"


def test_duplicate_column_rejected(tmp_path):
    yaml_text = """
data:
  seed: 1
  columns:
    - {name: x, kind: numeric, metric: mean_shift}
    - {name: x, kind: numeric, metric: hist_density}
  size_tiers:
    - {name: s, samples: 100}
algorithm: {window_count: 2}
"""
    path = tmp_path / "bad.yaml"
    path.write_text(yaml_text)
    with pytest.raises(ConfigInvalidError) as exc:
        load_config(path)
    assert exc.value.code == "E_CONFIG_INVALID"
    assert "name" in exc.value.field


def test_target_in_features_rejected(tmp_path):
    yaml_text = """
data:
  seed: 1
  target_column: x
  columns:
    - {name: x, kind: numeric, metric: mean_shift}
  size_tiers:
    - {name: s, samples: 100}
"""
    path = tmp_path / "bad.yaml"
    path.write_text(yaml_text)
    with pytest.raises(ConfigInvalidError) as exc:
        load_config(path)
    assert exc.value.field == "data.target_column"


def test_non_monotonic_size_tiers_rejected(tmp_path):
    yaml_text = """
data:
  seed: 1
  columns:
    - {name: x, kind: numeric, metric: mean_shift}
  size_tiers:
    - {name: a, samples: 200}
    - {name: b, samples: 100}
"""
    path = tmp_path / "bad.yaml"
    path.write_text(yaml_text)
    with pytest.raises(ConfigInvalidError) as exc:
        load_config(path)
    assert "samples" in exc.value.field


def test_bad_window_count_and_tiers(tmp_path):
    base = """
data:
  seed: 1
  columns:
    - {name: x, kind: numeric, metric: mean_shift}
  size_tiers:
    - {name: a, samples: 100}
"""
    bad_windows = tmp_path / "w.yaml"
    bad_windows.write_text(base + "algorithm: {window_count: 0}")
    with pytest.raises(ConfigInvalidError):
        load_config(bad_windows)

    bad_tiers = tmp_path / "t.yaml"
    bad_tiers.write_text(base + """
quantization:
  tiers:
    - {name: a, decimals: 4}
    - {name: b, decimals: 2}
""")
    with pytest.raises(ConfigInvalidError) as exc:
        load_config(bad_tiers)
    assert "decimals" in exc.value.field


def test_wrong_metric_for_column_kind(tmp_path):
    yaml_text = """
data:
  seed: 1
  columns:
    - {name: c, kind: categorical, metric: mean_shift}
  size_tiers:
    - {name: a, samples: 100}
"""
    path = tmp_path / "bad.yaml"
    path.write_text(yaml_text)
    with pytest.raises(ConfigInvalidError) as exc:
        load_config(path)
    assert "metric" in exc.value.field


def test_deterministic_generation_and_missing_markers(config):
    tier = config.data.size_tiers[0]
    ref_a, cur_a = generate_pair(config.data, tier)
    ref_b, cur_b = generate_pair(config.data, tier)
    assert [None if v is None else str(v) for v in ref_a.columns[0].values] == \
           [None if v is None else str(v) for v in ref_b.columns[0].values]
    assert ref_a.period == REF and cur_a.period == CUR
    total_missing = sum(c.missing_mask.sum() for c in ref_a.columns)
    assert total_missing > 0
    for column in ref_a.columns:
        assert all(column.values[i] is None for i in range(len(column.values))
                   if column.missing_mask[i])


def test_size_tier_scaling(config):
    small, medium = config.data.size_tiers[:2]
    small_cols = expanded_columns(config.data, small)
    medium_cols = expanded_columns(config.data, medium)
    assert len(medium_cols) > len(small_cols)
    ref, _ = generate_pair(config.data, medium)
    assert ref.n_samples == medium.samples


def test_window_short_tail_rules():
    regular = slice_windows(100, 4)
    assert [s.size for s in regular] == [25, 25, 25, 25]
    merged = slice_windows(16, 5)
    assert merged[-1].tag == "merged_short"
    assert len(merged) == 4
    standalone = slice_windows(3, 5)
    assert all(s.tag == "standalone_short" for s in standalone)
    ref_w, cur_w = aligned_window_plan(100, 100, 4)
    assert len(ref_w) == len(cur_w) == 4


def test_data_gen_failure_code(monkeypatch, config):
    from drift_monitor import datagen
    from drift_monitor.errors import DataGenError

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(datagen, "expanded_columns", boom)
    with pytest.raises(DataGenError) as exc:
        datagen.generate_pair(config.data, config.data.size_tiers[0])
    assert exc.value.code == "E_DATA_GEN_FAILED"
