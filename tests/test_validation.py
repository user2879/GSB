from __future__ import annotations

import json

import pytest

from cqdiag.config import validate_config
from cqdiag.errors import (
    CqError,
    E_CONFIG_INVALID,
    E_GROUP_MISMATCH,
    E_QUANT_WIDTH_INVALID,
    E_SAMPLE_INVALID,
)
from cqdiag.samples import load_samples

from conftest import make_config, make_config_raw


def test_valid_config_ok():
    cfg = make_config()
    assert cfg.quant_bits == 8 and cfg.frac_bits == 8


@pytest.mark.parametrize("mutate", [
    lambda r: r["tensors"][0].update(shape=[0, 4]),
    lambda r: r["tensors"][0].update(column_axis=5),
    lambda r: r.update(quantile_lo_permille=500, quantile_hi_permille=500),
    lambda r: r.update(quantile_lo_permille=-1),
    lambda r: r.update(quantile_hi_permille=1001),
    lambda r: r.update(frac_bits=0),
    lambda r: r.update(abs_tol=0),
    lambda r: r.update(rel_tol=float("inf")),
    lambda r: r.update(scheme="weird"),
    lambda r: r.update(saturation="explode"),
])
def test_config_invalid(mutate):
    raw = make_config_raw()
    mutate(raw)
    with pytest.raises(CqError) as ei:
        validate_config(raw)
    assert ei.value.code == E_CONFIG_INVALID


@pytest.mark.parametrize("mutate", [
    lambda r: r.update(quant_bits=3),
    lambda r: r.update(quant_bits_candidates=[8, 32]),
    lambda r: r.update(quant_bits=4, quant_bits_candidates=[8]),
])
def test_quant_width_invalid(mutate):
    raw = make_config_raw()
    mutate(raw)
    with pytest.raises(CqError) as ei:
        validate_config(raw)
    assert ei.value.code == E_QUANT_WIDTH_INVALID


@pytest.mark.parametrize("labels", [[0, 0, 1], [0, 0, 1, 1, 2], [0, 0, 1, -1]])
def test_group_mismatch(labels):
    raw = make_config_raw(groups={"t": labels})
    with pytest.raises(CqError) as ei:
        validate_config(raw)
    assert ei.value.code == E_GROUP_MISMATCH


def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def _full_record(shape=(6, 4), name="t"):
    elements = []
    for i in range(shape[0]):
        for j in range(shape[1]):
            elements.append({"index": [i, j], "value": float(i + j)})
    return {"tensor": name, "shape": list(shape), "elements": elements}


def test_samples_ok(tmp_path):
    cfg = make_config()
    p = tmp_path / "s.jsonl"
    _write_jsonl(p, [_full_record()])
    ss = load_samples(str(p), cfg)
    assert ss.tensor("t").values.shape == (6, 4)


def test_samples_shape_mismatch(tmp_path):
    cfg = make_config()
    p = tmp_path / "s.jsonl"
    rec = _full_record()
    rec["shape"] = [6, 5]
    _write_jsonl(p, [rec])
    with pytest.raises(CqError) as ei:
        load_samples(str(p), cfg)
    assert ei.value.code == E_SAMPLE_INVALID


def test_samples_missing_elements(tmp_path):
    cfg = make_config()
    p = tmp_path / "s.jsonl"
    rec = _full_record()
    rec["elements"] = rec["elements"][:-1]
    _write_jsonl(p, [rec])
    with pytest.raises(CqError) as ei:
        load_samples(str(p), cfg)
    assert ei.value.code == E_SAMPLE_INVALID


def test_samples_non_finite(tmp_path):
    cfg = make_config()
    p = tmp_path / "s.jsonl"
    rec = _full_record()
    rec["elements"][0]["value"] = "nan"
    _write_jsonl(p, [rec])
    with pytest.raises(CqError) as ei:
        load_samples(str(p), cfg)
    assert ei.value.code == E_SAMPLE_INVALID


def test_samples_duplicate_element(tmp_path):
    cfg = make_config()
    p = tmp_path / "s.jsonl"
    rec = _full_record()
    rec["elements"].append(dict(rec["elements"][0]))
    rec["elements"] = rec["elements"][:-1] + [rec["elements"][-1]]  # 保持数量但重复
    _write_jsonl(p, [rec])
    with pytest.raises(CqError) as ei:
        load_samples(str(p), cfg)
    assert ei.value.code == E_SAMPLE_INVALID
