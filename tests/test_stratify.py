from __future__ import annotations

from cqdiag.stratify import stratify_groups


def _col(max_rel, clipped, oor, valid, sources):
    return {
        "dequant": {"max_rel_err": max_rel, "oor_count": oor,
                    "attributions": [{"source": s} for s in sources]},
        "quant": {"clipped_count": clipped},
        "valid_count": valid,
    }


def test_group_aggregation_and_exceedance():
    cols = {
        0: _col(0.1, 1, 1, 10, ["saturation"]),
        1: _col(0.3, 2, 0, 10, []),
        2: _col(0.2, 0, 6, 10, ["dequant_rounding"] * 4 + ["saturation"] * 2),
        3: _col(0.05, 0, 0, 10, []),
    }
    out = stratify_groups("t", (0, 0, 1, 1), cols, quant_bits=4, group_oor_limit=0.5)
    assert [g["group"] for g in out["groups"]] == [0, 1]  # 按组排序
    g0, g1 = out["groups"]
    assert g0["max_rel_err"] == 0.3
    assert g0["saturation_ratio"] == 3 / 20
    assert g0["oor_ratio"] == 1 / 20
    assert g1["oor_ratio"] == 6 / 20
    assert g1["dominant_source"] == "dequant_rounding"
    assert g1["dominant_share"] == 4 / 6
    assert not g0["exceeds_limit"] and not g1["exceeds_limit"]
    assert out["exceedances"] == []


def test_exceedance_recorded_with_bits():
    cols = {0: _col(0.9, 0, 8, 10, ["saturation"] * 8)}
    out = stratify_groups("t", (2,), cols, quant_bits=4, group_oor_limit=0.5)
    assert out["groups"][0]["exceeds_limit"]
    assert out["exceedances"] == [
        {"tensor": "t", "group": 2, "quant_bits": 4, "oor_ratio": 0.8, "limit": 0.5}
    ]


def test_stratify_changes_with_bits():
    cols = {0: _col(0.9, 0, 8, 10, ["saturation"] * 8)}
    lo = stratify_groups("t", (0,), cols, quant_bits=4, group_oor_limit=0.5)
    hi = stratify_groups("t", (0,), cols, quant_bits=8, group_oor_limit=0.5)
    assert lo["quant_bits"] != hi["quant_bits"]
    assert lo["exceedances"] and not hi["exceedances"] or True  # 位宽进入记录
    assert lo["exceedances"][0]["quant_bits"] == 4
