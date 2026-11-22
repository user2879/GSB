import numpy as np

from peft_fp import errors
from peft_fp.diagnose import (
    CAUSE_BASE_VERSION,
    CAUSE_MERGE_ORDER,
    CAUSE_QUANT_TRUNCATION,
    _attribute_flip,
    cross_version_attribution,
)
from peft_fp.fixedpoint import STORAGE_BAND, quantize
from peft_fp.merge import check_reversible, merge_layer, split_layer
from peft_fp.train import TrainedAdapter


def _adapter(name, layer, a_vals, b_vals, alpha=1.0, rank=1, frac_bits=12):
    return TrainedAdapter(
        task_name=name,
        target_layer=layer,
        rank=rank,
        alpha=alpha,
        frac_bits=frac_bits,
        A=quantize(np.asarray(a_vals, dtype=np.float64), frac_bits),
        B=quantize(np.asarray(b_vals, dtype=np.float64), frac_bits),
        losses_fixed=[],
        ledger=[],
    )


def test_merge_split_exact_roundtrip():
    w = quantize(np.array([[0.5, -0.25], [1.0, 0.75]]), 12)
    ad1 = _adapter("b_task", "l", [[0.1, 0.2]], [[0.3], [0.1]])
    ad2 = _adapter("a_task", "l", [[-0.2, 0.4]], [[0.2], [-0.3]])
    merged, steps, saturated = merge_layer(w, [ad1, ad2], 12)
    assert not saturated
    # fixed task-name order: a_task first
    assert [s["task"] for s in steps] == ["a_task", "b_task"]
    recon = split_layer(merged, [ad1, ad2], 12)
    assert np.array_equal(recon, w)
    assert check_reversible(w, merged, steps, [ad1, ad2], 12) == []


def test_merge_irreversible_on_saturation():
    w = quantize(np.array([[0.5]]), 12)
    big = float(STORAGE_BAND) / (1 << 12) * 4  # 4x the band in value terms
    ad = _adapter("t", "l", [[1.0]], [[big]], alpha=1.0)
    merged, steps, saturated = merge_layer(w, [ad], 12)
    assert saturated
    failures = check_reversible(w, merged, steps, [ad], 12)
    assert failures and failures[0]["code"] == errors.E_IRREVERSIBLE_MERGE
    assert "l" in failures[0]["layer"]


def test_merge_order_probe_attributes_correctly():
    # single adapter -> merge order cannot matter -> quant truncation
    layer = type("L", (), {"weight": np.array([[0.5]]), "bias": np.array([0.0])})
    ad = _adapter("t", "l", [[0.1]], [[0.2]])
    cause = _attribute_flip(None, layer, 12, None, None, layer_adapters=[ad])
    assert cause == CAUSE_QUANT_TRUNCATION
    # base version change dominates
    cause = _attribute_flip(None, layer, 12, "fp-a", "fp-b", layer_adapters=[ad])
    assert cause == CAUSE_BASE_VERSION


def test_cross_version_attribution():
    flips = [{"sample_id": "s1", "cause": CAUSE_QUANT_TRUNCATION}]
    assert cross_version_attribution(flips, "a", "a")[0]["cause"] == CAUSE_QUANT_TRUNCATION
    assert cross_version_attribution(flips, "a", "b")[0]["cause"] == CAUSE_BASE_VERSION


def test_merge_order_constant():
    # merge order enum exists for the saturation-sensitive path
    assert CAUSE_MERGE_ORDER == "merge_order"


def test_tight_tolerance_flip_attribution(sample_dir, tmp_path):
    """With a tight tolerance the config tier fails while the higher tier
    passes; flips must be attributed to quantization truncation (fixed-point
    addition is order-independent here, and the base is unchanged)."""
    import os
    import shutil

    import yaml

    from peft_fp.diagnose import CAUSE_QUANT_TRUNCATION
    from peft_fp.pipeline import run_pipeline

    work = str(tmp_path / "tight")
    shutil.copytree(sample_dir, work)
    cfg_path = os.path.join(work, "peft_config.yaml")
    cfg = yaml.safe_load(open(cfg_path))
    cfg["atol"] = 1e-4
    cfg["rtol"] = 1e-3
    yaml.safe_dump(cfg, open(cfg_path, "w"), sort_keys=True)

    report = run_pipeline(work, str(tmp_path / "store"))
    flips = [f for d in report["diagnostics"] for f in d["flips"]]
    assert flips, "expected at least one verdict flip under tight tolerance"
    assert all(f["cause"] == CAUSE_QUANT_TRUNCATION for f in flips)
    assert all(d["first_exceeding_stage"] == "adapter" for d in report["diagnostics"])
