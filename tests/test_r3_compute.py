from concurrent.futures import ThreadPoolExecutor

from ckpt_diag.compute import compute_batch
from ckpt_diag.samplegen import build_spec_dict
from ckpt_diag.spec import load_spec

import json


def _shard(tmp_path):
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(build_spec_dict()))
    return load_spec(spec_path).shard("shard-alpha")


def test_deterministic_under_sample_permutation(tmp_path):
    shard = _shard(tmp_path)
    a = compute_batch(shard, "b1", 1, ("s-0", "s-1", "s-2"), ("L0", "L1"))
    b = compute_batch(shard, "b1", 1, ("s-2", "s-0", "s-1"), ("L1", "L0"))
    assert a.loss_summary == b.loss_summary
    assert a.param_summary == b.param_summary
    assert a.total_loss == b.total_loss
    assert a.sample_ids == b.sample_ids


def test_distinct_param_versions_change_summary(tmp_path):
    shard = _shard(tmp_path)
    v1 = compute_batch(shard, "b1", 1, ("s-0",), ())
    v2 = compute_batch(shard, "b1", 2, ("s-0",), ())
    assert v1.param_summary != v2.param_summary
    assert v1.loss_summary != v2.loss_summary


def test_concurrent_recompute_bit_identical(tmp_path):
    shard = _shard(tmp_path)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(
            lambda i: compute_batch(shard, "b-parallel", 1,
                                    ("s-0", "s-1", "s-2", "s-3"), ()),
            range(16)))
    digests = {r.loss_summary for r in results}
    params = {r.param_summary for r in results}
    assert digests == {results[0].loss_summary}
    assert params == {results[0].param_summary}


def test_scope_closure_includes_ancestors(tmp_path):
    shard = _shard(tmp_path)
    result = compute_batch(shard, "b-scope", 1, ("s-0", "s-1"), ("L2",))
    # L2 requires L1 which requires L0
    assert set(result.scope_layers) == {"L0", "L1", "L2"}


def test_loss_summary_binds_identifiers(tmp_path):
    shard = _shard(tmp_path)
    result = compute_batch(shard, "b-x", 1, ("s-0",), ())
    assert result.shard_id == "shard-alpha"
    assert result.batch_id == "b-x"
    assert result.sample_ids == ("s-0",)
