from pathlib import Path

from mvcc_lab.invalidation import (
    diff_chains,
    diff_transactions,
    propagate,
    recompute_invalid_subset,
    verify_preserved_unchanged,
)
from mvcc_lab.loader import load_spec
from mvcc_lab.visibility import decide_all

ROOT = Path(__file__).parents[1]


def test_transaction_and_chain_diffs():
    old = load_spec(ROOT / "samples" / "sample_spec.json")
    new = load_spec(ROOT / "samples" / "sample_spec_changed.json")
    assert diff_transactions(old, new)["T4"].startswith(
        "outcome pending->committed"
    )
    assert "chain_appended" in diff_chains(old, new)["R-A"]


def test_propagation_set_is_stable_and_complete():
    old = load_spec(ROOT / "samples" / "sample_spec.json")
    new = load_spec(ROOT / "samples" / "sample_spec_changed.json")
    report = propagate(old, new)
    # R-A 全部查询 + R-C（T4 自身行）失效；R-B 查询保持有效。
    assert set(report["invalidated_query_ids"]) == {
        "Q1", "Q2", "Q3", "Q4", "Q7", "Q8"
    }
    # 输出按读点标识序、行键序、查询标识序稳定。
    q_index = {q.query_id: q for q in new.queries}
    keyed = [
        (q_index[qid].read_point_id, q_index[qid].row_key, qid)
        for qid in report["invalidated_query_ids"]
    ]
    assert keyed == sorted(keyed)
    # 每步传播依据都有完整边描述。
    for step in report["steps"]:
        assert all(step[k] for k in (
            "source_kind", "source_id", "relation",
            "target_kind", "target_id", "reason"))
    assert report["steps"] == sorted(report["steps"],
                                     key=lambda s: s["step"])


def test_only_invalid_subset_recomputed_and_preserved_byte_stable():
    old = load_spec(ROOT / "samples" / "sample_spec.json")
    new = load_spec(ROOT / "samples" / "sample_spec_changed.json")
    report = propagate(old, new)
    old_decisions = {d.query_id: d for d in decide_all(old)}
    merged, preserved = recompute_invalid_subset(
        new, report, old_decisions
    )
    preserved_ids = {item["query_id"] for item in preserved}
    assert preserved_ids == {"Q5", "Q6"}
    verify_preserved_unchanged(old_decisions, preserved)
    # 未失效判定的摘要逐字节保持。
    for item in preserved:
        assert item["summary_digest"] == \
            old_decisions[item["query_id"]].summary_digest()
    # 失效查询已用新规格重算。
    assert "Q1" in merged and "Q7" in merged
