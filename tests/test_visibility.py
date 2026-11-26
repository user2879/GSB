from pathlib import Path

from mvcc_lab.loader import load_spec
from mvcc_lab.visibility import (
    cause_proportions,
    decide_all,
    decide_query,
)

ROOT = Path(__file__).parents[1]


def _decisions(name="sample_spec.json"):
    spec = load_spec(ROOT / "samples" / name)
    return spec, {d.query_id: d for d in decide_all(spec)}


def test_committed_visibility_and_too_late():
    _, decisions = _decisions()
    # 快照 3：T2(commit 5) 过晚，仅 T1 的 v1 可见。
    q1 = decisions["Q1"]
    assert q1.visible_version == 1
    assert q1.visible_txn_id == "T1"
    excluded = {e.version: e.reasons for e in q1.excluded}
    assert excluded[2] == ("commit_too_late",)


def test_rolled_back_and_pending_excluded():
    _, decisions = _decisions()
    q1 = decisions["Q1"]
    excluded = {e.version: e.reasons for e in q1.excluded}
    assert "rolled_back" in excluded[3]
    assert "uncommitted" in excluded[3]
    assert excluded[4] == ("uncommitted",)


def test_superseded_marks_older_visible():
    _, decisions = _decisions()
    q2 = decisions["Q2"]  # 快照 5，v2 可见
    assert q2.visible_version == 2
    excluded = {e.version: e.reasons for e in q2.excluded}
    assert excluded[1] == ("superseded",)


def test_own_pending_version_visible_to_self():
    _, decisions = _decisions()
    # RP-3 的读事务是 T4（pending），R-C 由 T4 写入 -> 自身可见。
    q7 = decisions["Q7"]
    assert q7.visible_version == 1
    assert q7.visible_txn_id == "T4"
    # 其它读事务看 T4 的未提交版本则不可见。
    q8 = decisions["Q8"]
    assert q8.visible_version is None
    assert q8.excluded[0].reasons == ("uncommitted",)


def test_reason_chain_stable_order():
    _, decisions = _decisions()
    q1 = decisions["Q1"]
    # 理由链始终按固定 REASON_ORDER 输出，与遍历顺序无关。
    assert q1.reason_chain == tuple(
        r for r in
        ("chain_broken", "commit_too_late", "rolled_back",
         "superseded", "uncommitted")
        if r in q1.reason_chain
    )


def test_summary_digest_is_stable():
    spec, first = _decisions()
    again = {d.query_id: d for d in decide_all(spec)}
    for qid, decision in first.items():
        assert decision.summary_digest() == again[qid].summary_digest()


def test_tie_break_when_indistinguishable():
    spec = load_spec(ROOT / "samples" / "sample_tie.json")
    decisions = {d.query_id: d for d in decide_all(spec)}
    qt1 = decisions["QT1"]
    # 两条候选提交序号相同且落在 epsilon=1 内 -> 按版本号降序选 v2。
    assert qt1.tie_applied is True
    assert qt1.error_influenced is True
    assert qt1.visible_version == 2
    assert "error_indistinguishable" in qt1.causes
    props = cause_proportions(list(decisions.values()))
    assert props["error_indistinguishable"] == 1.0
    assert props["rule_version_conflict"] == 0.0
    assert props["version_chain_broken"] == 0.0


def test_rule_conflict_cause():
    spec = load_spec(ROOT / "samples" / "sample_spec.json")
    decisions = decide_all(spec, forced_rule_conflict=True)
    assert all("rule_version_conflict" in d.causes for d in decisions)
    props = cause_proportions(decisions)
    assert props["rule_version_conflict"] == 1.0


def test_epsilon_zero_means_no_tie():
    _, decisions = _decisions()
    # 主样本 epsilon=0，即便存在同序号读点也不应标记误差影响。
    assert all(not d.error_influenced for d in decisions.values())
