from mvcc_lab import CAUSE_CHAIN_BROKEN, REASON_CHAIN_BROKEN
from mvcc_lab.model import (
    ReadPoint,
    RecordVersion,
    Spec,
    Transaction,
    VisibilityQuery,
)
from mvcc_lab.visibility import cause_proportions, decide_query


def _broken_spec():
    txns = (
        Transaction("T1", 1, "committed", 2),
        Transaction("T2", 2, "committed", 4),
    )
    # v2 指向不存在的 v9：越过加载校验、直接构造断链对象。
    records = (
        RecordVersion("R1", "T1", 1, "d1", None),
        RecordVersion("R1", "T2", 2, "d2", 9),
    )
    return Spec(
        spec_version=1,
        epsilon=0.0,
        global_concurrency=2,
        per_readpoint_quota=1,
        transactions=txns,
        records=records,
        read_points=(ReadPoint("RP1", 4, "T1"),),
        queries=(VisibilityQuery("Q1", "RP1", "R1"),),
        row_chains={"R1": records},
        max_seq=4,
    )


def test_broken_chain_is_flagged_as_cause():
    spec = _broken_spec()
    decision = decide_query(spec, spec.queries[0])
    assert CAUSE_CHAIN_BROKEN in decision.causes
    assert decision.error_influenced is True
    all_reasons = {r for exc in decision.excluded for r in exc.reasons}
    assert REASON_CHAIN_BROKEN in all_reasons
    props = cause_proportions([decision])
    assert props["version_chain_broken"] == 1.0
