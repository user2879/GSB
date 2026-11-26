import json
from pathlib import Path

import pytest

from dispatch_tokens.errors import ReplayError
from dispatch_tokens.ledger import (COMMITTED, UNKNOWN, Ledger,
                                    spec_digest)
from dispatch_tokens.runner import rebuild_from_scratch
from dispatch_tokens.spec import load_spec

SPEC_PATH = Path(__file__).resolve().parents[1] / "samples" / "demo_spec.json"


@pytest.fixture()
def spec():
    return load_spec(SPEC_PATH)


def test_commit_is_atomic_and_loadable(spec, tmp_path):
    db = tmp_path / "ledger.db"
    ledger = Ledger(db, spec)
    ledger.commit_outcome(
        shard_id="sh-0", node_id="node-a", status=COMMITTED,
        param_version=1, token_id="tok-1", token_version=1,
        summary={"checksum": 123},
        ownership={"decision": "GRANT"},
        bucket={"balance": 100}, cursor_seq=0, updated_ms=100,
        refill_records=[{"node_id": "node-a", "added": 1}],
        decision_records=[{"event_seq": 0, "decision": "GRANT"}])
    ledger.close()

    reopened = Ledger(db, spec)
    outcomes = reopened.load_outcomes()
    assert outcomes["sh-0"]["status"] == COMMITTED
    assert outcomes["sh-0"]["summary"]["checksum"] == 123
    meta = dict(reopened.conn.execute("SELECT key, value FROM meta").fetchall())
    assert int(meta["format_version"]) == 1
    assert meta["spec_digest"] == spec_digest(spec)
    reopened.close()


def test_snapshot_mismatch_is_wholesale_replaced(spec, tmp_path):
    db = tmp_path / "ledger.db"
    ledger = Ledger(db, spec)
    ledger.commit_outcome(
        shard_id="sh-0", node_id="node-a", status=COMMITTED,
        param_version=1, token_id="t", token_version=1, summary=None,
        ownership={"decision": "GRANT"}, bucket={"balance": 1},
        cursor_seq=0, updated_ms=1, refill_records=[], decision_records=[])
    ledger.close()

    mutated = json.loads(Path(SPEC_PATH).read_text())
    mutated["nodes"][0]["rate"] = spec.nodes[0].rate + 1
    from dispatch_tokens.spec import parse_spec
    changed_spec = parse_spec(mutated)
    reopened = Ledger(db, changed_spec)
    # 摘要不一致：旧记录被整体替换，而不是增量合并。
    assert reopened.load_outcomes() == {}
    meta = dict(reopened.conn.execute("SELECT key, value FROM meta").fetchall())
    assert meta["spec_digest"] == spec_digest(changed_spec)
    reopened.close()


def test_unknown_shard_marked_and_replay_verified(spec, tmp_path):
    db = tmp_path / "ledger.db"
    ledger = Ledger(db, spec)
    outcomes = rebuild_from_scratch(spec)
    for shard_id, outcome in outcomes.items():
        ledger.commit_outcome(
            shard_id=shard_id, node_id=outcome["node_id"],
            status=outcome["status"], param_version=outcome["param_version"],
            token_id=outcome["token_id"],
            token_version=outcome["token_version"],
            summary=outcome["summary"], ownership=outcome["ownership"],
            bucket=outcome["bucket"], cursor_seq=outcome["cursor_seq"],
            updated_ms=outcome["updated_ms"],
            refill_records=[], decision_records=[])
    # 强制终止：在飞分片没有完整提交记录，先标 UNKNOWN 并登记回收。
    ledger.mark_unknown("sh-3", "node-a", 99, 9999,
                        {"balance": 0, "param_version": 2})
    stored = ledger.load_outcomes()
    assert stored["sh-3"]["status"] == UNKNOWN
    assert stored["sh-3"]["ownership"]["decision"] == \
        "UNKNOWN_RERUN_RECLAIM_FIRST"
    # UNKNOWN 不参与逐字段重放比对，重放通过。
    ledger.verify_replay(outcomes)
    ledger.close()


def test_replay_mismatch_raises_e_replay_with_location(spec, tmp_path):
    db = tmp_path / "ledger.db"
    ledger = Ledger(db, spec)
    good = rebuild_from_scratch(spec)
    for shard_id, outcome in good.items():
        ledger.commit_outcome(
            shard_id=shard_id, node_id=outcome["node_id"],
            status=outcome["status"], param_version=outcome["param_version"],
            token_id=outcome["token_id"],
            token_version=outcome["token_version"], summary=None,
            ownership=outcome["ownership"], bucket=outcome["bucket"],
            cursor_seq=outcome["cursor_seq"],
            updated_ms=outcome["updated_ms"],
            refill_records=[], decision_records=[])
    tampered = {sid: dict(item) for sid, item in good.items()}
    tampered["sh-1"]["ownership"] = {"decision": "TAMPERED"}
    with pytest.raises(ReplayError) as exc:
        ledger.verify_replay(tampered)
    assert exc.value.code == "E_REPLAY"
    assert exc.value.location == "shards[sh-1]"
    assert exc.value.field == "ownership"
    ledger.close()
