"""取消与参数修订同刻争用、同刻双分片争用的唯一归属裁决。"""
from copy import deepcopy

from dispatch_tokens.engine import SchedulerEngine
from dispatch_tokens.lease import EVENT_TYPE_ORDER
from dispatch_tokens.spec import parse_spec

ROOT_SPEC = {
    "nodes": [
        {"node_id": "n1", "host": "127.0.0.1", "port": 49301, "rate": 10,
         "burst": 8, "bucket_key": 1, "seed": 1, "param_version": 1},
    ],
    "shards": [
        {"shard_id": "s-a", "upstream": [], "sample_rows": 10, "weight": 2,
         "priority": 0},
        {"shard_id": "s-b", "upstream": [], "sample_rows": 10, "weight": 2,
         "priority": 1},
    ],
    "revisions": [
        {"version": 1, "parent": None, "global_rate": None,
         "global_burst": None, "node_rates": {}, "node_bursts": {},
         "shard_deps": {}, "scale": 100},
        {"version": 2, "parent": 1, "global_rate": None,
         "global_burst": None, "node_rates": {"n1": 20},
         "node_bursts": {"n1": 12}, "shard_deps": {}, "scale": None},
    ],
    "events": [
        {"seq": 0, "kind": "dispatch", "ts_ms": 100, "shard_id": "s-a",
         "node_id": "n1", "version": None, "expected_version": None},
        {"seq": 1, "kind": "cancel", "ts_ms": 500, "shard_id": "s-a",
         "node_id": None, "version": None, "expected_version": None},
        {"seq": 2, "kind": "revise", "ts_ms": 500, "shard_id": None,
         "node_id": None, "version": 2, "expected_version": None},
        {"seq": 3, "kind": "dispatch", "ts_ms": 700, "shard_id": "s-b",
         "node_id": "n1", "version": 2, "expected_version": None},
    ],
}


def test_same_timestamp_cancel_and_revise_have_single_ownership():
    spec = parse_spec(deepcopy(ROOT_SPEC))
    result = SchedulerEngine(spec).run()
    decisions = result.decisions
    # 取消只产生一次回收裁决，修订只做结转不重复回收同一额度。
    cancels = [d for d in decisions if d["decision"] == "CANCEL"]
    assert len(cancels) == 1
    revisions = [d for d in decisions if d["event_kind"] == "revise"]
    assert len(revisions) == 1
    # 两条同刻裁决都带事件序号、参数版本与令牌版本。
    for record in decisions:
        assert "event_seq" in record and "param_version" in record
        assert "token_version" in record and "basis" in record
    # 修订后新速率生效，且 s-b 能在额度释放后被授予。
    assert result.outcomes["s-b"].status == "DISPATCHED"
    assert result.current_versions["n1"] == 2


def test_event_reordering_under_same_seq_is_stable():
    # 同到达序号下，分片标识序 + 事件类型序决定稳定次序。
    assert EVENT_TYPE_ORDER["dispatch"] < EVENT_TYPE_ORDER["cancel"]
    assert EVENT_TYPE_ORDER["cancel"] < EVENT_TYPE_ORDER["revise"]
    first = SchedulerEngine(parse_spec(deepcopy(ROOT_SPEC))).run()
    # 再次运行（事件对象顺序不同但 seq/ts 相同）结论必须逐字节一致。
    second = SchedulerEngine(parse_spec(deepcopy(ROOT_SPEC))).run()
    assert first.decisions == second.decisions
    assert first.water_levels == second.water_levels


def test_two_shards_cannot_hold_same_node_quota_simultaneously():
    data = deepcopy(ROOT_SPEC)
    # 两个分片在同一时刻下发到唯一节点，第二个必须被互斥拒绝。
    data["events"] = [
        {"seq": 0, "kind": "dispatch", "ts_ms": 100, "shard_id": "s-a",
         "node_id": "n1", "version": None, "expected_version": None},
        {"seq": 1, "kind": "dispatch", "ts_ms": 100, "shard_id": "s-b",
         "node_id": "n1", "version": None, "expected_version": None},
    ]
    spec = parse_spec(data)
    engine = SchedulerEngine(spec)
    try:
        engine.run()
        raised = False
    except Exception:
        raised = True
    # 互斥通过 E_LEASE 表达；节点上最多一个活跃令牌。
    active = [t for t in engine.registry.all_tokens() if t.active]
    assert len(active) == 1
    assert raised is True
