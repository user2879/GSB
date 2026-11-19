from __future__ import annotations

from audit_forensics.canonical import canonical_json
from helpers import (
    BASE_TS,
    case_signature,
    event,
    import_workspace,
    rules_config,
    write_v1,
    write_v2,
)


def _events() -> list[dict]:
    return [
        event(
            "a1",
            trace_id="T-A",
            actor="alice",
            object_name="vault",
            ts=BASE_TS + 1_000,
            seq=1,
            severity="high",
        ),
        event(
            "b1",
            trace_id="T-B",
            actor="alice",
            object_name="vault",
            ts=BASE_TS + 2_000,
            seq=1,
            severity="medium",
        ),
        event(
            "c1",
            trace_id="T-C",
            actor="alice",
            object_name="vault",
            ts=BASE_TS + 3_000,
            seq=1,
            severity="critical",
        ),
        event(
            "d1",
            trace_id="T-D",
            actor="bob",
            object_name="db",
            ts=BASE_TS + 4_000,
            seq=1,
            severity="high",
        ),
    ]


def _rule_set() -> dict:
    return rules_config(
        "rules-7",
        [
            {
                "rule_id": "R-ALICE",
                "conditions": {
                    "actors": ["alice"],
                    "objects": ["vault"],
                    "time_window": {"window_ms": 5_000},
                },
                "group_by": ["object"],
            },
            {
                "rule_id": "R-HIGH",
                "conditions": {"severities": ["high", "critical"]},
                "group_by": ["actor"],
            },
        ],
    )


def test_case_keys_and_members_are_order_and_format_invariant(tmp_path):
    all_v1_a = write_v1(tmp_path / "all-v1-a.jsonl", _events())
    all_v1_b = write_v1(tmp_path / "all-v1-b.jsonl", reversed(_events()))

    v2_events = []
    for item in _events():
        cloned = dict(item)
        cloned["ts"] = (
            f"2026-11-19T00:00:{int((item['ts'] - BASE_TS) / 1000):02d}Z"
        )
        v2_events.append(cloned)
    all_v2 = write_v2(tmp_path / "all-v2.json", v2_events)
    mixed_a = write_v1(tmp_path / "mixed-a.jsonl", [_events()[2], _events()[0]])
    mixed_b = write_v2(tmp_path / "mixed-b.json", [_events()[3], _events()[1]])

    cases = []
    for index, files in enumerate(
        [
            [all_v1_a, all_v1_b],
            [all_v2],
            [mixed_a, mixed_b],
            [mixed_b, mixed_a],
        ]
    ):
        result = import_workspace(
            tmp_path / f"ws-{index}", files, _rule_set(), workers=4
        )
        cases.append(case_signature(result.payload))

    assert len(set(cases)) == 1
    first = import_workspace(tmp_path / "ws-signature", [all_v1_a], _rule_set()).payload
    alice_case = next(case for case in first["cases"] if case["rule_id"] == "R-ALICE")
    assert alice_case["member_event_ids"] == ["a1", "b1", "c1"]
    assert alice_case["trace_ids"] == ["T-A", "T-B", "T-C"]


def test_rule_version_change_forces_full_rebuild(tmp_path):
    source = write_v1(tmp_path / "events.jsonl", _events())
    initial = import_workspace(
        tmp_path / "workspace", [source], _rule_set()
    ).payload
    initial_ids = {case["case_id"] for case in initial["cases"]}
    initial_events = len(initial["events"])

    changed = rules_config(
        "rules-8",
        [
            {
                "rule_id": "R-ONLY-CRITICAL",
                "conditions": {"severities": ["critical"]},
                "group_by": ["actor"],
            }
        ],
    )
    rebuilt = import_workspace(tmp_path / "workspace", [], changed)
    report = rebuilt.payload
    assert report["rebuild"]["required"] is True
    assert report["rebuild"]["reason"] == "rules_version_changed"
    assert any(
        notice["code"] == "E_RULES_VERSION_CHANGED" for notice in report["notices"]
    )
    assert {case["case_id"] for case in report["cases"]} != initial_ids
    assert len(report["events"]) == initial_events
    assert {case["rule_id"] for case in report["cases"]} == {"R-ONLY-CRITICAL"}


def test_same_version_but_changed_rule_content_rebuilds(tmp_path):
    source = write_v1(tmp_path / "events.jsonl", _events())
    import_workspace(tmp_path / "workspace", [source], _rule_set())
    changed_same_version = rules_config(
        "rules-7",
        [
            {
                "rule_id": "R-BOB",
                "conditions": {"actors": ["bob"]},
                "group_by": ["object"],
            }
        ],
    )
    report = import_workspace(
        tmp_path / "workspace", [], changed_same_version
    ).payload
    assert report["rebuild"]["reason"] == "rules_config_changed"
    assert any(
        notice["code"] == "E_RULES_CONFIG_CHANGED" for notice in report["notices"]
    )
    assert report["cases"][0]["member_event_ids"] == ["d1"]


def test_serial_and_parallel_import_are_identical(tmp_path):
    events = _events()
    files = [
        write_v1(tmp_path / "p1.jsonl", [events[2], events[0]]),
        write_v1(tmp_path / "p2.jsonl", [events[3]]),
        write_v2(
            tmp_path / "p3.json",
            [
                {
                    **events[1],
                    "ts": "2026-11-19T00:00:02Z",
                }
            ],
        ),
    ]
    serial = import_workspace(
        tmp_path / "serial", files, _rule_set(), workers=1
    ).payload
    parallel = import_workspace(
        tmp_path / "parallel", files, _rule_set(), workers=8
    ).payload
    assert canonical_json(serial["cases"]) == canonical_json(parallel["cases"])
    assert canonical_json(serial["events"]) == canonical_json(parallel["events"])
