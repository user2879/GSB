"""确定性样本生成：声明 JSON、会话 JSON 与单调不减的 NDJSON 事件流。

使用 ``numpy.random.default_rng(固定整数种子)``，两档样本在任意机器逐字节复现。
生成时刻主动制造：配额耗尽排队、乐观续约、到期回收再分配、跨域并发等场景。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .spec import ACTIONS

SMALL_PLAN = {
    "name": "small",
    "domains": [
        {"domain_id": "dom-a", "port": 19001, "slots": 4, "lease_cap": 3,
         "lease_ms": 600, "renew_ms": 200, "seed": 101},
        {"domain_id": "dom-b", "port": 19002, "slots": 3, "lease_cap": 2,
         "lease_ms": 500, "renew_ms": 150, "seed": 202},
    ],
    "instances": [
        ("svc-a1", "dom-a", 1, 2),
        ("svc-a2", "dom-a", 2, 1),
        ("svc-a3", "dom-a", 3, 2),
        ("svc-b1", "dom-b", 1, 1),
        ("svc-b2", "dom-b", 2, 2),
    ],
    "event_count": 48,
    "master_seed": 20261125,
}

MEDIUM_PLAN = {
    "name": "medium",
    "domains": [
        {"domain_id": "dom-a", "port": 19101, "slots": 8, "lease_cap": 6,
         "lease_ms": 800, "renew_ms": 250, "seed": 301},
        {"domain_id": "dom-b", "port": 19102, "slots": 6, "lease_cap": 4,
         "lease_ms": 700, "renew_ms": 220, "seed": 302},
        {"domain_id": "dom-c", "port": 19103, "slots": 5, "lease_cap": 4,
         "lease_ms": 650, "renew_ms": 200, "seed": 303},
    ],
    "instances": [
        ("svc-a1", "dom-a", 1, 2), ("svc-a2", "dom-a", 2, 3),
        ("svc-a3", "dom-a", 3, 1), ("svc-a4", "dom-a", 4, 2),
        ("svc-b1", "dom-b", 1, 2), ("svc-b2", "dom-b", 2, 1),
        ("svc-b3", "dom-b", 3, 3),
        ("svc-c1", "dom-c", 1, 1), ("svc-c2", "dom-c", 2, 2),
        ("svc-c3", "dom-c", 3, 2),
    ],
    "event_count": 160,
    "master_seed": 20261126,
}


def _domain_records(plan: dict) -> list[dict]:
    return [
        {
            "domain_id": d["domain_id"], "host": "127.0.0.1", "port": d["port"],
            "slots": d["slots"], "lease_cap": d["lease_cap"],
            "lease_ms": d["lease_ms"], "renew_ms": d["renew_ms"], "seed": d["seed"],
        }
        for d in plan["domains"]
    ]


def _instance_records(plan: dict) -> list[dict]:
    return [
        {"instance_id": iid, "domain_id": did, "priority": prio, "demand": demand}
        for iid, did, prio, demand in plan["instances"]
    ]


def _generate_events(plan: dict) -> list[dict]:
    rng = np.random.default_rng(plan["master_seed"])
    instance_ids = [iid for iid, _did, _prio, _demand in plan["instances"]]
    instance_domain = {iid: did for iid, did, _p, _d in plan["instances"]}
    domain_lease_ms = {d["domain_id"]: d["lease_ms"] for d in plan["domains"]}
    domain_renew_ms = {d["domain_id"]: d["renew_ms"] for d in plan["domains"]}

    events: list[dict] = []
    arrival = 0
    active: dict[str, dict] = {}
    seq = 1
    while len(events) < plan["event_count"]:
        iid = str(rng.choice(instance_ids))
        did = instance_domain[iid]
        if iid in active:
            grant_arrival = active[iid]["arrival"]
            age = arrival - grant_arrival
            # 租约窗口内优先续约，窗口外释放，制造到期回收再分配
            if age < domain_lease_ms[did] - domain_renew_ms[did]:
                action = "renew" if rng.random() < 0.6 else "release"
            else:
                action = "release"
        else:
            action = "acquire"
        events.append({
            "seq": seq, "arrival_ms": int(arrival),
            "instance_id": iid, "action": action,
        })
        if action == "acquire":
            active[iid] = {"arrival": arrival}
        elif action == "release":
            active.pop(iid, None)
        seq += 1
        arrival += int(rng.integers(40, 180))
        # 周期性让一部分活跃实例跨过租约窗口后自然到期
        if len(events) % 12 == 0:
            arrival += domain_renew_ms[max(domain_lease_ms, key=lambda k: domain_lease_ms[k])]
    return events


def generate_sample(plan: dict, out_dir: str | Path,
                    global_lease_cap: int = 4, scale_power: int = 2,
                    pacing: float = 0.05, batch_size: int = 32) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    events_file = f"events-{plan['name']}.ndjson"
    spec = {
        "domains": _domain_records(plan),
        "instances": _instance_records(plan),
        "events_file": events_file,
    }
    spec_path = out / f"spec-{plan['name']}.json"
    spec_path.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    session = {
        "global_lease_cap": global_lease_cap,
        "scale_power": scale_power,
        "pacing": pacing,
        "batch_size": batch_size,
    }
    session_path = out / "session.json"
    session_path.write_text(json.dumps(session, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    events = _generate_events(plan)
    events_path = out / events_file
    with events_path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, sort_keys=True, ensure_ascii=False) + "\n")
    return {"spec": spec_path, "session": session_path, "events": events_path}


def generate_all(base_dir: str | Path) -> dict[str, dict[str, Path]]:
    base = Path(base_dir)
    return {
        "small": generate_sample(SMALL_PLAN, base / "small", global_lease_cap=3, batch_size=16),
        "medium": generate_sample(MEDIUM_PLAN, base / "medium", global_lease_cap=5, batch_size=40),
    }
