"""生成确定性本地样本：多来源配额记录 JSON、访问事件流 NDJSON、仲裁规则说明。
固定整数种子，重复运行得到相同内容摘要。"""

import hashlib
import json
import os
import random

SEED = 20261122
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "samples")


def _rec(source_id, source_type, format_version, fields):
    return {"source_id": source_id, "source_type": source_type,
            "format_version": format_version, "field_count": len(fields),
            "fields": fields}


def generate(rng):
    sources = []
    volumes = []
    for i in range(1, 13):
        vid = f"vol-{i:02d}"
        kind = ("A", "B", "C")[(i - 1) % 3]
        sids = []

        def add(rec):
            sources.append(rec)
            sids.append(rec["source_id"])

        cap_base = rng.randint(4, 16) * 1024 * 1024
        ops_base = rng.randint(50, 400)
        if kind in ("A", "C"):
            add(_rec(f"{vid}-cap2", "capacity_claim", 2,
                     {"capacity_limit_bytes": cap_base,
                      "block_size_bytes": 4096,
                      "block_count": cap_base // 4096}))
        if kind in ("A", "B"):
            add(_rec(f"{vid}-cap1", "capacity_claim", 1,
                     {"capacity_limit_bytes": cap_base + 4096}))
        if kind in ("A", "B"):
            add(_rec(f"{vid}-rate2", "rate_claim", 2,
                     {"ops_per_second": ops_base,
                      "burst_bucket_capacity": ops_base * 2,
                      "scale_factor": rng.choice((1, 2, 3))}))
        if kind in ("A", "C"):
            add(_rec(f"{vid}-rate1", "rate_claim", 1,
                     {"ops_per_second": ops_base}))
        if kind == "B":
            add(_rec(f"{vid}-der2", "derived_claim", 2,
                     {"metric": "capacity_limit_bytes",
                      "base_source_id": f"{vid}-cap1",
                      "coefficient": 1, "offset": 8192,
                      "divisor": 1, "rounding": "floor"}))
        elif kind == "C":
            add(_rec(f"{vid}-der2", "derived_claim", 2,
                     {"metric": "ops_per_second",
                      "base_source_id": f"{vid}-rate1",
                      "coefficient": 2, "offset": 0,
                      "divisor": 1, "rounding": "floor"}))
        else:
            add(_rec(f"{vid}-der1", "derived_claim", 1,
                     {"metric": "capacity_limit_bytes",
                      "base_source_id": f"{vid}-cap2",
                      "coefficient": 1, "offset": 0}))
        volumes.append({"volume_id": vid, "sources": sorted(sids)})
    return {"volumes": volumes, "sources": sources}


def generate_events(rng, declaration):
    events = []
    ts = 1_700_000_000_000
    for vol in declaration["volumes"]:
        vid = vol["volume_id"]
        op_heavy = int(vid.split("-")[1]) % 2 == 0
        for _ in range(250):
            ts += rng.randint(1, 20)
            if op_heavy:
                events.append({"volume_id": vid, "ts_ms": ts,
                               "op_count": rng.randint(5, 20),
                               "op_bytes": rng.randint(4096, 32768)})
            else:
                events.append({"volume_id": vid, "ts_ms": ts,
                               "op_count": rng.randint(1, 3),
                               "op_bytes": rng.randint(4096, 1024 * 1024)})
    return events


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = random.Random(SEED)
    declaration = generate(rng)
    events = generate_events(rng, declaration)
    rules = {
        "rule_version": "arb-v1",
        "description": "仲裁规则：来源类型偏置 -> 格式版本降序 -> 来源标识升序；"
                       "目标口径仅接受 v2 及以上格式版本的记录。",
        "target_profile": {
            "profile_id": "claims_v2_plus",
            "profile_version": 2,
            "allowed_source_types": ["capacity_claim", "rate_claim",
                                     "derived_claim"],
            "min_format_version": 2,
        },
    }
    paths = {}
    with open(os.path.join(OUT_DIR, "declaration.json"), "w", encoding="utf-8") as fh:
        json.dump(declaration, fh, ensure_ascii=False, sort_keys=True, indent=2)
        paths["declaration"] = fh.name
    with open(os.path.join(OUT_DIR, "events.ndjson"), "w", encoding="utf-8") as fh:
        for ev in events:
            fh.write(json.dumps(ev, sort_keys=True) + "\n")
        paths["events"] = fh.name
    with open(os.path.join(OUT_DIR, "rules.json"), "w", encoding="utf-8") as fh:
        json.dump(rules, fh, ensure_ascii=False, sort_keys=True, indent=2)
        paths["rules"] = fh.name
    for name, path in sorted(paths.items()):
        with open(path, "rb") as fh:
            digest = hashlib.sha256(fh.read()).hexdigest()
        print(f"{name}: {os.path.basename(path)} sha256={digest}")
    print(f"volumes={len(declaration['volumes'])} "
          f"sources={len(declaration['sources'])} events={len(events)}")


if __name__ == "__main__":
    main()
