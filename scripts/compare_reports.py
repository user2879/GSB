"""比对增量最终态报告与空白全量重建报告的业务字段是否逐字段一致。"""

from __future__ import annotations

import argparse
import json

COMPARE_KEYS = (
    "request_id", "request_content_fp", "normalize_cache_key",
    "fusion_cache_keys", "rerank_cache_key", "discount_cache_keys",
    "fusion_order", "fusion_scores", "contributions", "rerank",
    "high_tier_score_diff",
)


def normalize(per_request: list[dict]) -> dict:
    return {item["request_id"]: {key: item[key] for key in COMPARE_KEYS}
            for item in per_request}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--incremental", required=True)
    parser.add_argument("--full", required=True)
    args = parser.parse_args()
    inc = json.load(open(args.incremental, encoding="utf-8"))
    full = json.load(open(args.full, encoding="utf-8"))

    assert inc["config_hash"] == full["config_hash"], "config_hash 不一致"
    assert inc["candidates_hash"] == full["candidates_hash"], "candidates_hash 不一致"
    assert inc["groups_hash"] == full["groups_hash"], "groups_hash 不一致"
    assert inc["request_order"] == full["request_order"], "请求顺序不一致"

    inc_map = normalize(inc["per_request"])
    full_map = normalize(full["per_request"])
    assert set(inc_map) == set(full_map), "请求集合不一致"
    mismatches = []
    for request_id in inc_map:
        if inc_map[request_id] != full_map[request_id]:
            for key in COMPARE_KEYS:
                if inc_map[request_id][key] != full_map[request_id][key]:
                    mismatches.append((request_id, key))
    if mismatches:
        raise SystemExit(f"发现 {len(mismatches)} 处不一致: {mismatches[:10]}")
    print(f"逐字段一致：{len(inc_map)} 个请求，全部 {len(COMPARE_KEYS)} 类字段匹配")


if __name__ == "__main__":
    main()
