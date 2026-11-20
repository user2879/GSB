"""确定性样本生成器：租户/资源池声明、NDJSON 请求流与异常注入。

使用固定种子的伪随机数生成器，相同参数永远产出相同文件，
可在数百租户、数万请求规模下复现实验。注入只改输入文件或参数，
不修改任何源码。
"""
from __future__ import annotations

import json
import os
import random

from .util import canonical_json

DEFAULT_SEED = 20261121


def generate_spec(out_dir: str, *, tenant_count: int = 8, pool_count: int = 3,
                  seed: int = DEFAULT_SEED):
    rng = random.Random(seed)
    os.makedirs(out_dir, exist_ok=True)
    tenant_ids = [f"T{idx:04d}" for idx in range(tenant_count)]
    tenants = []
    for tid in tenant_ids:
        tenants.append(
            {
                "tid": tid,
                "quota": rng.randint(40, 240),
                "concurrency_cap": rng.randint(2, 12),
                "burst": rng.randint(0, 60),
            }
        )

    pools = []
    for pidx in range(pool_count):
        pid = f"P{pidx:03d}"
        shuffled = list(tenant_ids)
        rng.shuffle(shuffled)
        # 每个池包含全部租户，排序即按稀缺度的优先级序
        pools.append(
            {
                "pid": pid,
                "capacity": rng.randint(60, 320),
                "priority_order": shuffled,
            }
        )

    spec_path = os.path.join(out_dir, "spec.json")
    with open(spec_path, "w", encoding="utf-8") as fh:
        json.dump({"tenants": tenants, "pools": pools}, fh, ensure_ascii=False, indent=2)
    return spec_path, tenants, pools


def generate_requests(out_dir: str, tenants, pools, *, request_count: int = 500,
                      seed: int = DEFAULT_SEED, arrival_step: int = 7):
    rng = random.Random(seed + 1)
    tenant_ids = [t["tid"] for t in tenants]
    pool_ids = [p["pid"] for p in pools]
    # 每租户独立到达游标，保证同租户到达时刻严格递增且不重复
    tenant_clock = {tid: rng.randint(0, 5) for tid in tenant_ids}
    path = os.path.join(out_dir, "requests.ndjson")
    with open(path, "w", encoding="utf-8") as fh:
        for idx in range(request_count):
            tid = tenant_ids[rng.randrange(len(tenant_ids))]
            pid = pool_ids[rng.randrange(len(pool_ids))]
            arrival = tenant_clock[tid]
            tenant_clock[tid] += rng.randint(1, arrival_step)
            request = {
                "rid": f"R{idx:06d}",
                "tid": tid,
                "pid": pid,
                "demand": rng.randint(1, 28),
                "arrival": arrival,
                "priority": rng.randint(0, 9),
            }
            fh.write(canonical_json(request) + "\n")
    return path


def generate_dataset(out_dir: str, *, tenant_count=8, pool_count=3, request_count=500,
                     seed=DEFAULT_SEED):
    spec_path, tenants, pools = generate_spec(
        out_dir, tenant_count=tenant_count, pool_count=pool_count, seed=seed
    )
    requests_path = generate_requests(
        out_dir, tenants, pools, request_count=request_count, seed=seed
    )
    return {"spec": spec_path, "requests": requests_path,
            "tenant_count": tenant_count, "pool_count": pool_count,
            "request_count": request_count}


# ---------------------------------------------------------------------- #
# 异常注入：仅生成被污染的输入文件，原样本保持不变
# ---------------------------------------------------------------------- #
def inject_order_regression(src_requests: str, dst_requests: str):
    """E_ORDER：把第二条请求的到达时刻改为早于第一条（全局倒退）。"""
    with open(src_requests, "r", encoding="utf-8") as fh:
        lines = [line for line in fh.read().splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError("请求流不足两条，无法注入到达倒退")
    first = json.loads(lines[0])
    second = json.loads(lines[1])
    second["arrival"] = max(0, first["arrival"] - 5)
    lines[1] = canonical_json(second)
    with open(dst_requests, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return {"rid": second["rid"], "field": "arrival", "arrival": second["arrival"]}


def inject_missing_priority_tenant(src_spec: str, dst_spec: str):
    """E_PRIORITY：把第一个资源池优先级序首位改成不存在的租户标识。"""
    with open(src_spec, "r", encoding="utf-8") as fh:
        spec = json.load(fh)
    missing_tid = "T_MISSING_9999"
    spec["pools"][0]["priority_order"][0] = missing_tid
    with open(dst_spec, "w", encoding="utf-8") as fh:
        json.dump(spec, fh, ensure_ascii=False, indent=2)
    return {"pid": spec["pools"][0]["pid"], "field": "priority_order", "missing_tid": missing_tid}


def inject_ledger_digest(ledger, rid: str):
    """E_LEDGER：改写账本中一条结局但不更新摘要链，制造内容摘要不符。"""
    ledger.tamper_outcome(rid, "REJECT")
    return {"rid": rid, "field": "verdict"}
