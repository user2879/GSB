"""隔离指令回执与域级隔离结论的逐字节一致性判定(业务码 E_CONVERGE)。

平台按事件流与域状态重算每个实例的期望隔离状态并聚合成域级结论,
与回执聚合视图做逐字节(规范化 JSON)比对; 不一致、实例未回执或
回执序号乱序时, 稳定列出偏差实例、域、期望与实际摘要。
"""

from __future__ import annotations

from .dispatch import CommandResult
from .errors import DiagError
from .state import Engine
from .util import canonical, digest_obj


def check_domain_convergence(engine: Engine, domain: str,
                             results: list[CommandResult]) -> dict:
    """校验一个域的指令回执与期望结论, 通过时返回域级结论视图。"""
    expected = engine.expected_view(domain)
    ok_results = [r for r in results if r.domain == domain and r.status == "ok"]
    by_instance: dict[str, list[CommandResult]] = {}
    for res in ok_results:
        by_instance.setdefault(res.instance, []).append(res)

    deviations: list[dict] = []
    actual_instances: dict[str, dict] = {}
    for iid in engine.domain_instances(domain):
        exp_iso = expected["instances"][iid]["isolated"]
        receipts = by_instance.get(iid, [])
        if receipts:
            seqs = [r.receipt["cmd_seq"] for r in receipts]
            if seqs != sorted(seqs):
                deviations.append({"instance": iid, "field": "cmd_seq",
                                   "expected": "increasing", "actual": seqs})
            latest = receipts[-1].receipt
            actual_instances[iid] = {"isolated": latest["isolated"]}
            if latest["isolated"] != exp_iso:
                deviations.append({"instance": iid, "field": "isolated",
                                   "expected": exp_iso, "actual": latest["isolated"]})
        else:
            actual_instances[iid] = {"isolated": "<missing>"}
            if exp_iso:
                deviations.append({"instance": iid, "field": "receipt",
                                   "expected": True, "actual": "<missing>"})

    # 逐字节比对: 仅针对本批被下发指令的实例子集, 期望视图与回执视图
    # 规范化后逐字节一致才算收敛; 未回执且期望隔离的实例已在上方记录偏差。
    commanded = sorted(by_instance)
    expected_cmp = {"domain": domain,
                    "instances": {iid: expected["instances"][iid]
                                  for iid in commanded}}
    actual_cmp = {"domain": domain,
                  "instances": {iid: actual_instances[iid] for iid in commanded}}
    expected_bytes = canonical(expected_cmp).encode("utf-8")
    actual_bytes = canonical(actual_cmp).encode("utf-8")
    if deviations or expected_bytes != actual_bytes:
        if not deviations:
            deviations.append({"instance": "<domain>", "field": "view",
                               "expected": digest_obj(expected_cmp),
                               "actual": digest_obj(actual_cmp)})
        raise DiagError("E_CONVERGE",
                        f"domain {domain!r}: isolation conclusion mismatch "
                        f"on {len(deviations)} item(s)",
                        {"domain": domain,
                         "deviations": sorted(deviations,
                                              key=lambda d: (d["instance"], d["field"])),
                         "expected_digest": digest_obj(expected_cmp),
                         "actual_digest": digest_obj(actual_cmp)})
    return expected
