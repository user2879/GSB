"""独立复算器。

只消费生成输出（spec.canonical.json、continuous/、mapping/、grid.json、
ledger.json、manifest.json），不复用生成器任何内存状态：
  1. 用 manifest 校验所有文件逐字节哈希（被外部篡改 -> E_LEDGER_STALE）；
  2. 重算 spec/continuous/mapping/grid/ledger 内容哈希并与账本引用比对
     （账本引用过期 -> E_LEDGER_STALE）；
  3. 仅依据工件中的连续值、量化差值、规则与阈值，重算每条判定的
     误差区间与结论；
  4. 与生成器账本逐条比对，首个不一致 -> E_LEDGER_MISMATCH，
     报错明细包含观测值、量化前后差值与阈值。
"""

from __future__ import annotations

import json
from pathlib import Path

from .errors import ErrorCode, LevelDiagError
from .util import classify, content_hash, file_hash


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _stale(message: str, details: dict) -> LevelDiagError:
    return LevelDiagError(ErrorCode.E_LEDGER_STALE, message, details)


def verify_freshness(out_dir: str | Path) -> dict:
    out = Path(out_dir)
    manifest = _read(out / "manifest.json")
    for rel, declared in manifest["files"].items():
        p = out / rel
        if not p.exists():
            raise _stale(f"工件缺失: {rel}", {"file": rel})
        actual = file_hash(p)
        if actual != declared:
            raise _stale(f"工件被篡改且未重算: {rel}",
                         {"file": rel, "declared": declared, "actual": actual})

    spec = _read(out / "spec.canonical.json")
    if content_hash(spec) != manifest["spec_hash"]:
        raise _stale("规格内容哈希与 manifest 不一致", {"file": "spec.canonical.json"})
    ledger = _read(out / "ledger.json")
    for ref in ("spec_hash", "continuous_hash", "mapping_hash", "grid_hash"):
        if ledger[ref] != manifest[ref]:
            raise _stale(f"账本引用 {ref} 过期",
                         {"reference": ref, "ledger": ledger[ref],
                          "manifest": manifest[ref]})
    if ledger["ledger_hash"] != manifest["ledger_hash"]:
        raise _stale("ledger_hash 与 manifest 不一致",
                     {"file": "ledger.json"})
    # 重算 ledger_hash，防止条目被改后哈希未更新
    rebuilt = content_hash({
        "spec_hash": ledger["spec_hash"],
        "continuous_hash": ledger["continuous_hash"],
        "mapping_hash": ledger["mapping_hash"],
        "grid_hash": ledger["grid_hash"],
        "entries": ledger["entries"]})
    if rebuilt != ledger["ledger_hash"]:
        raise _stale("账本条目与其 ledger_hash 不自洽", {"file": "ledger.json"})
    return manifest


def _recompute_status(entry: dict) -> str:
    """复算器独立实现的判定口径（与 util.classify 同口径，但仅消费工件数据）。"""
    direction = entry["direction"]
    observed = entry["observed"]
    threshold = entry["threshold"]
    band = float(entry["comparison_tolerance"]) + \
        float(entry["propagated_error_bound"])
    if direction == "interval":
        lo, hi = threshold["min"], threshold["max"]
        return "PASS" if lo <= observed <= hi else "FAIL"
    if direction == "eq":
        return "PASS" if abs(float(observed) - float(threshold)) <= band else "FAIL"
    return classify(direction, float(observed), float(threshold),
                    float(entry["comparison_tolerance"]),
                    float(entry["propagated_error_bound"]))


def recompute(out_dir: str | Path, verify: bool = True) -> dict:
    out = Path(out_dir)
    if verify:
        verify_freshness(out)
    ledger = _read(out / "ledger.json")
    mapping_index = _read(out / "mapping" / "index.json")

    # 独立重算映射表内容哈希，验证量化分片与账本引用一致
    shards = {}
    for rel in mapping_index["region_shard_files"]:
        shard = _read(out / rel)
        shards[shard["region_id"]] = shard
    rebuilt_mapping_hash = content_hash(
        {"rules": mapping_index["rules"], "region_shards": shards})
    if rebuilt_mapping_hash != mapping_index["mapping_hash"]:
        raise _stale("映射表分片重算哈希与索引不一致",
                     {"mapping_index": mapping_index["mapping_hash"],
                      "rebuilt": rebuilt_mapping_hash})

    mismatches = []
    first = None
    for entry in ledger["entries"]:
        status = _recompute_status(entry)
        band = float(entry["comparison_tolerance"]) + \
            float(entry["propagated_error_bound"])
        rebuilt_interval = [float(entry["observed"]) - band,
                            float(entry["observed"]) + band]
        if status != entry["status"] or \
                rebuilt_interval != [entry["plausible_low"], entry["plausible_high"]]:
            detail = {
                "constraint_id": entry["constraint_id"],
                "generator_status": entry["status"],
                "recomputed_status": status,
                "observed": entry["observed"],
                "threshold": entry["threshold"],
                "direction": entry["direction"],
                "plausible_interval": rebuilt_interval,
                "truth_sources": entry["truth_sources"],
            }
            if "evidence" in entry and isinstance(entry["evidence"], dict):
                ev = entry["evidence"]
                if "continuous" in ev and "quantized" in ev and "delta" in ev:
                    detail["quantization_trace"] = {
                        "continuous": ev["continuous"],
                        "quantized": ev["quantized"],
                        "delta": ev["delta"]}
                detail["evidence"] = ev
            mismatches.append(detail)
            if first is None:
                first = detail

    result = {
        "spec_hash": ledger["spec_hash"],
        "checked": len(ledger["entries"]),
        "mismatch_count": len(mismatches),
        "recomputed_summary": {
            "pass": sum(1 for e in ledger["entries"]
                        if _recompute_status(e) == "PASS"),
            "fail": sum(1 for e in ledger["entries"]
                        if _recompute_status(e) == "FAIL"),
            "uncertain": sum(1 for e in ledger["entries"]
                             if _recompute_status(e) == "UNCERTAIN"),
        },
        "mismatches": mismatches,
    }
    if first is not None:
        raise LevelDiagError(
            ErrorCode.E_LEDGER_MISMATCH,
            f"首个不一致判定: {first['constraint_id']}",
            {"first_mismatch": first, "all_mismatches": mismatches})
    return result
