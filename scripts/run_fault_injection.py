"""故障注入验证：E_RECORD_INVALID / E_CONFIG_INVALID / E_CONFIG_MISMATCH / 阈值边界。"""

from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import json
import shutil
from pathlib import Path

from rerank_core.config import load_config
from rerank_core.errors import (
    ConfigInvalidError,
    ConfigMismatchError,
    RecordInvalidError,
)
from rerank_core.loader import load_candidates
from rerank_core.snapshot import SnapshotStore
from rerank_cli.runner import run_full_rebuild

FAULT_DIR = Path("samples/fault_injection")
RESULT: list[dict] = []


def record_case(name: str, ok: bool, detail: str) -> None:
    RESULT.append({"case": name, "passed": ok, "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def case_bad_records() -> None:
    config = load_config(
        FAULT_DIR / "threshold_config.json", FAULT_DIR / "feature_groups.json"
    )
    candidates, failures = load_candidates(
        FAULT_DIR / "bad_candidates.jsonl", config
    )
    codes = sorted({f.code for f in failures})
    record_case(
        "非有限/负值/空ID/重复ID/坏JSON 全部隔离",
        codes == ["E_RECORD_INVALID"] and len(failures) >= 6,
        f"{len(failures)} 条失败，代码集合={codes}",
    )
    record_case(
        "异常隔离不影响合法请求",
        "req-bad-3" in candidates and len(candidates["req-bad-3"]) == 1,
        f"保留请求={sorted(candidates)}",
    )


def case_bad_config(filename: str, label: str) -> None:
    try:
        load_config(FAULT_DIR / filename, FAULT_DIR / "feature_groups.json")
    except ConfigInvalidError as exc:
        record_case(label, exc.code == "E_CONFIG_INVALID", exc.message)
        return
    record_case(label, False, "未抛出配置异常")


def case_threshold_boundary() -> None:
    snapshot = Path("snapshots/fault_threshold")
    shutil.rmtree(snapshot, ignore_errors=True)
    # 借用 threshold 输入目录结构
    input_dir = Path("samples/fault_threshold_input")
    shutil.rmtree(input_dir, ignore_errors=True)
    input_dir.mkdir(parents=True)
    shutil.copy(FAULT_DIR / "threshold_candidates.jsonl",
                input_dir / "candidates.jsonl")
    shutil.copy(FAULT_DIR / "threshold_config.json",
                input_dir / "rerank_config.json")
    shutil.copy(FAULT_DIR / "feature_groups.json",
                input_dir / "feature_groups.json")
    report = run_full_rebuild(input_dir, snapshot)
    req = next(q for q in report["per_request"] if q["request_id"] == "req-threshold")
    discounts = req["rerank"]["discounts"]
    boundary = [d for d in discounts if d["similarity_fixed"] == d["threshold_fixed"]]
    record_case(
        "相似度恰好等于门限触发折减（>= 判定）",
        bool(boundary),
        f"边界折减 {len(boundary)} 次，示例={boundary[:1]}",
    )
    record_case(
        "折减在定点整数域执行且严格记录前后分",
        all(isinstance(d["before_score_fixed"], int)
            and isinstance(d["after_score_fixed"], int)
            and d["after_score_fixed"] < d["before_score_fixed"]
            for d in discounts),
        f"折减次数={len(discounts)}",
    )


def case_config_mismatch() -> None:
    snapshot = Path("snapshots/fault_mismatch")
    shutil.rmtree(snapshot, ignore_errors=True)
    base_config = load_config(
        Path("samples/sample_input/rerank_config.json"),
        Path("samples/sample_input/feature_groups.json"),
    )
    candidates, failures = load_candidates(
        Path("samples/sample_input/candidates.jsonl"), base_config
    )
    store = SnapshotStore(snapshot)
    from rerank_core.engine import RerankEngine

    engine = RerankEngine(base_config, candidates, failures, store)
    from rerank_core.commands import Command

    engine.handle_command(Command(0, "poll", {}))
    engine.flush_snapshot()
    # 篡改配置权重后另起引擎尝试续跑
    import dataclasses

    changed = dataclasses.replace(
        base_config,
        weights=tuple(base_config.weights[:-1]) + (base_config.weights[-1] + 1,),
    )
    try:
        RerankEngine(changed, candidates, failures, store, resume=True)
    except ConfigMismatchError as exc:
        record_case(
            "配置哈希不一致时 E_CONFIG_MISMATCH 拒绝续跑",
            exc.code == "E_CONFIG_MISMATCH",
            exc.message,
        )
        return
    record_case("配置哈希不一致时 E_CONFIG_MISMATCH 拒绝续跑", False, "未拒绝")


def main() -> None:
    case_bad_records()
    case_bad_config("bad_weights_config.json", "权重长度不匹配 -> E_CONFIG_INVALID")
    case_bad_config("bad_bits_config.json", "位宽越界 -> E_CONFIG_INVALID")
    case_threshold_boundary()
    case_config_mismatch()
    out = Path("reports/fault_injection_report.json")
    out.write_text(json.dumps(RESULT, ensure_ascii=False, indent=2), encoding="utf-8")
    failed = [item for item in RESULT if not item["passed"]]
    print(f"\n汇总: {len(RESULT) - len(failed)}/{len(RESULT)} 通过")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
