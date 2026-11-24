from __future__ import annotations

import copy
import json
import platform
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any, Callable

from .concurrency import ConcurrencyMonitor, parallel_map_ordered
from .config import load_and_validate_config
from .config import LoadedConfig
from .engine import run_pipeline
from .faults import FaultInjector
from .generator import write_example_bundle, write_injection_bundle
from .hashing import fingerprint
from .precision import classify_disagreements
from .samples import load_and_validate_samples
from .snapshot import SnapshotStore, atomic_write_json
from .stats import drift_decision


ROOT = Path.cwd()
ACCEPTANCE_DIR = ROOT / "artifacts" / "acceptance"
EXAMPLE_DIR = ROOT / "examples" / "local_sample"
INJECTION_DIR = ROOT / "examples" / "injections"


def semantic_result(result: dict[str, Any]) -> dict[str, Any]:
    report = copy.deepcopy(result["report"])
    report.pop("snapshot", None)
    report.pop("concurrency", None)
    report.pop("cache_actions", None)
    report.pop("invalidation_list", None)
    report.pop("reuse_list", None)
    for row in report.get("ledger", []):
        row.pop("status", None)
    return {"report": report, "products": copy.deepcopy(result["products"])}


def semantic_fingerprint(result: dict[str, Any]) -> str:
    return fingerprint(semantic_result(result))


def load_bundle(config_path: Path, samples_path: Path):
    config, config_errors = load_and_validate_config(config_path)
    if config is None:
        raise AssertionError([e.to_dict() for e in config_errors])
    samples, sample_errors, sample_fp = load_and_validate_samples(
        samples_path, len(config.channels)
    )
    if sample_errors:
        raise AssertionError([e.to_dict() for e in sample_errors])
    return config, samples


def run_bundle(
    config_path: Path,
    samples_path: Path,
    out_dir: Path | None,
    **kwargs: Any,
) -> dict[str, Any]:
    config, samples = load_bundle(config_path, samples_path)
    return run_pipeline(config, samples, out_dir, **kwargs)


def assert_semantic_equal(actual: dict[str, Any], expected: dict[str, Any], label: str) -> None:
    left = semantic_result(actual)
    right = semantic_result(expected)
    if left != right:
        raise AssertionError(f"{label}: semantic mismatch: {diff_preview(left, right)}")


def diff_preview(left: Any, right: Any, path: str = "$", limit: int = 12) -> list[str]:
    diffs: list[str] = []

    def walk(a: Any, b: Any, p: str) -> None:
        if len(diffs) >= limit:
            return
        if type(a) is not type(b):
            diffs.append(f"{p}: type {type(a).__name__} != {type(b).__name__}")
            return
        if isinstance(a, dict):
            keys = sorted(set(a) | set(b))
            for key in keys:
                if key not in a or key not in b:
                    diffs.append(f"{p}.{key}: missing side")
                else:
                    walk(a[key], b[key], f"{p}.{key}")
        elif isinstance(a, list):
            if len(a) != len(b):
                diffs.append(f"{p}: list length {len(a)} != {len(b)}")
            for i, (ai, bi) in enumerate(zip(a, b)):
                walk(ai, bi, f"{p}[{i}]")
        elif a != b:
            diffs.append(f"{p}: {a!r} != {b!r}")

    walk(left, right, path)
    return diffs


class Verification:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def check(self, name: str, fn: Callable[[], dict[str, Any] | None]) -> None:
        try:
            detail = fn() or {}
            self.checks.append({"name": name, "passed": True, **detail})
        except Exception as exc:
            self.checks.append(
                {
                    "name": name,
                    "passed": False,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )

    @property
    def passed(self) -> bool:
        return all(item["passed"] for item in self.checks)


def verify_concurrency_primitive() -> dict[str, Any]:
    monitor = ConcurrencyMonitor(limit=4)
    barrier = __import__("threading").Barrier(4, timeout=5)

    def worker(value: int) -> tuple[int, int]:
        barrier.wait()
        return value, value * value

    items = list(range(20))
    results = parallel_map_ordered(items, worker, 4, monitor)
    assert [value for value, _ in results] == items
    assert monitor.max_observed == 4, monitor.max_observed
    return {"max_observed_workers": monitor.max_observed, "task_count": len(items)}


def verify_serial_concurrent() -> dict[str, Any]:
    concurrent = run_bundle(
        EXAMPLE_DIR / "pipeline_config.json",
        EXAMPLE_DIR / "samples.jsonl",
        ACCEPTANCE_DIR / "concurrent",
        max_workers=4,
    )
    serial = run_bundle(
        EXAMPLE_DIR / "pipeline_config.json",
        EXAMPLE_DIR / "samples.jsonl",
        ACCEPTANCE_DIR / "serial",
        max_workers=1,
    )
    assert_semantic_equal(concurrent, serial, "serial vs concurrent")
    drift = concurrent["report"]["drift"]
    return {
        "semantic_fingerprint": semantic_fingerprint(concurrent),
        "feature_count": len(drift),
        "drift_ids": [d["feature_id"] for d in drift],
        "drift_decisions": [d["drift"] for d in drift],
        "max_observed_pipeline_workers": concurrent["report"]["concurrency"]["max_observed_workers"],
    }


def verify_warm_cache_and_snapshot() -> dict[str, Any]:
    out = ACCEPTANCE_DIR / "warm"
    first = run_bundle(
        EXAMPLE_DIR / "pipeline_config.json",
        EXAMPLE_DIR / "samples.jsonl",
        out,
        max_workers=2,
    )
    second = run_bundle(
        EXAMPLE_DIR / "pipeline_config.json",
        EXAMPLE_DIR / "samples.jsonl",
        out,
        max_workers=2,
    )
    assert_semantic_equal(first, second, "warm rerun")
    main_actions = [a for a in second["report"]["cache_actions"] if a["stage"] != "source"]
    assert all(a["status"] == "REUSED" for a in second["report"]["cache_actions"]), second["report"]["cache_actions"]
    store = SnapshotStore(out)
    config, _ = load_bundle(EXAMPLE_DIR / "pipeline_config.json", EXAMPLE_DIR / "samples.jsonl")
    recovery = store.recovery_decision(config.fingerprint)
    assert recovery["decision"] == "resume", recovery
    assert recovery["manifest"]["global_version"] == 2
    return {
        "cache_actions": second["report"]["cache_actions"],
        "recovery_decision": recovery["decision"],
        "global_version": recovery["manifest"]["global_version"],
        "ledger_stages": [row["stage"] for row in second["report"]["ledger"]],
    }


def verify_edit_sequence() -> dict[str, Any]:
    work_dir = ACCEPTANCE_DIR / "edit_workspace"
    cold_root = ACCEPTANCE_DIR / "edit_cold"
    work_dir.mkdir(parents=True, exist_ok=True)
    config_path = work_dir / "config.json"
    samples_path = work_dir / "samples.jsonl"
    base_config = json.loads((EXAMPLE_DIR / "pipeline_config.json").read_text(encoding="utf-8"))
    base_samples = (EXAMPLE_DIR / "samples.jsonl").read_text(encoding="utf-8").splitlines()
    atomic_write_json(config_path, base_config)
    samples_path.write_text("\n".join(base_samples) + "\n", encoding="utf-8", newline="\n")

    steps: list[dict[str, Any]] = []
    expected_patterns = [
        ("baseline", {"invalid": 6, "reused": 0}),
        ("append_batch", {"invalid": 6, "reused": 0}),
        ("correct_sample", {"invalid": 6, "reused": 0}),
        ("clip_parameter", {"invalid": ["clip"], "reused_contains": ["source", "standardize", "bucket", "log_stage", "interaction"]}),
        ("stage_reference", {"invalid": ["interaction", "clip"], "reused_contains": ["standardize", "bucket", "log_stage"]}),
        ("gear_switch", {"invalid": 6, "reused": 0}),
        (
            "window_change",
            {
                "invalid": ["source", "standardize", "bucket"],
                "reused_contains": ["log_stage", "interaction", "clip"],
            },
        ),
    ]

    incremental_out = ACCEPTANCE_DIR / "edit_incremental"
    for step_index, (step_name, expectation) in enumerate(expected_patterns):
        if step_name == "append_batch":
            rows = [json.loads(line) for line in samples_path.read_text(encoding="utf-8").splitlines()]
            for offset in range(8):
                seq = 65 + offset
                rows.append(
                    {
                        "seq": seq,
                        "batch": 8,
                        "values": [
                            round(0.31 * __import__("math").sin(seq * 0.61), 9),
                            round(0.27 * __import__("math").sin(seq * 0.77 + 0.8), 9),
                            round(0.23 * __import__("math").sin(seq * 0.93 + 1.6), 9),
                        ],
                        "missing": [False, False, False],
                    }
                )
            samples_path.write_text(
                "\n".join(json.dumps(row, separators=(",", ":")) for row in rows) + "\n",
                encoding="utf-8",
                newline="\n",
            )
        elif step_name == "correct_sample":
            rows = [json.loads(line) for line in samples_path.read_text(encoding="utf-8").splitlines()]
            for row in rows:
                if row["seq"] == 10:
                    row["values"][0] = 0.912345678
                    row["missing"][0] = False
            samples_path.write_text(
                "\n".join(json.dumps(row, separators=(",", ":")) for row in rows) + "\n",
                encoding="utf-8",
                newline="\n",
            )
        elif step_name == "clip_parameter":
            raw = json.loads(config_path.read_text(encoding="utf-8"))
            scale = 10**12
            for stage in raw["stages"]:
                if stage["name"] == "clip":
                    stage["params"]["max_q"] = 2 * scale
            atomic_write_json(config_path, raw)
        elif step_name == "stage_reference":
            raw = json.loads(config_path.read_text(encoding="utf-8"))
            for stage in raw["stages"]:
                if stage["name"] == "interaction":
                    stage["upstream"] = "bucket"
            atomic_write_json(config_path, raw)
        elif step_name == "gear_switch":
            raw = json.loads(config_path.read_text(encoding="utf-8"))
            raw["fixed_point"]["quantization_gear"] = "Q9"
            atomic_write_json(config_path, raw)
        elif step_name == "window_change":
            raw = json.loads(config_path.read_text(encoding="utf-8"))
            raw["windows"]["observation"]["end_seq"] = 56
            atomic_write_json(config_path, raw)

        incremental = run_bundle(config_path, samples_path, incremental_out, max_workers=3)
        cold_out = cold_root / f"step_{step_index:02d}_{step_name}"
        cold = run_bundle(config_path, samples_path, cold_out, max_workers=1)
        assert_semantic_equal(incremental, cold, f"edit step {step_name}")

        report = incremental["report"]
        invalid = report["invalidation_list"]
        reused = report["reuse_list"]
        if "invalid" in expectation and isinstance(expectation["invalid"], int):
            assert len(invalid) == expectation["invalid"], (step_name, invalid)
        elif "invalid" in expectation:
            assert invalid == expectation["invalid"], (step_name, invalid)
        if expectation.get("reused") == 0:
            assert reused == [], (step_name, reused)
        for stage in expectation.get("reused_contains", []):
            assert stage in reused, (step_name, stage, reused)
        ledger_stages = [row["stage"] for row in report["ledger"]]
        assert ledger_stages == ["source", "standardize", "bucket", "log_stage", "interaction", "clip"]
        steps.append(
            {
                "step": step_name,
                "semantic_match": True,
                "invalid": invalid,
                "reused": reused,
                "ledger_order": ledger_stages,
                "fingerprint": semantic_fingerprint(incremental),
            }
        )
    return {"steps": steps}


def verify_budget() -> dict[str, Any]:
    out = ACCEPTANCE_DIR / "budget"
    low = run_bundle(
        EXAMPLE_DIR / "pipeline_config.json",
        EXAMPLE_DIR / "samples.jsonl",
        out,
        budget=39,
        max_workers=1,
    )
    report = low["report"]
    assert report["status"] == "budget_pending", report["status"]
    assert report["summary"]["pending_count"] > 0
    assert any(f["code"] == "E_POINT_BUDGET_PENDING" for f in report["failures"])
    store = SnapshotStore(out)
    pending = store.load_pending()
    assert len(pending) == report["summary"]["pending_count"]
    assert pending == sorted(pending, key=lambda item: tuple(item["sort_key"]))

    low_again = run_bundle(
        EXAMPLE_DIR / "pipeline_config.json",
        EXAMPLE_DIR / "samples.jsonl",
        out,
        budget=39,
        max_workers=1,
    )
    assert low_again["report"]["pending"] == report["pending"]

    complete = run_bundle(
        EXAMPLE_DIR / "pipeline_config.json",
        EXAMPLE_DIR / "samples.jsonl",
        ACCEPTANCE_DIR / "budget_complete",
        max_workers=2,
    )
    normal = run_bundle(
        EXAMPLE_DIR / "pipeline_config.json",
        EXAMPLE_DIR / "samples.jsonl",
        ACCEPTANCE_DIR / "budget_normal_cold",
        max_workers=1,
    )
    assert_semantic_equal(complete, normal, "budget completion")
    return {
        "pending_count": len(pending),
        "first_pending": pending[0],
        "pending_snapshot": str(store.root / "CURRENT.json"),
        "completion_fingerprint": semantic_fingerprint(complete),
    }


def verify_injections() -> dict[str, Any]:
    manifest = json.loads((INJECTION_DIR / "injection_manifest.json").read_text(encoding="utf-8"))
    results: list[dict[str, Any]] = []
    for case in manifest:
        if case["kind"] in {"config", "sample"}:
            config, config_errors = load_and_validate_config(
                case["config"], case["injection_id"]
            )
            failures = list(config_errors)
            if config is not None and case["kind"] == "sample":
                _, sample_errors, _ = load_and_validate_samples(
                    case["samples"], len(config.channels), case["injection_id"]
                )
                failures.extend(sample_errors)
            assert failures, case["injection_id"]
            assert any(f.code == case["expected_code"] for f in failures), [f.to_dict() for f in failures]
            assert any(f.injection_id == case["injection_id"] for f in failures)
            results.append(
                {
                    "injection_id": case["injection_id"],
                    "hit": True,
                    "code": case["expected_code"],
                    "path": failures[0].path,
                }
            )
        elif case["kind"] == "runtime":
            payload = json.loads(Path(case["runtime_config"]).read_text(encoding="utf-8"))
            injector = FaultInjector.from_dict(payload)
            result = run_bundle(
                Path(case["config"]),
                Path(case["samples"]),
                ACCEPTANCE_DIR / "runtime_faults",
                fault_injector=injector,
                include_attribution=False,
                use_cache=False,
            )
            report = result["report"]
            codes = {f["code"] for f in report["failures"]}
            assert set(case["expected_codes"]).issubset(codes), report["failures"]
            assert set(payload["injection_id"] and injector.injection_ids).issubset(
                set(report["injection_hits"])
            ) or len(report["injection_hits"]) >= 2
            assert report["snapshot"]["manifest"]["completed_batch_cursor"] == 5
            clip_rows = len(result["products"]["clip"]["rows"])
            assert clip_rows == 56, clip_rows
            results.append(
                {
                    "injection_id": case["injection_id"],
                    "hit": True,
                    "codes": sorted(codes),
                    "injection_hits": report["injection_hits"],
                    "completed_batch_cursor": report["snapshot"]["manifest"]["completed_batch_cursor"],
                    "clip_row_count_after_isolated_batch_failure": clip_rows,
                }
            )
    return {"cases": results}


def verify_attribution_present() -> dict[str, Any]:
    result = run_bundle(
        EXAMPLE_DIR / "pipeline_config.json",
        EXAMPLE_DIR / "samples.jsonl",
        ACCEPTANCE_DIR / "attribution",
        max_workers=1,
    )
    attribution = result["report"]["attribution"]
    assert attribution is not None
    assert attribution["high_precision_gear"] == "Q9"
    assert attribution["stage_rows"]
    stages = sorted({row["stage"] for row in attribution["stage_rows"]})
    assert stages == ["bucket", "clip", "interaction", "log_stage", "source", "standardize"]
    return {
        "configured_gear": attribution["configured_gear"],
        "high_precision_gear": attribution["high_precision_gear"],
        "first_exceeded_stage": attribution["first_exceeded_stage"],
        "row_count": len(attribution["stage_rows"]),
        "disagreement_count": attribution["disagreement_count"],
    }


def verify_boundary_and_flip_rules() -> dict[str, Any]:
    stat = lambda mean: {
        "feature_id": "clip:x",
        "feature_name": "x",
        "scale": 10**12,
        "mean_q": mean,
        "sample_set": [1, 2],
        "count": 2,
        "missing_or_imputed_count": 0,
    }
    exact = drift_decision(stat(10_000), stat(10_100), 100, "Q6", 1)
    half = drift_decision(stat(10_000_000), stat(10_001_400), 1, "Q6", 1)
    assert exact["boundary_reason"] == "exact_threshold_equality"
    assert half["boundary_reason"] == "within_half_gear_step"

    raw = json.loads((EXAMPLE_DIR / "pipeline_config.json").read_text(encoding="utf-8"))
    config = LoadedConfig(EXAMPLE_DIR / "pipeline_config.json", raw, "rule-fp")
    products = {
        "clip": {"stage": "clip", "features": [{"id": "clip:x", "source_feature": "interaction:x"}]},
        "interaction": {
            "stage": "interaction",
            "type": "interaction",
            "features": [{"id": "interaction:x", "pair": [0, 1]}],
        },
        "log_stage": {"stage": "log_stage", "type": "log", "features": [{"id": "log:a"}]},
        "standardize": {"stage": "standardize", "type": "standardize", "features": [{"id": "std:a"}]},
        "source": {"features": [{"id": "source:a"}]},
    }
    stages = ["source", "standardize", "bucket", "log_stage", "interaction", "clip"]

    def drift_row(value: bool, boundary: bool) -> dict[str, Any]:
        return {
            "feature_id": "clip:x",
            "drift": value,
            "boundary": boundary,
            "threshold_bp": 100,
            "drift_bp_rounded": 90 if not value else 110,
            "exact_compare": {"left": 9, "threshold_right": 10},
            "distance_to_threshold_q": "1000",
            "half_gear_step_q": "500000",
            "boundary_reason": "within_half_gear_step" if boundary else None,
        }

    def diff(stage: str, feature_id: str, value: int, exceeds: bool) -> dict[str, Any]:
        return {
            "stage": stage,
            "feature_id": feature_id,
            "diff_canonical_q": value,
            "exceeds_tolerance": exceeds,
        }

    boundary_flip = classify_disagreements(
        config, products, [drift_row(False, True)], [drift_row(True, True)], [], stages, 10
    )[0]["classification"]
    log_flip = classify_disagreements(
        config,
        products,
        [drift_row(False, False)],
        [drift_row(True, False)],
        [diff("standardize", "std:a", 1, False), diff("log_stage", "log:a", 50, True)],
        stages,
        10,
    )[0]["classification"]
    quant_flip = classify_disagreements(
        config,
        products,
        [drift_row(False, False)],
        [drift_row(True, False)],
        [diff("standardize", "std:a", 50, True)],
        stages,
        10,
    )[0]["classification"]
    assert boundary_flip == "threshold_boundary_decision"
    assert log_flip == "log_domain_approximation"
    assert quant_flip == "quantization_truncation"
    return {
        "exact_equality_boundary": exact["boundary"],
        "half_step_boundary": half["boundary"],
        "flip_classes": [boundary_flip, log_flip, quant_flip],
    }


def main() -> int:
    shutil.rmtree(ACCEPTANCE_DIR, ignore_errors=True)
    ACCEPTANCE_DIR.mkdir(parents=True, exist_ok=True)
    write_example_bundle(EXAMPLE_DIR)
    write_injection_bundle(EXAMPLE_DIR, INJECTION_DIR)
    verification = Verification()
    verification.check("concurrency_primitive", verify_concurrency_primitive)
    verification.check("serial_concurrent_equivalence", verify_serial_concurrent)
    verification.check("warm_cache_snapshot_recovery", verify_warm_cache_and_snapshot)
    verification.check("incremental_edit_sequence", verify_edit_sequence)
    verification.check("point_budget_pending_and_completion", verify_budget)
    verification.check("injection_and_batch_faults", verify_injections)
    verification.check("precision_attribution", verify_attribution_present)
    verification.check("boundary_and_flip_rules", verify_boundary_and_flip_rules)

    report = {
        "schema": "vibpipe-verification/v1",
        "passed": verification.passed,
        "environment": {
            "python": sys.version.split()[0],
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "checks": verification.checks,
    }
    try:
        import numpy

        report["environment"]["numpy"] = numpy.__version__
    except ImportError:
        pass
    try:
        import yaml

        report["environment"]["pyyaml"] = yaml.__version__
    except ImportError:
        pass
    json_path = ROOT / "artifacts" / "verification_report.json"
    atomic_write_json(json_path, report)
    md_path = ROOT / "docs" / "VERIFICATION_RESULTS.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_markdown(report), encoding="utf-8", newline="\n")
    print(json.dumps({"passed": report["passed"], "report": str(json_path)}, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


def render_markdown(report: dict[str, Any]) -> str:
    env = report.get("environment", {})
    lines = [
        "# 实际验证结果",
        "",
        f"总体结果：{'通过' if report['passed'] else '失败'}",
        "",
        "## 环境",
        "",
        f"- Python：`{env.get('python')}`",
        f"- 平台：`{env.get('platform')}`",
        f"- numpy：`{env.get('numpy')}`",
        f"- PyYAML：`{env.get('pyyaml')}`",
        "",
        "## 检查项",
        "",
    ]
    for check in report["checks"]:
        mark = "PASS" if check["passed"] else "FAIL"
        lines.append(f"- [{mark}] `{check['name']}`")
        if not check["passed"]:
            lines.append(f"  - 错误：{check.get('error', '')}")
    lines.append("")
    lines.extend([
        "机器可读完整报告：`artifacts/verification_report.json`。",
        "",
        "复跑命令：`python -m unittest discover -s tests` 与 `python -m vibpipe.verify`。",
        "",
    ])
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
