from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pagecache.fsutil import remove_tree


ROOT = Path(__file__).resolve().parent.parent
ACCEPTANCE = ROOT / "artifacts" / "acceptance"
RUNS = ACCEPTANCE / "runs"
LOG_PATH = ACCEPTANCE / "structured.log"
PYTHON = sys.executable


class AcceptanceFailure(RuntimeError):
    pass


def main() -> int:
    checks: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    try:
        _reset()
        _check(checks, "environment", _environment)
        meta = _check(checks, "sample_generation", _generate_samples)
        details["sample"] = {
            "seed": meta["seed"],
            "page_count": meta["page_count"],
            "row_count": meta["row_count"],
            "scan_order": meta["scan_order"],
            "page_sha256": meta["page_sha256"],
        }
        details["format_read"] = _check(checks, "multi_version_read", _read_all_formats)
        upgrade = _check(checks, "upgrade_v1_and_v2", _upgrade_samples)
        details["upgrade"] = upgrade
        cache = _check(checks, "cache_and_aggregation", lambda: _run_cache_checks(meta))
        details["cache_and_aggregation"] = cache
        snapshot = _check(checks, "snapshot_replay_and_verification", lambda: _run_snapshot_checks(meta))
        details["snapshot"] = snapshot
        injections = _check(checks, "three_anomaly_injections", lambda: _run_injections(meta))
        details["injections"] = injections
        tests = _check(checks, "unit_test_suite", _run_unit_tests)
        details["unit_tests"] = tests
        status = "PASS"
    except Exception as exc:
        status = "FAIL"
        details["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        if not checks or checks[-1]["status"] != "FAIL":
            checks.append({"name": "acceptance_execution", "status": "FAIL", "details": details["failure"]})

    report = {
        "status": status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "checks": checks,
        "details": details,
    }
    ACCEPTANCE.mkdir(parents=True, exist_ok=True)
    (ACCEPTANCE / "acceptance-report.json").write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (ACCEPTANCE / "acceptance-summary.md").write_text(_summary_markdown(report), encoding="utf-8")
    print(json.dumps({"status": status, "report": str(ACCEPTANCE / "acceptance-report.json")}, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if status == "PASS" else 1


def _check(checks: list[dict[str, Any]], name: str, function: Any) -> Any:
    try:
        value = function()
        checks.append({"name": name, "status": "PASS", "details": value})
        return value
    except Exception as exc:
        checks.append({"name": name, "status": "FAIL", "details": {"type": type(exc).__name__, "message": str(exc)}})
        raise


def _reset() -> None:
    remove_tree(ACCEPTANCE)
    RUNS.mkdir(parents=True)


def _environment() -> dict[str, Any]:
    if sys.version_info < (3, 11):
        raise AcceptanceFailure("Python 3.11 or newer is required")
    if np.__version__ != "1.26.4":
        raise AcceptanceFailure(f"numpy 1.26.4 is required, found {np.__version__}")
    return {"python": sys.version, "numpy": np.__version__, "platform": platform.platform()}


def _generate_samples() -> dict[str, Any]:
    result = _run_cli("generate", "--out", str(ROOT / "samples" / "data"), "--seed", "20261118")
    return result


def _read_all_formats() -> dict[str, Any]:
    sys.path.insert(0, str(ROOT))
    from pagecache.manifest import read_manifest
    from pagecache.segment import read_segment_file

    result: dict[str, Any] = {}
    for version in (1, 2, 3):
        base = ROOT / "samples" / "data" / "versions" / f"v{version}"
        manifest = read_manifest(base / "manifest.pcm")
        segment = read_segment_file(base / "overflow" / "p0001.seg")
        result[f"v{version}"] = {
            "header_version": manifest.header_version,
            "record_versions": list(manifest.format_versions),
            "segments_read": manifest.page_count,
            "sample_segment_version": segment.version,
            "sample_segment_compressed": segment.compressed,
        }
    return result


def _upgrade_samples() -> dict[str, Any]:
    _copy_version(1, RUNS / "upgrade-v1")
    first = _run_cli(
        "upgrade",
        "--manifest", str(RUNS / "upgrade-v1" / "manifest.pcm"),
        "--pages-dir", str(ROOT / "samples" / "data" / "pages"),
        "--overflow-dir", str(RUNS / "upgrade-v1" / "overflow"),
    )
    _copy_version(2, RUNS / "upgrade-v2")
    second = _run_cli(
        "upgrade",
        "--manifest", str(RUNS / "upgrade-v2" / "manifest.pcm"),
        "--pages-dir", str(ROOT / "samples" / "data" / "pages"),
        "--overflow-dir", str(RUNS / "upgrade-v2" / "overflow"),
    )

    _copy_version(1, RUNS / "upgrade-resume")
    _copy_version(1, RUNS / "upgrade-one-shot")
    interrupted = _run_cli(
        "upgrade",
        "--manifest", str(RUNS / "upgrade-resume" / "manifest.pcm"),
        "--pages-dir", str(ROOT / "samples" / "data" / "pages"),
        "--overflow-dir", str(RUNS / "upgrade-resume" / "overflow"),
        "--crash-after-checkpoint", "2",
        expected_returncode=75,
    )
    resumed = _run_cli(
        "upgrade",
        "--manifest", str(RUNS / "upgrade-resume" / "manifest.pcm"),
        "--pages-dir", str(ROOT / "samples" / "data" / "pages"),
        "--overflow-dir", str(RUNS / "upgrade-resume" / "overflow"),
    )
    one_shot = _run_cli(
        "upgrade",
        "--manifest", str(RUNS / "upgrade-one-shot" / "manifest.pcm"),
        "--pages-dir", str(ROOT / "samples" / "data" / "pages"),
        "--overflow-dir", str(RUNS / "upgrade-one-shot" / "overflow"),
    )
    resume_bytes = (RUNS / "upgrade-resume" / "manifest.pcm").read_bytes()
    one_shot_bytes = (RUNS / "upgrade-one-shot" / "manifest.pcm").read_bytes()
    if resume_bytes != one_shot_bytes:
        raise AcceptanceFailure("resumed upgrade manifest differs byte-for-byte from one-shot upgrade")
    checkpoint = _read_json(RUNS / "upgrade-resume" / "manifest.pcm.upgrade-checkpoint.json")
    backup_stat = (RUNS / "upgrade-resume" / "manifest.pcm.pre-upgrade.backup").stat()
    return {
        "v1": {"status": first["status"], "filled_fields": first["filled_fields"]},
        "v2": {"status": second["status"], "filled_fields": second["filled_fields"]},
        "interrupted_after_checkpoint": interrupted["completed_records"],
        "resumed": resumed["resumed"],
        "processed_record_indices": resumed["processed_record_indices"],
        "manifest_sha256": hashlib.sha256(resume_bytes).hexdigest(),
        "resumed_equals_one_shot": resume_bytes == one_shot_bytes,
        "checkpoint_stage": checkpoint["stage"],
        "backup_read_only": (backup_stat.st_mode & 0o777) == 0o444,
    }


def _run_cache_checks(meta: dict[str, Any]) -> dict[str, Any]:
    full_dir = RUNS / "cache-full"
    small_dir = RUNS / "cache-small"
    repeat_dir = RUNS / "cache-small-repeat"
    for directory in (full_dir, small_dir, repeat_dir):
        directory.mkdir(parents=True, exist_ok=True)
        _copy_segments(ROOT / "samples" / "data" / "versions" / "v3" / "overflow", directory)
    common = [
        "--manifest", str(ROOT / "samples" / "data" / "versions" / "v3" / "manifest.pcm"),
        "--pages-dir", str(ROOT / "samples" / "data" / "pages"),
        "--page-size", str(meta["page_size"]),
    ]
    full = _run_cli(
        "run", *common,
        "--overflow-dir", str(full_dir),
        "--budget-bytes", str(meta["full_budget_bytes"]),
        "--output", str(full_dir / "output.json"),
        "--events", str(full_dir / "events.json"),
    )
    small = _run_cli(
        "run", *common,
        "--overflow-dir", str(small_dir),
        "--budget-bytes", str(meta["small_budget_bytes"]),
        "--output", str(small_dir / "output.json"),
        "--events", str(small_dir / "events.json"),
    )
    repeat = _run_cli(
        "run", *common,
        "--overflow-dir", str(repeat_dir),
        "--budget-bytes", str(meta["small_budget_bytes"]),
        "--output", str(repeat_dir / "output.json"),
        "--events", str(repeat_dir / "events.json"),
    )
    full_output = (full_dir / "output.json").read_bytes()
    small_output = (small_dir / "output.json").read_bytes()
    repeat_output = (repeat_dir / "output.json").read_bytes()
    if full_output != small_output:
        raise AcceptanceFailure("full-resident and small-budget aggregation outputs differ")
    if small_output != repeat_output:
        raise AcceptanceFailure("repeated deterministic small-budget output differs")
    if small["overflow_sequence"] != repeat["overflow_sequence"] or small["swap_in_sequence"] != repeat["swap_in_sequence"]:
        raise AcceptanceFailure("repeated deterministic cache sequences differ")
    if not small["overflow_sequence"]:
        raise AcceptanceFailure("small-budget run did not spill any page")
    return {
        "full_capacity_frames": full["capacity_frames"],
        "small_capacity_frames": small["capacity_frames"],
        "full_overflow_sequence": full["overflow_sequence"],
        "small_overflow_sequence": small["overflow_sequence"],
        "small_swap_in_sequence": small["swap_in_sequence"],
        "small_swap_in_count": small["swap_in_count"],
        "repeat_sequences_equal": True,
        "aggregation_output_sha256": hashlib.sha256(small_output).hexdigest(),
        "full_and_small_byte_identical": True,
        "small_and_repeat_byte_identical": True,
        "fixed_point_mean_error_bound": "3/2000000 (0.0000015 absolute, per group mean)",
    }


def _run_snapshot_checks(meta: dict[str, Any]) -> dict[str, Any]:
    run_dir = RUNS / "snapshot"
    run_dir.mkdir(parents=True)
    overflow = run_dir / "overflow"
    overflow.mkdir()
    _copy_segments(ROOT / "samples" / "data" / "versions" / "v3" / "overflow", overflow)
    common = [
        "--manifest", str(ROOT / "samples" / "data" / "versions" / "v3" / "manifest.pcm"),
        "--pages-dir", str(ROOT / "samples" / "data" / "pages"),
        "--overflow-dir", str(overflow),
        "--budget-bytes", str(meta["small_budget_bytes"]),
        "--page-size", str(meta["page_size"]),
    ]
    run = _run_cli(
        "run", *common,
        "--output", str(run_dir / "output.json"),
        "--events", str(run_dir / "events.json"),
        "--snapshot", str(run_dir / "snapshot.json"),
        "--snapshot-at", "5",
    )
    replay = _run_cli("replay", *common, "--snapshot", str(run_dir / "snapshot.json"))
    verify = _run_cli(
        "verify",
        "--snapshot", str(run_dir / "snapshot.json"),
        "--output", str(run_dir / "output.json"),
        "--manifest", str(ROOT / "samples" / "data" / "versions" / "v3" / "manifest.pcm"),
        "--pages-dir", str(ROOT / "samples" / "data" / "pages"),
        "--overflow-dir", str(overflow),
        "--report", str(run_dir / "verify-report.json"),
    )
    if replay["status"] != "reproducible":
        raise AcceptanceFailure("snapshot replay was not reproducible")
    if verify["status"] != "PASS":
        raise AcceptanceFailure("independent verifier did not pass")
    return {
        "snapshot_position": run["snapshot_at"],
        "snapshot_matches_reference": run["matches_reference"],
        "replay_status": replay["status"],
        "replay_compared_records": replay["compared_records"],
        "replay_final_output_sha256": replay["final_output_sha256"],
        "verifier_status": verify["status"],
        "verifier_checks": verify["summary"],
        "lost_overflow_segments": replay["disk_diff"]["lost_overflow_segments"],
        "new_overflow_segments": replay["disk_diff"]["new_overflow_segments"],
    }


def _run_injections(meta: dict[str, Any]) -> dict[str, Any]:
    work = RUNS / "injections"
    common = [
        "--manifest", str(ROOT / "samples" / "data" / "versions" / "v3" / "manifest.pcm"),
        "--pages-dir", str(ROOT / "samples" / "data" / "pages"),
        "--overflow-dir", str(ROOT / "samples" / "data" / "versions" / "v3" / "overflow"),
        "--log", str(LOG_PATH),
    ]
    results = {}
    for kind in ("manifest_tail_truncate", "segment_checksum_mismatch", "memory_budget_zero"):
        result = _run_cli("inject", "--kind", kind, *common, "--work-dir", str(work / kind))
        results[kind] = {"expected_code": result["expected_code"], "actual_code": result["actual_code"]}
    expected = {
        "manifest_tail_truncate": "E_FORMAT",
        "segment_checksum_mismatch": "E_INTEGRITY",
        "memory_budget_zero": "E_BUDGET",
    }
    for kind, code in expected.items():
        if results[kind]["actual_code"] != code:
            raise AcceptanceFailure(f"injection {kind} reported {results[kind]['actual_code']} instead of {code}")
    log_rows = _read_json_lines(LOG_PATH)
    if [row.get("code") for row in log_rows[-3:]] != ["E_FORMAT", "E_INTEGRITY", "E_BUDGET"]:
        raise AcceptanceFailure("structured injection log does not contain the three expected codes in order")
    return {"injections": results, "structured_log_rows": len(log_rows), "log_path": str(LOG_PATH)}


def _run_unit_tests() -> dict[str, Any]:
    completed = subprocess.run(
        [PYTHON, "-m", "unittest", "discover", "-v"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    (ACCEPTANCE / "unit-tests.log").write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise AcceptanceFailure(f"unit tests failed with exit code {completed.returncode}")
    tail = [line for line in (completed.stdout + completed.stderr).splitlines() if line.startswith("Ran ") or line == "OK"]
    return {"returncode": completed.returncode, "summary": tail, "log": str(ACCEPTANCE / "unit-tests.log")}


def _run_cli(*arguments: str, expected_returncode: int = 0) -> Any:
    completed = subprocess.run(
        [PYTHON, "-m", "pagecache", *arguments],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != expected_returncode:
        raise AcceptanceFailure(
            f"command failed: pagecache {' '.join(arguments)}; exit={completed.returncode}; stderr={completed.stderr.strip()}; stdout={completed.stdout.strip()}"
        )
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise AcceptanceFailure(f"command did not return JSON: {completed.stdout}") from exc


def _copy_version(version: int, target: Path) -> None:
    source = ROOT / "samples" / "data" / "versions" / f"v{version}"
    remove_tree(target)
    shutil.copytree(source, target)


def _copy_segments(source: Path, target: Path) -> None:
    for path in sorted(source.glob("*.seg"), key=lambda item: item.name):
        shutil.copy2(path, target / path.name)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _summary_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Acceptance Summary",
        "",
        f"- Status: **{report['status']}**",
        f"- Generated (UTC): `{report['generated_at_utc']}`",
        f"- Python: `{report['environment']['python']}`",
        f"- numpy: `{report['environment']['numpy']}`",
        "",
        "| Check | Status |",
        "|---|---|",
    ]
    for check in report["checks"]:
        lines.append(f"| `{check['name']}` | {check['status']} |")
    cache = report.get("details", {}).get("cache_and_aggregation")
    if cache:
        lines.extend(
            [
                "",
                "## Determinism",
                "",
                f"- Full-resident capacity: `{cache['full_capacity_frames']}` frames",
                f"- Small-budget capacity: `{cache['small_capacity_frames']}` frames",
                f"- Small-budget overflow sequence: `{cache['small_overflow_sequence']}`",
                f"- Small-budget swap-in count: `{cache['small_swap_in_count']}`",
                f"- Aggregation SHA-256: `{cache['aggregation_output_sha256']}`",
                f"- Full/small output byte-identical: `{cache['full_and_small_byte_identical']}`",
                f"- Mean absolute error bound: `{cache['fixed_point_mean_error_bound']}`",
            ]
        )
    snapshot = report.get("details", {}).get("snapshot")
    if snapshot:
        lines.extend(
            [
                "",
                "## Snapshot / Replay",
                "",
                f"- Replay status: `{snapshot['replay_status']}`",
                f"- Compared records: `{snapshot['replay_compared_records']}`",
                f"- Verifier: `{snapshot['verifier_status']}` ({snapshot['verifier_checks']['checks_passed']}/{snapshot['verifier_checks']['checks_total']} checks)",
            ]
        )
    injections = report.get("details", {}).get("injections")
    if injections:
        lines.extend(["", "## Injections", "", "| Injection | Expected | Actual |", "|---|---|---|"])
        for name, item in injections["injections"].items():
            lines.append(f"| `{name}` | {item['expected_code']} | {item['actual_code']} |")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
