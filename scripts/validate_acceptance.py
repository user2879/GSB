"""Run the local acceptance matrix and write a machine-readable record."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from txdiag.canonical import digest  # noqa: E402
from txdiag.engine import compare_canonical, execute  # noqa: E402
from txdiag.gui_model import build_gui_model  # noqa: E402
from txdiag.model import RuntimeConfig  # noqa: E402
from txdiag.spec import load_spec  # noqa: E402


def run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=str(ROOT), text=True, capture_output=True,
                          timeout=60, **kwargs)


def cli(*args: str) -> subprocess.CompletedProcess[str]:
    return run([sys.executable, "-m", "txdiag", *args])


def check(condition: bool, item: str, detail: Any, results: list[dict[str, Any]]) -> None:
    results.append({"check": item, "passed": bool(condition), "detail": detail})
    if not condition:
        raise AssertionError(f"acceptance check failed: {item}: {detail}")


def main() -> int:
    artifacts = ROOT / "artifacts"
    acceptance = artifacts / "acceptance"
    acceptance.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "python": sys.version,
        "platform": sys.platform,
        "checks": results,
    }

    compile_proc = run([sys.executable, "-m", "compileall", "-q", "txdiag", "tests", "scripts"])
    check(compile_proc.returncode == 0, "python_compileall",
          {"returncode": compile_proc.returncode, "stderr": compile_proc.stderr}, results)

    unit = run([sys.executable, "-m", "unittest", "discover", "-v"])
    (artifacts / "unittest.log").write_text(unit.stdout + unit.stderr, encoding="utf-8")
    check(unit.returncode == 0, "unittest_suite",
          {"returncode": unit.returncode, "tail": (unit.stdout + unit.stderr)[-800:]}, results)

    sample_runs: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        for name in ("all_isolation", "deadlock", "wait", "large_range"):
            root = tmp / name
            root.mkdir()
            trace = root / "trace.json"
            report = root / "report.json"
            proc = cli(
                "run", "--spec", str(ROOT / "samples" / f"{name}.json"),
                "--max-running", "4", "--path", "both",
                "--ledger", str(root / "ledger.sqlite"),
                "--snapshot", str(root / "session.snapshot.json"),
                "--trace-out", str(trace), "--report", str(report), "--quiet",
            )
            check(proc.returncode == 0, f"{name}:both_exit",
                  {"returncode": proc.returncode, "stderr": proc.stderr}, results)
            row_trace = root / "trace.row.json"
            range_trace = root / "trace.range.json"
            check(row_trace.read_bytes() == range_trace.read_bytes(),
                  f"{name}:byte_identical_paths",
                  {"bytes": len(row_trace.read_bytes()), "sha256": digest(json.loads(row_trace.read_text()))},
                  results)
            data = json.loads(report.read_text(encoding="utf-8"))
            check(data["canonical_bytes_equal"], f"{name}:canonical_compare",
                  data["comparison"], results)
            sample_runs[name] = {
                "report": data,
                "row_trace_path": str(row_trace),
                "range_trace_path": str(range_trace),
            }

        # Persist representative trace bytes without the temporary databases.
        for name in ("large_range", "deadlock"):
            for path_key in ("row_trace_path", "range_trace_path"):
                source = Path(sample_runs[name][path_key])
                target = acceptance / f"{name}.{path_key.split('_')[0]}.trace.json"
                shutil.copyfile(source, target)

        spec = load_spec(ROOT / "samples" / "all_isolation.json")
        cfg = RuntimeConfig(max_running=8, wait_timeout_ms=1000, wait_budget_ms=5000)
        isolation_result = execute(spec, cfg, "row")
        by_tx = {item["txid"]: item for item in isolation_result.isolation_conclusions}
        rc_values = [item["value"] for item in by_tx["RC1"]["reads"][:2]]
        rr_values = [item["value"] for item in by_tx["RR1"]["reads"][:2]]
        ru_read = by_tx["RU1"]["reads"][0]
        ser_reads = by_tx["SER1"]["reads"]
        check(rc_values == [0, 10], "isolation:read_committed_visibility", rc_values, results)
        check(rr_values == [0, 0], "isolation:repeatable_read_fixed_version", rr_values, results)
        check(ru_read["version_status"] == "uncommitted",
              "isolation:dirty_read", ru_read, results)
        check(any(item["lock_mode"] == "S+X" for item in ser_reads),
              "isolation:serializable_s_plus_x", ser_reads, results)

        wait_spec = load_spec(ROOT / "samples" / "wait.json")
        wait_result = execute(wait_spec, RuntimeConfig(
            max_running=4, wait_timeout_ms=1000, wait_budget_ms=5000), "row")
        granted_waits = [item for observed in wait_result.observed_rounds
                         for item in observed["wait_observations"]
                         if item.get("event") == "granted"]
        check(bool(granted_waits), "locks:real_wait_recorded", granted_waits, results)
        check(all(item["observed_wait_ms"] > 0 for item in granted_waits),
              "locks:monotonic_elapsed_positive",
              [item["observed_wait_ms"] for item in granted_waits], results)

        deadlock_spec = load_spec(ROOT / "samples" / "deadlock.json")
        deadlock_cfg = RuntimeConfig(max_running=4, wait_timeout_ms=1000,
                                     wait_budget_ms=5000)
        signatures = []
        for _ in range(2):
            value = execute(deadlock_spec, deadlock_cfg, "row")
            signatures.append((value.deadlocks[0]["cycle"], value.deadlocks[0]["victim"]))
        check(signatures[0] == signatures[1] == (["TX1", "TX2"], "TX2"),
              "deadlock:stable_victim_and_cycle", signatures, results)

        # Crash boundary, unknown transaction marking, resume, and replay.
        crash_root = tmp / "crash"
        crash_root.mkdir()
        crash_ledger = crash_root / "ledger.sqlite"
        crash_snapshot = crash_root / "session.snapshot.json"
        crashed = cli(
            "run", "--spec", str(ROOT / "samples" / "wait.json"),
            "--ledger", str(crash_ledger), "--snapshot", str(crash_snapshot),
            "--inject", "crash-in-round=2", "--quiet",
        )
        check(crashed.returncode == 70, "recovery:forced_termination", crashed.returncode, results)
        recover_report_path = crash_root / "recover.json"
        recovered = cli(
            "recover", "--spec", str(ROOT / "samples" / "wait.json"),
            "--ledger", str(crash_ledger), "--snapshot", str(crash_snapshot),
            "--report", str(recover_report_path), "--quiet",
        )
        recover_report = json.loads(recover_report_path.read_text(encoding="utf-8"))
        check(recovered.returncode == 6, "recovery:incomplete_exit", recovered.returncode, results)
        check(recover_report["session"]["unknown_txids"] == ["HOLDER", "WAITER"],
              "recovery:unknown_transactions", recover_report["session"], results)
        check(not recover_report["exit_report_available"],
              "recovery:no_forced_exit_report", recover_report["exit_report_available"], results)
        resumed = cli(
            "run", "--spec", str(ROOT / "samples" / "wait.json"),
            "--ledger", str(crash_ledger), "--snapshot", str(crash_snapshot),
            "--report", str(crash_root / "resume.json"), "--quiet",
        )
        check(resumed.returncode == 0, "recovery:resume", resumed.returncode, results)
        replay_report_path = crash_root / "replay.json"
        replayed = cli(
            "replay", "--spec", str(ROOT / "samples" / "wait.json"),
            "--ledger", str(crash_ledger), "--snapshot", str(crash_snapshot),
            "--report", str(replay_report_path), "--quiet",
        )
        replay_report = json.loads(replay_report_path.read_text(encoding="utf-8"))
        check(replayed.returncode == 0 and replay_report["replay"]["prefix_equal"],
              "recovery:replay_equal", replay_report.get("replay"), results)

        # An explicit divergence switch must be caught with a stable first location.
        divergence_root = tmp / "divergence"
        divergence_root.mkdir()
        divergence_report_path = divergence_root / "report.json"
        divergence = cli(
            "run", "--spec", str(ROOT / "samples" / "wait.json"), "--path", "both",
            "--inject", "diverge",
            "--ledger", str(divergence_root / "ledger.sqlite"),
            "--snapshot", str(divergence_root / "session.snapshot.json"),
            "--report", str(divergence_report_path), "--quiet",
        )
        divergence_report = json.loads(divergence_report_path.read_text(encoding="utf-8"))
        check(divergence.returncode == 5, "divergence:exit_code", divergence.returncode, results)
        check((divergence_report["error"]["txid"], divergence_report["error"]["row"],
               divergence_report["error"]["field"]) == ("HOLDER", "lock:A", "event"),
              "divergence:first_location", divergence_report["error"], results)

        gui_spec = load_spec(ROOT / "samples" / "deadlock.json")
        gui_result = execute(gui_spec, deadlock_cfg, "row")
        model_a = build_gui_model(gui_spec, gui_result)
        model_b = build_gui_model(gui_spec, execute(gui_spec, deadlock_cfg, "row"))
        check(model_a == model_b, "gui:stable_model",
              {"transactions": len(model_a["transactions"]),
               "events": len(model_a["events"])}, results)

    try:
        import tkinter
        tk = tkinter.Tk()
        tk.update_idletasks()
        tk.destroy()
        tk_status = {"available": True, "version": tkinter.TkVersion}
    except Exception as exc:
        tk_status = {"available": False, "error": repr(exc)}
    summary["tkinter"] = tk_status
    summary["status"] = "PASS"
    summary["checks_passed"] = len(results)
    summary["checks_failed"] = 0
    summary["artifacts"] = {
        "unittest_log": "artifacts/unittest.log",
        "acceptance_directory": "artifacts/acceptance",
    }
    summary["note"] = (
        "Tk rendering is not executable in this Linux container when libtk is absent; "
        "the GUI module is compiled and its deterministic view model is verified. "
        "Use the documented Windows command for interactive launch."
    )
    (artifacts / "validation.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASS", "checks": len(results), "tkinter": tk_status},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

