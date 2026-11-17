"""End-to-end local demo: reproducibility, hard-kill recovery, replay diff.

Run with::

    python examples/demo_run.py

Everything stays on 127.0.0.1 high ports, uses only the standard library and
never touches external services or source code.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chaoskit.engine import ChaosEngine  # noqa: E402
from chaoskit.errors import ChaosError, ReplayError  # noqa: E402
from chaoskit.gui import render_headless  # noqa: E402
from chaoskit.ledger import LEDGER_FILENAME  # noqa: E402
from chaoskit.replay import verify_identity, verify_observations, verify_timeline_entries  # noqa: E402
from chaoskit.spec import load_spec  # noqa: E402
from chaoskit.timeline import build_timeline, timeline_digest  # noqa: E402
from chaoskit.viewmodel import build_view_model  # noqa: E402

SPEC = ROOT / "examples" / "sample_experiment.json"
SPEC_SHUFFLED = ROOT / "examples" / "sample_experiment_shuffled.json"
SPEC_DRIFT = ROOT / "examples" / "sample_experiment_drift.json"
WORK_BASE = ROOT / "artifacts" / "demo"
WORKER = ROOT / "examples" / "demo_worker.py"


def banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def run_worker(spec: Path, workdir: Path, max_injections: int | None = None,
               kill_after: float | None = None) -> dict:
    args = [sys.executable, str(WORKER), str(spec), str(workdir)]
    if max_injections is not None:
        args.append(str(max_injections))
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        if kill_after is not None:
            time.sleep(kill_after)
            proc.kill()
            output, _ = proc.communicate(timeout=10)
            return {"returncode": proc.returncode, "killed": True, "output": output}
        output, _ = proc.communicate(timeout=120)
        return {"returncode": proc.returncode, "killed": False, "output": output}
    finally:
        if proc.poll() is None:
            proc.kill()


def scenario_reproducibility() -> dict:
    banner("Scenario 1: same seed -> byte-identical timeline and comparable observations")
    spec_a = load_spec(str(SPEC))
    spec_b = load_spec(str(SPEC_SHUFFLED))
    digest_a = timeline_digest(build_timeline(spec_a))
    digest_b = timeline_digest(build_timeline(spec_b))
    print(f"timeline digest (original order) : {digest_a}")
    print(f"timeline digest (shuffled order) : {digest_b}")
    assert digest_a == digest_b, "timeline must be independent of declaration order"

    work_a = WORK_BASE / "run_a"
    work_b = WORK_BASE / "run_b"
    for path in (work_a, work_b):
        shutil.rmtree(path, ignore_errors=True)
    result_a = run_worker(SPEC, work_a)
    result_b = run_worker(SPEC, work_b)
    assert result_a["returncode"] == 0 and result_b["returncode"] == 0

    def content_digest(workdir: Path) -> str:
        conn = sqlite3.connect(str(workdir / LEDGER_FILENAME))
        conn.row_factory = sqlite3.Row
        try:
            view = build_view_model(conn)
            exp = conn.execute("SELECT value AS v FROM meta WHERE key='content_digest'").fetchone()
            return {"meta_digest": exp["v"], "view": view}
        finally:
            conn.close()

    info_a = content_digest(work_a)
    info_b = content_digest(work_b)
    print(f"ledger content digest run A: {info_a['meta_digest']}")
    print(f"ledger content digest run B: {info_b['meta_digest']}")

    # Structural observation conclusions must compare equal across runs.
    conclusions_a = _conclusions(work_a)
    conclusions_b = _conclusions(work_b)
    print("active-phase conclusions (run A):")
    print(json.dumps(conclusions_a, indent=2, sort_keys=True))
    assert conclusions_a == conclusions_b, "observation conclusions must reproduce"
    return {"timeline_digest": digest_a, "conclusions": conclusions_a,
            "workdirs": [str(work_a), str(work_b)]}


def _conclusions(workdir: Path) -> list:
    conn = sqlite3.connect(str(workdir / LEDGER_FILENAME))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT injection_idx, summary_json FROM observations WHERE phase='active' ORDER BY injection_idx"
        ).fetchall()
        return [
            {
                "injection_idx": row["injection_idx"],
                **{
                    key: value
                    for key, value in json.loads(row["summary_json"]).items()
                    # Success/fail outcomes are seed-determined and must reproduce
                    # exactly; absolute latency carries scheduling jitter and is
                    # compared only via tolerance-aware replay verification.
                    if key in ("success_rate_bps", "insufficient",
                               "rate_quant_error_bps", "latency_quant_error_ms",
                               "success_count", "failure_count", "sample_count")
                },
            }
            for row in rows
        ]
    finally:
        conn.close()


def scenario_interrupt_recovery() -> dict:
    banner("Scenario 2: forced termination -> ledger recovery, unknown, rerun, replay")
    work = WORK_BASE / "interrupt"
    shutil.rmtree(work, ignore_errors=True)

    killed = run_worker(SPEC, work, kill_after=3.0)
    print(f"worker killed mid-run: rc={killed['returncode']}")
    assert killed["killed"]

    spec = load_spec(str(SPEC))
    engine = ChaosEngine(spec, work)
    engine.setup()
    try:
        report = engine.ledger.recovery_report(spec)
        print("recovery report after hard kill:")
        print(json.dumps(report, indent=2, sort_keys=True))
        assert report["unknown_indexes"], "an in-flight injection must be marked unknown"
        assert report["exit_report_promised"] is False
        unknown_before = list(report["unknown_indexes"])

        continued = engine.run()
        print("continuation report:")
        print(json.dumps({key: continued[key] for key in
                          ("recovered_committed", "unknown_indexes", "remaining_budget")},
                         indent=2, sort_keys=True))
    finally:
        engine.teardown()

    # Replay verification against the recovered ledger.
    conn = sqlite3.connect(str(work / LEDGER_FILENAME))
    conn.row_factory = sqlite3.Row
    try:
        verify_identity(spec, _ledger_like(conn))
    finally:
        conn.close()
    return {"workdir": str(work), "unknown_before_rerun": unknown_before}


def _ledger_like(conn: sqlite3.Connection):
    from chaoskit.ledger import Ledger
    ledger = object.__new__(Ledger)
    ledger.conn = conn
    return ledger


def scenario_replay_diff() -> dict:
    banner("Scenario 3: replay difference localization (E_REPLAY with field and both values)")
    work = WORK_BASE / "replay_drift"
    shutil.rmtree(work, ignore_errors=True)
    full = run_worker(SPEC, work)
    assert full["returncode"] == 0

    drifted_spec = load_spec(str(SPEC_DRIFT))
    conn = sqlite3.connect(str(work / LEDGER_FILENAME))
    conn.row_factory = sqlite3.Row
    try:
        ledger = _ledger_like(conn)
        try:
            verify_identity(drifted_spec, ledger)
            raise AssertionError("E_REPLAY was expected but not raised")
        except ReplayError as exc:
            print("caught expected replay error:")
            print(json.dumps(exc.to_dict(), indent=2, sort_keys=True))
            identity_error = exc.to_dict()

        # Field-level structural mismatch: drift spec changes an injection duration.
        try:
            verify_timeline_entries(drifted_spec, ledger)
            raise AssertionError("E_REPLAY was expected but not raised")
        except ReplayError as exc:
            print("caught expected structural replay error:")
            print(json.dumps(exc.to_dict(), indent=2, sort_keys=True))
            structural_error = exc.to_dict()
    finally:
        conn.close()

    # Whole-snapshot replacement when the same workdir is reopened with drift spec.
    engine = ChaosEngine(drifted_spec, work)
    try:
        open_event = engine.setup()
        print("whole-snapshot replacement event:")
        print(json.dumps(open_event, indent=2, sort_keys=True))
        assert open_event["replaced"] is True
        engine.run()
        report = engine.ledger.recovery_report(drifted_spec)
        assert report["replacement_events"], "replacement must be recorded and listed"
    finally:
        engine.teardown()
    return {"identity_error": identity_error, "structural_error": structural_error}


def scenario_observation_replay() -> dict:
    banner("Scenario 3b: observation-level replay across two independent runs")
    work_a = WORK_BASE / "run_a"
    work_c = WORK_BASE / "run_c"
    shutil.rmtree(work_c, ignore_errors=True)
    result = run_worker(SPEC, work_c)
    assert result["returncode"] == 0

    spec = load_spec(str(SPEC))
    ledger_conn = sqlite3.connect(str(work_a / LEDGER_FILENAME))
    ledger_conn.row_factory = sqlite3.Row
    try:
        ledger = _ledger_like(ledger_conn)
        replay_conn = sqlite3.connect(str(work_c / LEDGER_FILENAME))
        replay_conn.row_factory = sqlite3.Row
        try:
            summaries: dict = {}
            for row in replay_conn.execute("SELECT * FROM observations ORDER BY id"):
                summaries.setdefault(row["injection_idx"], []).append(
                    json.loads(row["summary_json"])
                )
            verify_observations(ledger, summaries, spec.observation.sample_interval_ms)
            print("observation replay verified within declared quantization bounds")
        finally:
            replay_conn.close()
    finally:
        ledger_conn.close()
    return {"verified": True}


def print_headless_view() -> None:
    banner("Diagnostic view model (headless rendering, same data the Tk GUI shows)")
    work = WORK_BASE / "run_a" / LEDGER_FILENAME
    print(render_headless(str(work)))


def main() -> int:
    mp.set_start_method("spawn", force=True)
    WORK_BASE.mkdir(parents=True, exist_ok=True)
    summary = {}
    try:
        summary["reproducibility"] = scenario_reproducibility()
        summary["interrupt_recovery"] = scenario_interrupt_recovery()
        summary["replay_diff"] = scenario_replay_diff()
        summary["observation_replay"] = scenario_observation_replay()
        print_headless_view()
    except ChaosError as exc:
        print("UNEXPECTED business error:", json.dumps(exc.to_dict(), indent=2))
        return 1
    except AssertionError as exc:
        print(f"ASSERTION FAILED: {exc}")
        return 2
    banner("ALL DEMO SCENARIOS PASSED")
    (WORK_BASE / "demo_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8"
    )
    print(f"summary written to {WORK_BASE / 'demo_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
