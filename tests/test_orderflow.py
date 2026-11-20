import json
import random
from pathlib import Path

import pytest

from orderflow.adjudicate import adjudicate, adjudication_queue
from orderflow.cli import main
from orderflow.generator import gen_samples
from orderflow.model import (
    E_BUDGET_DENIED,
    E_CELL_BOUNDS,
    E_DUPLICATE_ORDER,
    E_LATE_ORDER,
    E_ORDER_OVERRIDDEN,
    E_ORDER_TYPE,
    E_SEQ_GAP,
    E_SUPPLY_OVERFLOW,
    E_TIE_UNRESOLVED,
    Config,
)
from orderflow.parser import parse_lines
from orderflow.recompute import verify_workspace
from orderflow.scheduler import MatchTask, schedule_matches
from orderflow.storage import CrashAbort, MatchStore

CFG = Config()


def _line(tick, cmdr, unit, otype, cell, prio, lat, seq):
    return json.dumps({
        "tick": tick, "commander": cmdr, "unit_id": unit, "order_type": otype,
        "target_cell": cell, "priority": prio, "latency_ms": lat, "seq": seq,
    })


def _sample_lines(tmp_path, match_id):
    gen_samples(tmp_path / "samples")
    return (tmp_path / "samples" / f"{match_id}.jsonl").read_text().splitlines()


def test_parse_error_codes():
    lines = [
        _line(0, "CMD_A", "U1", "MOVE", [1, 1], 1, 0, 0),
        _line(1, "CMD_A", "U1", "FLY", [1, 1], 1, 0, 1),          # E_ORDER_TYPE
        _line(2, "CMD_B", "U1", "MOVE", [99, 1], 1, 0, 0),        # E_CELL_BOUNDS
        _line(3, "CMD_C", "U1", "MOVE", [1, 1], 1, 0, 5),         # E_SEQ_GAP (expected 0)
        _line(4, "CMD_A", "U1", "MOVE", [2, 2], 1, 0, 0),         # E_SEQ_GAP (conflicting dup seq)
    ]
    _orders, errors, _anomalies = parse_lines(lines, CFG)
    by_code = {}
    for err in errors:
        by_code.setdefault(err["code"], []).append(err)
    assert by_code[E_ORDER_TYPE][0]["line"] == 2
    assert by_code[E_ORDER_TYPE][0]["commander"] == "CMD_A"
    assert by_code[E_CELL_BOUNDS][0]["line"] == 3
    assert by_code[E_CELL_BOUNDS][0]["commander"] == "CMD_B"
    gap_lines = {e["line"] for e in by_code[E_SEQ_GAP]}
    assert gap_lines == {4, 5}
    assert all(e["commander"] for e in by_code[E_SEQ_GAP])


def test_late_and_duplicate_delivery():
    late = _line(0, "CMD_A", "U1", "MOVE", [1, 1], 1, 999, 0)
    dup = _line(1, "CMD_B", "U2", "MOVE", [2, 2], 1, 0, 0)
    orders, errors, anomalies = parse_lines([late, dup, dup], CFG)
    assert not errors
    codes = [a["code"] for a in anomalies]
    assert codes.count(E_LATE_ORDER) == 1
    assert codes.count(E_DUPLICATE_ORDER) == 1
    queue = adjudication_queue(orders, CFG)
    assert len(queue) == 1  # late order expired, duplicate deduped
    assert queue[0].commander == "CMD_B"


def test_shuffle_determinism(tmp_path):
    lines = _sample_lines(tmp_path, "M01_NORMAL")
    orders, errors, anomalies = parse_lines(lines, CFG)
    assert not errors
    base = adjudicate("M01_NORMAL", orders, anomalies, CFG)
    for seed in range(5):
        shuffled = lines[:]
        random.Random(seed).shuffle(shuffled)
        o2, e2, a2 = parse_lines(shuffled, CFG)
        assert e2 == errors
        assert a2 == anomalies
        res = adjudicate("M01_NORMAL", o2, a2, CFG)
        assert [s["order_id"] for s in res["steps"]] == [s["order_id"] for s in base["steps"]]
        assert [s["hash"] for s in res["steps"]] == [s["hash"] for s in base["steps"]]
        assert res["final_hash"] == base["final_hash"]


def test_tie_and_override(tmp_path):
    lines = _sample_lines(tmp_path, "M03_TIE")
    orders, errors, anomalies = parse_lines(lines, CFG)
    assert not errors
    res = adjudicate("M03_TIE", orders, anomalies, CFG)
    ties = [a for a in res["anomalies"] if a["code"] == E_TIE_UNRESOLVED]
    overs = [a for a in res["anomalies"] if a["code"] == E_ORDER_OVERRIDDEN]
    assert len(ties) == 1
    assert ties[0]["winner"] == "CMD_A#0"
    assert len(overs) == 1
    assert overs[0]["loser"] == "CMD_B#0"
    assert overs[0]["winner"] == "CMD_A#0"
    assert overs[0]["winner_time"] == 600
    step1 = res["steps"][1]
    assert step1["order_id"] == "CMD_B#0" and step1["status"] == "overridden"


def test_supply_overflow(tmp_path):
    lines = _sample_lines(tmp_path, "M04_SUPPLY")
    orders, errors, anomalies = parse_lines(lines, CFG)
    assert not errors
    res = adjudicate("M04_SUPPLY", orders, anomalies, CFG)
    overflows = [a for a in res["anomalies"] if a["code"] == E_SUPPLY_OVERFLOW]
    assert overflows, "expected overflow anomalies"
    firsts = [a for a in overflows if a["first"]]
    assert len(firsts) == 1
    assert res["first_overflow"] == {
        "step": firsts[0]["step"], "unit_id": "U8", "balance": 220,
    }
    units = {a["unit_id"] for a in overflows}
    assert units == {"U8", "U9"}


def test_ledger_snapshot_correspondence(tmp_path):
    lines = _sample_lines(tmp_path, "M04_SUPPLY")
    orders, _errors, anomalies = parse_lines(lines, CFG)
    store = MatchStore(tmp_path, "M04_SUPPLY")
    res = adjudicate("M04_SUPPLY", orders, anomalies, CFG, store=store)
    ledger_lines = (store.dir / "ledger.jsonl").read_text().splitlines()
    assert len(ledger_lines) == len(res["steps"])
    for line, step in zip(ledger_lines, res["steps"]):
        entry = json.loads(line)
        assert entry["snapshot_hash"] == step["hash"]
        assert entry["step"] == step["step"]


def _tasks(tmp_path):
    metas = gen_samples(tmp_path / "samples")
    return [
        MatchTask(m["match_id"], m["sched_priority"], m["est_mem_mb"], p)
        for m, p in metas
    ]


def test_budget_denied_and_downgrade(tmp_path):
    out = schedule_matches(_tasks(tmp_path), CFG, tmp_path, max_threads=8, mem_budget_mb=40)
    denied = out["denied"]
    assert [d["match_id"] for d in denied] == ["M05_DENIED"]
    assert denied[0]["code"] == E_BUDGET_DENIED
    assert out["schedule"]["downgrades"] == [
        {"reason": "memory_budget", "before": 8, "after": 4}
    ]
    assert out["schedule"]["workers"] == 4
    ids = [r["match_id"] for r in out["results"]]
    assert ids == sorted(ids)  # merged by match id, not completion order


def test_thread_count_invariance(tmp_path):
    results = {}
    for threads in (1, 2, 4):
        ws = tmp_path / f"w{threads}"
        out = schedule_matches(_tasks(ws), CFG, ws, max_threads=threads, mem_budget_mb=40)
        results[threads] = {
            r["match_id"]: (r["final_hash"], r["anomalies"]) for r in out["results"]
        }
    assert results[1] == results[2] == results[4]


def test_crash_and_recovery(tmp_path):
    crash_ws = tmp_path / "crash"
    lines = _sample_lines(crash_ws, "M01_NORMAL")
    orders, _errors, anomalies = parse_lines(lines, CFG)
    clean = adjudicate("M01_NORMAL", orders, anomalies, CFG)

    store = MatchStore(crash_ws, "M01_NORMAL")
    with pytest.raises(CrashAbort):
        adjudicate("M01_NORMAL", orders, anomalies, CFG, store=store, crash_after=5)
    manifest = store.read_manifest()
    assert manifest["committed"] == 5
    assert list(store.snap_dir.glob("*.tmp"))  # uncommitted partial artifact left behind

    resumed = adjudicate("M01_NORMAL", orders, anomalies, CFG, store=store)
    assert resumed["final_hash"] == clean["final_hash"]
    assert store.committed_hashes() == [s["hash"] for s in clean["steps"]]
    check = verify_workspace(crash_ws, CFG)
    assert check["M01_NORMAL"]["status"] == "OK"


def test_cli_end_to_end_byte_identical(tmp_path):
    ws1, ws2 = tmp_path / "ws1", tmp_path / "ws2"
    assert main(["run", "--workspace", str(ws1)]) == 0
    assert main(["run", "--workspace", str(ws2)]) == 0
    assert (ws1 / "report.json").read_bytes() == (ws2 / "report.json").read_bytes()
    assert (ws1 / "schedule.json").read_bytes() == (ws2 / "schedule.json").read_bytes()

    report = json.loads((ws1 / "report.json").read_text())
    statuses = {m["match_id"]: m["status"] for m in report["matches"]}
    assert statuses["M05_DENIED"] == E_BUDGET_DENIED
    assert statuses["M01_NORMAL"] == "OK"
    assert all(c["status"] == "OK" for c in report["recompute"].values())
    assert (ws1 / "CLOSED").exists()


def test_cli_crash_then_resume_matches_clean_run(tmp_path):
    clean = tmp_path / "clean"
    crashed = tmp_path / "crashed"
    assert main(["run", "--workspace", str(clean)]) == 0
    rc = main([
        "run", "--workspace", str(crashed),
        "--crash-match", "M01_NORMAL", "--crash-after", "7",
    ])
    assert rc == 3
    assert not (crashed / "CLOSED").exists()
    assert not (crashed / "report.json").exists()
    assert main(["run", "--workspace", str(crashed)]) == 0
    assert (crashed / "report.json").read_bytes() == (clean / "report.json").read_bytes()
    assert main(["recompute", "--workspace", str(crashed)]) == 0
