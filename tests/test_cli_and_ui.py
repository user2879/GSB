from __future__ import annotations

import json
import subprocess
import sys
import tkinter as tk
from pathlib import Path

import pytest

from pageagg.sample import generate_sample
from pageagg.service import aggregate_database
from pageagg.ui import Dashboard, load_dashboard_data


ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "pageagg", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )


@pytest.mark.parametrize(
    ("fault", "expected_code", "exit_code"),
    [
        ("catalog_truncate", "E_PAGE", 10),
        ("nullmap_misalign", "E_NULLMAP", 11),
    ],
)
def test_cli_fault_codes_and_structured_log(tmp_path, fault, expected_code, exit_code) -> None:
    sample = generate_sample(tmp_path / "sample", force=True)
    fault_result = run_cli(
        "inject",
        "--catalog",
        sample["catalog_path"],
        "--out",
        str(tmp_path / "fault"),
        "--kind",
        fault,
    )
    assert fault_result.returncode == 0, fault_result.stderr
    log_path = tmp_path / "business.jsonl"
    result = run_cli(
        "aggregate",
        "--catalog",
        str(tmp_path / "fault" / "catalog.json"),
        "--db",
        str(tmp_path / "fault.db"),
        "--budget",
        "4",
        "--log-path",
        str(log_path),
    )
    assert result.returncode == exit_code
    response = json.loads(result.stdout)
    assert response["code"] == expected_code
    record = json.loads(log_path.read_text(encoding="utf-8").splitlines()[-1])
    assert record["code"] == expected_code
    assert record["category"] == (
        "E_PAGE" if expected_code == "E_NULLMAP" else expected_code
    )


def test_cli_zero_budget_code(tmp_path) -> None:
    sample = generate_sample(tmp_path / "sample", force=True)
    result = run_cli(
        "aggregate",
        "--catalog",
        sample["catalog_path"],
        "--db",
        str(tmp_path / "zero.db"),
        "--budget",
        "0",
    )
    assert result.returncode == 12
    response = json.loads(result.stdout)
    assert response["code"] == "E_BUDGET"
    assert response["details"]["minimum_feasible_slots"] == 1


def test_compare_routes_cli_writes_identical_files(tmp_path) -> None:
    sample = generate_sample(tmp_path / "sample", force=True)
    out_dir = tmp_path / "compare"
    result = run_cli(
        "compare-routes",
        "--catalog",
        sample["catalog_path"],
        "--budget",
        "4",
        "--out-dir",
        str(out_dir),
    )
    assert result.returncode == 0
    response = json.loads(result.stdout)
    assert response["equivalence"]["result_bytes_equal"] is True
    assert (out_dir / "resident.result.json").read_bytes() == (
        out_dir / "rescan.result.json"
    ).read_bytes()
    assert (out_dir / "eviction.events.json").read_bytes() == (
        out_dir / "eviction.events.from-rescan.json"
    ).read_bytes()


def test_dashboard_data_and_window(tmp_path) -> None:
    sample = generate_sample(tmp_path / "sample", force=True)
    db = tmp_path / "ui.db"
    aggregate_database(sample["catalog_path"], db, 4)
    data = load_dashboard_data(db)
    assert len(data["page_outcomes"]) == 12
    assert len(data["evictions"]) == 5
    assert len(data["result"]["groups"]) == 5
    root = tk.Tk()
    root.withdraw()
    try:
        Dashboard(root, data, error_code="NONE")
        root.update_idletasks()
        root.update()
    finally:
        root.destroy()
