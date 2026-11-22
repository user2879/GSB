import json

import pytest

from gpu_diag.cli import main
from gpu_diag.samplegen import inject_fault, write_samples
from gpu_diag.ui.app import read_cached_view, render_headless


def test_fault_injection_codes(sample_dir, state_dir, tmp_path, monkeypatch):
    log = tmp_path / "events.jsonl"
    expected = {
        "memory": "E_MEMORY", "quota": "E_QUOTA",
        "cycle": "E_CYCLE", "ledger": "E_LEDGER",
    }
    for fault, code in expected.items():
        write_samples(sample_dir)
        argv = ["--samples", str(sample_dir), "--state", str(state_dir),
                "--log", str(log), "inject", fault]
        assert main(argv) == 2
    lines = [json.loads(line) for line in log.read_text().splitlines()]
    codes = [line["code"] for line in lines]
    assert set(expected.values()).issubset(set(codes))


def test_injection_only_changes_inputs(sample_dir):
    write_samples(sample_dir)
    before = (sample_dir / "pools.json").read_bytes()
    inject_fault(sample_dir, "quota")
    after = (sample_dir / "pools.json").read_bytes()
    assert before != after


def test_cached_ui_view_and_headless_render(sample_dir, state_dir):
    from gpu_diag.session import run_session
    pools = sample_dir / "pools.json"
    tasks = sample_dir / "tasks.json"
    params = sample_dir / "params.json"
    run_session(pools, tasks, params, state_dir)
    view = read_cached_view(state_dir)
    assert view["rounds"] > 0
    assert view["pools"]
    text = render_headless(view)
    assert "GPU Diagnostic Fast View" in text
    assert "__global__" not in text or "GLOBAL" in text


def test_run_command_exit_zero(sample_dir, state_dir, tmp_path):
    argv = ["--samples", str(sample_dir), "--state", str(state_dir),
            "--log", str(tmp_path / "l.jsonl"), "run"]
    assert main(argv) == 0


def test_recompute_command_exit_zero(sample_dir, state_dir, tmp_path):
    from gpu_diag.session import run_session
    run_session(sample_dir / "pools.json", sample_dir / "tasks.json",
                sample_dir / "params.json", state_dir)
    argv = ["--samples", str(sample_dir), "--state", str(state_dir),
            "--log", str(tmp_path / "l.jsonl"), "recompute"]
    assert main(argv) == 0
