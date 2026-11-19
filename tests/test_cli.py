import json
import subprocess
import sys

import pytest


def run_cli(*args):
    return subprocess.run([sys.executable, "-m", "probegi", *args],
                          capture_output=True, text=True, timeout=600)


def test_run_all_and_verify(tmp_path):
    ws = str(tmp_path / "out")
    proc = run_cli("run-all", "--workspace", ws, "--seed", "77",
                   "--directions", "64")
    assert proc.returncode == 0, proc.stderr
    lines = [json.loads(l) for l in proc.stdout.strip().splitlines()]
    steps = [l["step"] for l in lines]
    assert steps.count("generate_probes") == 3
    assert steps.count("project_sh") == 9
    assert steps.count("select_strategy") == 9
    assert steps.count("diagnose") == 9
    assert "preview" in steps and "snapshot" in steps
    chosen = {(l["grid_n"], l["order"]): l["strategy"] for l in lines
              if l["step"] == "select_strategy"}
    assert chosen[(32, 3)] == "cluster"
    assert chosen[(16, 3)] == "nearest_gradient"
    assert chosen[(8, 1)] == "full"
    # Determinism: a second run must reproduce every content hash.
    proc2 = run_cli("run-all", "--workspace", str(tmp_path / "out2"),
                    "--seed", "77", "--directions", "64")
    assert proc2.returncode == 0, proc2.stderr
    assert proc.stdout == proc2.stdout
    # Independent recomputation from the committed snapshot.
    proc3 = run_cli("verify", "--workspace", ws, "--seed", "77")
    assert proc3.returncode == 0, proc3.stderr
    last = json.loads(proc3.stdout.strip().splitlines()[-1])
    assert last["verified"] is True


def test_error_code_on_stderr(tmp_path):
    proc = run_cli("run-all", "--workspace", str(tmp_path / "bad"),
                   "--directions", "8")
    assert proc.returncode == 2
    err = json.loads(proc.stderr.strip())
    assert err["error"]["code"] == "E_SAMPLE_DIR"
