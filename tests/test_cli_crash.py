from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from qosfdiag.generator import DeterministicUdpGenerator
from qosfdiag.ledger import DecisionLedger
from qosfdiag.policy import load_policy
from qosfdiag.samples import three_stage_generator, three_stage_policy


ROOT = Path(__file__).resolve().parents[1]


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    return env


def test_cli_hard_kill_then_restart(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.sqlite"
    base = [
        sys.executable,
        "-m",
        "qosfdiag",
        "run",
        "--policy",
        str(ROOT / "examples" / "data" / "three_stage_policy.json"),
        "--generator",
        str(ROOT / "examples" / "data" / "three_stage_generator.json"),
        "--scale",
        "3",
        "--ledger",
        str(ledger_path),
        "--in-memory",
        "--json",
    ]
    crashed = subprocess.run(
        base + ["--crash-after-marker-batch", "1"],
        env=_env(),
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert crashed.returncode == 137
    assert crashed.stdout == ""
    resumed = subprocess.run(
        base,
        env=_env(),
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert resumed.returncode == 0, resumed.stderr
    payload = json.loads(resumed.stdout)
    assert payload["summary"]["final_released_bytes"] == 21771
    assert payload["recovery"]["requires_rerun"] is True
    assert payload["recovery"]["unknown_packet_ids"] == [
        "p00000004",
        "p00000005",
        "p00000006",
        "p00000007",
    ]
    policy = load_policy(three_stage_policy())
    generator = DeterministicUdpGenerator(three_stage_generator())
    with DecisionLedger(ledger_path) as ledger:
        ledger.initialize(policy=policy, scale=3, generator=generator)
        assert ledger.progress().cursor_seq == 40
        assert ledger.unknown_packets(unresolved_only=True) == ()
        assert len(ledger.decisions()) == 184
