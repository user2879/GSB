import json
import os
import subprocess
import sys

from chaos import recover, timeline
from chaos.structlog import StructLog

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _cli(args, cwd=REPO):
    env = dict(os.environ, PYTHONPATH=REPO)
    return subprocess.run([sys.executable, "-m", "chaos.cli"] + args,
                          cwd=cwd, env=env, capture_output=True, text=True,
                          timeout=120)


def _make_ledger(tmp_path, resolved_spec):
    entries = timeline.generate(resolved_spec)
    tdigest = timeline.timeline_digest(entries)
    log = StructLog(str(tmp_path / "run.jsonl"))
    ledger, _ = recover.reconcile(resolved_spec, tdigest,
                                  str(tmp_path / "l.db"),
                                  str(tmp_path / "snap"), log)
    ledger.close()
    return entries


def _spec_file(tmp_path, resolved_spec):
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(resolved_spec.to_dict()))
    return str(path)


def test_fault_corrupt_spec_reports_e_spec(tmp_path, resolved_spec):
    spec = _spec_file(tmp_path, resolved_spec)
    _make_ledger(tmp_path, resolved_spec)
    out = _cli(["fault", "corrupt-spec", "--spec", spec,
                "--ledger", str(tmp_path / "l.db"),
                "--log-dir", str(tmp_path / "logs")])
    assert json.loads(out.stdout)["expected_code"] == "E_SPEC"
    res = _cli(["verify", "--spec", spec, "--ledger", str(tmp_path / "l.db"),
                "--snapshot-dir", str(tmp_path / "snap"),
                "--log-dir", str(tmp_path / "logs")])
    assert res.returncode == 2
    assert json.loads(res.stdout)["code"] == "E_SPEC"


def test_fault_seed_mismatch_reports_e_seed(tmp_path, resolved_spec):
    spec = _spec_file(tmp_path, resolved_spec)
    _make_ledger(tmp_path, resolved_spec)
    _cli(["fault", "seed-mismatch", "--spec", spec,
          "--ledger", str(tmp_path / "l.db"),
          "--log-dir", str(tmp_path / "logs")])
    res = _cli(["verify", "--spec", spec, "--ledger", str(tmp_path / "l.db"),
                "--snapshot-dir", str(tmp_path / "snap"),
                "--log-dir", str(tmp_path / "logs")])
    assert res.returncode == 3
    assert json.loads(res.stdout)["code"] == "E_SEED"


def test_fault_ledger_digest_reports_e_snapshot(tmp_path, resolved_spec):
    spec = _spec_file(tmp_path, resolved_spec)
    _make_ledger(tmp_path, resolved_spec)
    _cli(["fault", "ledger-digest", "--spec", spec,
          "--ledger", str(tmp_path / "l.db"),
          "--log-dir", str(tmp_path / "logs")])
    res = _cli(["verify", "--spec", spec, "--ledger", str(tmp_path / "l.db"),
                "--snapshot-dir", str(tmp_path / "snap"),
                "--log-dir", str(tmp_path / "logs")])
    assert res.returncode == 4
    assert json.loads(res.stdout)["code"] == "E_SNAPSHOT"


def test_faults_write_structured_log(tmp_path, resolved_spec):
    spec = _spec_file(tmp_path, resolved_spec)
    _make_ledger(tmp_path, resolved_spec)
    logs = tmp_path / "logs"
    _cli(["fault", "corrupt-spec", "--spec", spec,
          "--ledger", str(tmp_path / "l.db"), "--log-dir", str(logs)])
    _cli(["verify", "--spec", spec, "--ledger", str(tmp_path / "l.db"),
          "--snapshot-dir", str(tmp_path / "snap"), "--log-dir", str(logs)])
    lines = (logs / "run.jsonl").read_text().strip().splitlines()
    events = [json.loads(l) for l in lines]
    kinds = [(e.get("event"), e.get("code") or e.get("expected_code"))
             for e in events]
    assert ("fault_injected", "E_SPEC") in kinds
    assert ("error", "E_SPEC") in kinds
