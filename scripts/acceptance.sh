#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  . .venv/bin/activate
fi

mkdir -p artifacts

section() {
  printf '\n===== %s =====\n' "$1"
}

section "environment"
python --version
python -c 'import numpy as np; print("numpy", np.__version__)'

section "generate-and-validate"
python -m quota_lab.cli init-demo --result artifacts/init-demo.json
python -m quota_lab.cli validate \
  --declaration configs/declaration.json \
  --shards configs/shards.json \
  --capacity configs/capacity.json \
  --events samples/event_stream.jsonl \
  --result artifacts/validate.json

section "e-spec-first-field"
python - <<'PY'
import json
from quota_lab.fixtures import demo_declaration, demo_shards
from quota_lab.spec import parse_session, SpecError
decl = demo_declaration()
decl["nodes"][0]["port"] = 70000
try:
    parse_session(decl, demo_shards())
except SpecError as exc:
    payload = exc.to_dict()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    assert payload["error"]["code"] == "E_SPEC"
    assert payload["error"]["first_error_field"] == "declaration.nodes[0].port"
else:
    raise SystemExit("invalid port was accepted")
PY

section "network-e-port-e-timeout"
python -m quota_lab.cli verify-network --result artifacts/network-result.json

section "unit-and-integration-tests"
python -m unittest discover -s tests -v 2>&1 | tee artifacts/unittest-results.txt

section "normal-scheduler-fixed-point-boundary"
python -m quota_lab.cli run \
  --declaration configs/declaration.json \
  --shards configs/shards.json \
  --capacity configs/capacity.json \
  --events samples/event_stream.jsonl \
  --ledger artifacts/run.sqlite3 \
  --snapshot artifacts/run.snapshot.json \
  --result artifacts/run-result.json \
  --reset-ledger

section "blank-rebuild-consistency"
python -m quota_lab.cli rebuild-check \
  --declaration configs/declaration.json \
  --shards configs/shards.json \
  --capacity configs/capacity.json \
  --events samples/event_stream.jsonl \
  --ledger artifacts/run.sqlite3 \
  --result artifacts/rebuild-check.json

section "capacity-version-evolution"
python -m quota_lab.cli make-evolution-spec
python -m quota_lab.cli evolve \
  --old-declaration configs/declaration.json \
  --old-shards configs/shards.json \
  --old-capacity configs/capacity.json \
  --new-declaration configs/declaration.v2.json \
  --new-shards configs/shards.json \
  --new-capacity configs/capacity.v2.json \
  --events samples/event_stream.jsonl \
  --old-ledger artifacts/run.sqlite3 \
  --new-ledger artifacts/evolved.sqlite3 \
  --combined-spec artifacts/evolved-runtime-spec.json \
  --log-dir artifacts/logs-evolve \
  --result artifacts/evolve-result.json \
  --reset-new-ledger

section "force-kill-and-recovery"
set +e
python -c 'from pathlib import Path; Path("artifacts/crash-result.json").unlink(missing_ok=True)'
python -m quota_lab.cli run \
  --declaration configs/declaration.crash.json \
  --shards configs/shards.json \
  --capacity configs/capacity.json \
  --events samples/event_stream.jsonl \
  --ledger artifacts/crash-ledger.sqlite3 \
  --result artifacts/crash-result.json \
  --log-dir artifacts/logs-crash \
  --reset-ledger
crash_rc=$?
set -e
echo "force_kill_exit_code=$crash_rc (137 expected)"
test "$crash_rc" -eq 137
test ! -e artifacts/crash-result.json || {
  echo "unexpected clean crash result file"
  exit 1
}

python -m quota_lab.cli run \
  --declaration configs/declaration.crash.json \
  --shards configs/shards.json \
  --capacity configs/capacity.json \
  --events samples/event_stream.jsonl \
  --ledger artifacts/crash-ledger.sqlite3 \
  --snapshot artifacts/crash.snapshot.json \
  --result artifacts/recovery-result.json \
  --log-dir artifacts/logs-recovery \
  --resume

python -m quota_lab.cli rebuild-check \
  --declaration configs/declaration.crash.json \
  --shards configs/shards.json \
  --capacity configs/capacity.json \
  --events samples/event_stream.jsonl \
  --ledger artifacts/crash-ledger.sqlite3 \
  --result artifacts/recovery-rebuild.json

section "snapshot-version-replacement"
python - <<'PY'
from quota_lab.ledger import replace_json_snapshot_if_stale
first = replace_json_snapshot_if_stale(
    "artifacts/snapshot-replace-demo.json",
    expected_version=7,
    spec_digest="old-digest",
    fresh_payload={"whole": "old"},
)
second = replace_json_snapshot_if_stale(
    "artifacts/snapshot-replace-demo.json",
    expected_version=8,
    spec_digest="new-digest",
    fresh_payload={"whole": "new"},
)
print({"first": first, "second": second})
assert first["replaced"] and second["replaced"]
PY

section "summary"
python - <<'PY'
import json
from pathlib import Path

def load(name):
    return json.loads(Path(name).read_text(encoding="utf-8"))

run = load("artifacts/run-result.json")
evolve = load("artifacts/evolve-result.json")
recovery = load("artifacts/recovery-result.json")
rebuild = load("artifacts/rebuild-check.json")
recovery_rebuild = load("artifacts/recovery-rebuild.json")
network = load("artifacts/network-result.json")
summary = {
    "run_ok": run["ok"],
    "completed": run["completed_shard_ids"],
    "budget_peak_q": run["constraint_check"]["peak_active_demand_q"],
    "budget_q": run["constraint_check"]["global_budget_q"],
    "budget_never_exceeded": run["constraint_check"]["budget_never_exceeded_exact_integer"],
    "dominant_error_level": run["error_report"]["dominant_level"],
    "error_shares_bp": run["error_report"]["share_basis_points_floor"],
    "boundary_decisions": [
        {
            "decision_id": item["decision_id"],
            "released": item["released_shard_id"],
            "budget_gap_q": item["budget"]["remaining_minus_demand_q"],
            "bound_q": item["budget"]["synthetic_error_bound_q"],
            "budget_indistinguishable": item["budget"]["indistinguishable"],
            "water_difference_q": None if not item.get("watermark_pair") else item["watermark_pair"]["difference_q"],
            "water_pair_bound_q": None if not item.get("watermark_pair") else item["watermark_pair"]["pair_error_bound_q"],
            "water_indistinguishable": None if not item.get("watermark_pair") else item["watermark_pair"]["indistinguishable"],
        }
        for item in run["decisions"]
        if item.get("released_shard_id") and item.get("error_affected")
    ],
    "network_ok": network["ok"],
    "e_port": network["port_conflict_error"]["code"],
    "e_timeout": network["timeout_error"]["code"],
    "rebuild_ok": rebuild["ok"],
    "evolution_ok": evolve["ok"],
    "invalid_set": evolve["invalidation"]["invalid_shard_ids"],
    "unchanged_set": evolve["invalidation"]["unchanged_shard_ids"],
    "byte_preservation_ok": evolve["byte_preservation_after_run"]["ok"],
    "recovery_ok": recovery["ok"],
    "reclaimed": [item["shard_id"] for item in recovery["recovered"]],
    "recovery_rebuild_ok": recovery_rebuild["ok"],
}
Path("artifacts/acceptance-summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
assert summary["run_ok"]
assert summary["budget_never_exceeded"]
assert summary["rebuild_ok"]
assert summary["evolution_ok"]
assert summary["byte_preservation_ok"]
assert summary["recovery_ok"]
assert summary["recovery_rebuild_ok"]
assert len(summary["boundary_decisions"]) >= 1
PY

section "acceptance-complete"
