#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
mkdir -p artifacts

python3 -m compileall -q src
python3 -m ins_diag validate \
  --config examples/config.yaml \
  --output artifacts/config-validation.json >/dev/null

python3 - <<'PY'
import json
from pathlib import Path
value = json.loads(Path("artifacts/config-validation.json").read_text(encoding="utf-8"))
Path("artifacts/config-validation.json").write_text(
    json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    encoding="utf-8",
)
PY

set +e
python3 -m pytest > artifacts/pytest-output.txt 2>&1
pytest_exit=$?
set -e

RUN_STATE="$(mktemp -d /tmp/ins-diag-acceptance.XXXXXX)"
python3 -m ins_diag run \
  --config examples/config.yaml \
  --state-dir "$RUN_STATE" \
  --output artifacts/example-report.json >/dev/null

python3 -m ins_diag verify \
  --config examples/config.yaml \
  --output artifacts/verify-report.json >/dev/null

python3 -m ins_diag faults \
  --root examples/faults \
  --output artifacts/fault-report.json >/dev/null

INS_DIAG_REPORT_DIR=artifacts bash scripts/crash_experiment.sh \
  > artifacts/crash-output.txt

python3 - "$pytest_exit" <<'PY'
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy
import yaml

pytest_exit = int(sys.argv[1])
root = Path("artifacts")
example = json.loads((root / "example-report.json").read_text(encoding="utf-8"))
verify = json.loads((root / "verify-report.json").read_text(encoding="utf-8"))
faults = json.loads((root / "fault-report.json").read_text(encoding="utf-8"))
crash = json.loads((root / "crash-report.json").read_text(encoding="utf-8"))
pytest_lines = (root / "pytest-output.txt").read_text(encoding="utf-8").strip().splitlines()
summary = {
    "schema_version": 1,
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "environment": {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "pyyaml": yaml.__version__,
    },
    "pytest": {
        "exit_code": pytest_exit,
        "last_line": pytest_lines[-1] if pytest_lines else None,
    },
    "default_run": {
        "status": example["status"],
        "window_count": example["data"]["window_count"],
        "cluster_count": len(example["clusters"]["centroids_units"]),
        "sse_units": example["clusters"]["sse_units"],
        "boundary_windows": example["boundary_windows"],
        "config_hash": example["config"]["hash"],
        "data_fingerprint": example["data"]["fingerprint"],
        "snapshot_version": example["snapshot"]["commit"]["version"],
        "cache_reused": example["cache"]["reused_count"],
        "cache_recomputed": example["cache"]["recomputed_count"],
    },
    "incremental_vs_full": verify,
    "fault_injection": faults,
    "crash_recovery": crash,
}
status = (
    pytest_exit == 0
    and verify["status"] == "ok"
    and faults["status"] == "ok"
    and crash["status"] == "ok"
    and example["status"] == "ok"
)
summary["status"] = "ok" if status else "failed"
(root / "validation-summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
raise SystemExit(0 if status else 1)
PY

