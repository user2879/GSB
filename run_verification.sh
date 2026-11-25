#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

python3 - <<'PY'
import glob
import os
for pattern in ("ledger/rdiq.sqlite*", "ledger/crash.sqlite*", "ledger/backups/*"):
    for path in glob.glob(pattern):
        if os.path.isfile(path):
            os.chmod(path, 0o644)
            os.remove(path)
PY

python3 -m unittest discover -s tests -v

python3 -m rdiq.cli run \
  --sample-dir samples/base \
  --output data/baseline.json \
  --ledger-db ledger/rdiq.sqlite \
  --backup-dir ledger/backups

for edit in weights rules precision channel-weight; do
  python3 - <<PY
import glob, os
for pattern in ("ledger/rdiq.sqlite*", "ledger/backups/*"):
    for path in glob.glob(pattern):
        if os.path.isfile(path):
            os.chmod(path, 0o644)
            os.remove(path)
PY
  python3 -m rdiq.cli edit-recompute \
    --base-dir samples/base \
    --edited-dir "samples/edit/${edit}" \
    --output "data/edit-${edit}.json" \
    --ledger-db ledger/rdiq.sqlite \
    --backup-dir ledger/backups
done

python3 - <<'PY'
import glob
import os
for pattern in ("ledger/crash.sqlite*", "ledger/backups/*"):
    for path in glob.glob(pattern):
        if os.path.isfile(path):
            os.chmod(path, 0o644)
            os.remove(path)
PY

python3 -m rdiq.cli crash \
  --sample-dir samples/base \
  --output data/crash-open-transaction.json \
  --ledger-db ledger/crash.sqlite \
  --backup-dir ledger/backups

python3 -m rdiq.cli recover \
  --sample-dir samples/base \
  --output data/recovered.json \
  --recovery-output data/recovery.json \
  --ledger-db ledger/crash.sqlite \
  --backup-dir ledger/backups

python3 tools/verify_byte_identity.py
